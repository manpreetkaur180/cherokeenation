import json
import time
import requests
from concurrent.futures import TimeoutError
from . import config
from google.cloud import pubsub_v1
from . import vector_store
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler

import concurrent.futures

data = config.proj_config()
logger = config.setup_logger()

GCP_PROJECT_ID = data.get("GCP_PROJECT_ID")
PUB_SUB_SUBSCRIPTION_ID = data.get("PUB_SUB_SUBSCRIPTION_ID")

PROXIES = {
    "http": data.get("PROXY_HTTP"),
    "https": data.get("PROXY_HTTPS"),
}
HEADERS = {
    'User-Agent': data.get("USER_AGENT")
}

allowed_prefixes_str = data.get("URL_PREFIXES_EMBEDDING", "")
ALLOWED_URL_PREFIXES = tuple([prefix.strip() for prefix in allowed_prefixes_str.split(',') if prefix.strip()])

try:
   vector_store_instance = vector_store.VectorStore(
        project_id=data.get("GCP_PROJECT_ID"),
        region=data.get('GCP_REGION'),
        bucket_name=data.get("GCS_BUCKET_NAME"),
        corpus_display_name=data.get("CORPUS_DISPLAY_NAME")
    )
except ValueError as e:
    logger.critical(f"Failed to initialize VectorStore. Shutting down. Error: {e}")
    exit(1)

def message_processing_callback(message: pubsub_v1.subscriber.message.Message) -> None:
    """The callback function to process a single Pub/Sub message."""
    url = "N/A"
    try:
        message_data = message.data.decode('utf-8')
        data = json.loads(message_data)
        url = data.get('url')
        action = data.get('action', 'upsert')
        data_type = data.get('type', 'content')

        if not url:
            logger.warning(f"Message ID {message.message_id} is missing a 'url'. Discarding.")
            message.ack()
            return

        if not ALLOWED_URL_PREFIXES:
            logger.critical(f"URL_PREFIXES_EMBEDDING is not configured. Cannot validate URL: {url}. Nacking message.")
            message.nack()
            return

        if not url.startswith(ALLOWED_URL_PREFIXES):
            logger.warning(f"URL '{url}' is not in the allowed list of prefixes. Discarding message {message.message_id}.")
            message.ack()
            return
        
        logger.info(f" Received valid message {message.message_id}. Processing URL: {url} ")

        if action == 'delete':
            logger.info(f"Received 'delete' message {message.message_id}. Processing URL: {url} ")
            vector_store_instance.delete_document_by_url(url)
        
        elif action == 'upsert':
            data_type = data.get('type', 'content')
            logger.info(f" Received 'upsert' message {message.message_id}. Processing URL: {url} (Type: {data_type}) ")
            
            if data_type == "media":
                response = requests.get(url, headers=HEADERS, proxies=PROXIES, verify=False, timeout=60)
                response.raise_for_status()
                vector_store_instance.upsert_pdf_with_llm_parser(response.content, url)
            else:
                vector_store_instance.upsert_scraped_url(url)
        else:
            logger.warning(f"Unknown action '{action}' in message. Discarding.")

        logger.info(f" Successfully processed '{action}' message for URL: {url}. Acknowledging. ")
        message.ack()

    except Exception as e:
        # logger.error(f"--- Failed to process message for URL: {data.get('url', 'N/A')}. Nacking for retry. Error: {e} ---", exc_info=True)
        logger.error(f" Failed to process message for URL: {data.get('url', 'N/A')}. Nacking for retry.")
        message.nack()
    
    finally:
        logger.info("Pausing for 1 second to respect rate limits...")
        time.sleep(1)


def main():
    """Starts the subscriber and listens for messages indefinitely."""
    if not PUB_SUB_SUBSCRIPTION_ID:
        logger.critical("FATAL: PUB_SUB_SUBSCRIPTION_ID environment variable is not set.")
        return
    
    flow_control_settings = pubsub_v1.types.FlowControl(max_messages=1)

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(GCP_PROJECT_ID, PUB_SUB_SUBSCRIPTION_ID)
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    scheduler = ThreadScheduler(executor=executor)
    
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=message_processing_callback, flow_control=flow_control_settings, scheduler=scheduler)
    logger.info(f"Listening for messages on {subscription_path}. (SEQUENTIAL MODE - 1 message at a time)")
    logger.info("Press Ctrl+C to stop the worker.")

    try:
        streaming_pull_future.result()
    except TimeoutError:
        streaming_pull_future.cancel()
        streaming_pull_future.result()
    except KeyboardInterrupt:
        logger.info("Subscriber stopped by user.")
        streaming_pull_future.cancel()
    finally:
        subscriber.close()
        logger.info("Pub/Sub subscriber has been shut down.")

if __name__ == "__main__":
    main()