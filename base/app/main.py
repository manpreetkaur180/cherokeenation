import re
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google.cloud import pubsub_v1
from . import vector_store
from . import pii_check
from . import config
from google import genai
from google import api_core
from . import services, utils
import threading
import queue

custom_publisher_retry = api_core.retry.Retry(
    initial=0.750,
    maximum=300.0,
    multiplier=1.45,
    deadline=900.0,
    predicate=api_core.retry.if_exception_type(
        api_core.exceptions.Aborted,
        api_core.exceptions.DeadlineExceeded,
        api_core.exceptions.ResourceExhausted,
        api_core.exceptions.ServiceUnavailable,
        api_core.exceptions.Unknown,
        api_core.exceptions.Cancelled,
    ),
)

logger = config.setup_logger()
data = config.proj_config()

WHITELISTED_DOMAINS_STR = data["WHITELISTED_DOMAINS"]
PROMPT_INJECTION_KEYWORDS = config.PROMPT_INJECTION_KEYWORDS
allowed_origins = [
    domain.strip().rstrip("/")
    for domain in WHITELISTED_DOMAINS_STR.split(",")
    if domain.strip()
]

try:
    vector_store_instance = vector_store.VectorStore(
        project_id=str(data["GCP_PROJECT_ID"]),
        region=str(data["GCP_REGION"]),
        bucket_name=str(data["GCS_BUCKET_NAME"]),
        corpus_display_name=str(data["CORPUS_DISPLAY_NAME"]),
    )
except ValueError as e:
    logger.critical(f"Failed to initialize VectorStore. Shutting down. Error: {e}")
    exit(1)

app = Flask(__name__)

# CORS setup
if allowed_origins:
    logger.info(f"CORS is configured for the following origins: {allowed_origins}")
    CORS(app, origins=allowed_origins, supports_credentials=True)
else:
    logger.warning("CORS is not configured. No whitelisted domains found.")

client = genai.Client(
    vertexai=True,
    project=data["GCP_PROJECT_ID"],
    location=data["GENAI_CLIENT"],
)

# @app.route("/update-embedding", methods=["POST"])
# def update_embedding():
#     """
#     Receives a payload to update a document. In production, it publishes to
#     Pub/Sub. In local dev mode, it processes the task synchronously.
#     """
#     logger.info("Received request on /update-embedding endpoint.")
#     query_data = request.get_json()
#     if not query_data:
#         return jsonify({"error": "Missing JSON payload."}), 400

#     final_url = ""
#     data_type = query_data.get('type')
#     if data_type == "content":
#         urls = query_data.get('urls')
#         if not urls or "en-us" not in urls:
#             return jsonify({"msg": "For 'content' type, 'urls' must contain an 'en-us' key."}), 400
#         final_url = urls["en-us"]
#     elif data_type == "media":
#         media_url = query_data.get('mediaUrl')
#         if not media_url or not media_url.lower().endswith(".pdf"):
#             return jsonify({"msg": "For 'media' type, 'mediaUrl' must be a valid URL."}), 400
#         final_url = media_url
#     else:
#         return jsonify({"msg": f"Invalid or missing 'type': '{data_type}'."}), 400

#     allowed_prefixes_str = data["URL_PREFIXES_EMBEDDING"]
#     allowed_url_prefixes = tuple([prefix.strip() for prefix in allowed_prefixes_str.split(',') if prefix.strip()])

#     if not allowed_url_prefixes:
#         logger.critical("SECURITY ALERT: URL_PREFIXES_EMBEDDING is not configured. Rejecting all update requests.")
#         return jsonify({"error": "Forbidden: URL validation is not configured on the server."}), 403

#     if final_url.startswith(allowed_url_prefixes):
#         logger.info(f"URL domain is authorized. Publishing task for: {final_url}")

#         try:
#             publisher = pubsub_v1.PublisherClient()
#             topic_path = publisher.topic_path(data["GCP_PROJECT_ID"], data["PUBSUB_TOPIC_ID"])
#             message_data = json.dumps({"url": final_url, "type": data_type}).encode("utf-8")
#             future = publisher.publish(
#                 topic_path,
#                 message_data,
#                 retry=custom_publisher_retry
#             )
#             logger.info(f"Published message ID {future.result()} to trigger background worker for URL: {final_url}")
#             return jsonify({"message": f"Update request for {final_url} received and is being processed."}), 202

#         except api_core.exceptions.InternalServerError as e:
#             logger.error(f"Failed to publish to Pub/Sub due to a 500 Internal Server Error (no retry). URL: '{final_url}': {e}", exc_info=True)
#             return jsonify({"error": "Failed to queue update task due to a persistent server error."}), 502
#         except Exception as e:
#             logger.critical(f"Failed to publish to Pub/Sub for URL '{final_url}': {e}", exc_info=True)
#             return jsonify({"error": "Failed to queue update task."}), 500

#     else:
#         logger.warning(f"Rejected request for an unauthorized URL domain: {final_url}")
#         return jsonify({
#             "error": f"Forbidden: The provided URL domain is not on the list of authorized sources."
#         }), 403


@app.route("/update-embedding", methods=["POST"])
def update_embedding():
    """
    Receives a payload for a document and determines whether
    to upsert (embed) or delete it from the knowledge base. It then publishes
    the corresponding task to Pub/Sub for background processing.
    """
    logger.info("Received request on /update-embedding endpoint.")
    query_data = request.get_json()
    if not query_data:
        return jsonify({"error": "Missing JSON payload."}), 400

    action = None
    final_url = ""
    data_type = query_data.get("type")

    if not data_type:
        return jsonify({"error": "Payload is missing the required 'type' field."}), 400

    if data_type in [
        "content-unpublished",
        "content-deleted",
        "media-deleted",
        "content-moving-to-recycle-bin",
        "content-moved-to-recycle-bin",
        "media-moving-to-recycle-bin",
        "media-moved-to-recycle-bin",
    ]:
        action = "delete"

    elif data_type in ["content", "media"]:
        action = "upsert"
    else:
        logger.warning(f"Received webhook with unknown type: '{data_type}'.")
        return jsonify({"error": f"Invalid or unsupported 'type': '{data_type}'."}), 400

    try:
        if "content" in data_type:
            final_url = query_data["urls"]["en-us"]
        elif "media" in data_type:
            final_url = query_data["mediaUrl"]
    except (KeyError, TypeError):
        logger.error(
            f"Payload with type '{data_type}' is missing required URL fields ('urls[en-us]' or 'mediaUrl')."
        )
        return (
            jsonify({"error": "Payload structure is invalid for the given type."}),
            400,
        )

    allowed_prefixes_str = data["URL_PREFIXES_EMBEDDING"]
    allowed_url_prefixes = tuple(
        [prefix.strip() for prefix in allowed_prefixes_str.split(",") if prefix.strip()]
    )

    if not allowed_url_prefixes:
        logger.critical(
            "SECURITY ALERT: URL_PREFIXES_EMBEDDING is not configured. Rejecting all webhook requests."
        )
        return (
            jsonify(
                {"error": "Forbidden: URL validation is not configured on the server."}
            ),
            403,
        )

    if not final_url.startswith(allowed_url_prefixes):
        logger.warning(
            f"Rejected {action} request for an unauthorized URL domain: {final_url}"
        )
        return (
            jsonify(
                {
                    "error": f"Forbidden: The provided URL domain is not on the list of authorized sources."
                }
            ),
            403,
        )

    logger.info(
        f"URL domain is authorized. Publishing '{action}' task for: {final_url}"
    )
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(
            data["GCP_PROJECT_ID"], data["PUBSUB_TOPIC_ID"]
        )

        message_payload = {"action": action, "url": final_url}
        if action == "upsert":
            message_payload["type"] = data_type

        message_data = json.dumps(message_payload).encode("utf-8")

        future = publisher.publish(
            topic_path, message_data, retry=custom_publisher_retry
        )
        logger.info(
            f"Published '{action}' message ID {future.result()} for URL: {final_url}"
        )
        return (
            jsonify(
                {
                    "message": f"{action.capitalize()} request for {final_url} received and is being processed."
                }
            ),
            202,
        )

    except Exception as e:
        logger.critical(
            f"Failed to publish '{action}' task to Pub/Sub for URL '{final_url}': {e}",
            exc_info=True,
        )
        return jsonify({"error": f"Failed to queue {action} task."}), 500


@app.route("/conversation", methods=["POST"])
def conversation_handler_dev():
    """
    Handles chat requests via POST using the google.genai SDK directly
    and streams the response chunk by chunk, following the correct API signature.
    It also sends a final payload with follow-up questions.
    """
    try:
        payload = request.get_json()
        message = payload.get("message")
        history = payload.get("history", [])
        if not message:
            return Response(
                '{"error": "Message is required."}',
                status=400,
                mimetype="application/json",
            )
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        return Response(
            '{"error": "Invalid request body."}',
            status=400,
            mimetype="application/json",
        )

    if len(message.strip()) < 3 or len(message) > 2048:
        logger.warning(f"Rejected query due to invalid length. Query: '{message}'")
        message = "UNANSWERABLE_QUERY_DUE_TO_INVALID_INPUT"

    if not re.search(r"[a-zA-Z0-9\u13A0-\u13FF\uAB70-\uABBF]", message):
        logger.warning(
            f"Rejected query due to non-alphanumeric content. Query: '{message}'"
        )
        message = "UNANSWERABLE_QUERY_DUE_TO_INVALID_INPUT"

    normalized_message = utils.clean_text(message.lower())
    if any(keyword in normalized_message for keyword in PROMPT_INJECTION_KEYWORDS) or message == "null":
        logger.warning(
            f"Potential prompt injection attempt detected. Original message: '{message}'. "
            "Replacing with a safe query to trigger fallback."
        )
        message = "UNANSWERABLE_QUERY_DUE_TO_PROMPT_FILTER"

    pii_found, masked_message = pii_check.detect_and_mask_pii(message)

    def generate_response_stream():
        full_response_text = ""
        summary = ""
        source_urls = None
        processed_history = history
        try:
            html_tag_pattern = re.compile(r"<[^>]+>")
            if html_tag_pattern.search(message):
                logger.warning(
                    f"Rejected query due to presence of HTML tags. Query: '{message}'"
                )
                answer_chunk = "It looks like your message had some special symbols or formatting I couldn't interpret. Please try again using simple, plain text."
                response = json.dumps({"answer_chunk": answer_chunk, "error": True})
                yield f"data: {response}\n\n"

            elif pii_found:
                logger.warning(
                    f"PII detected. Yielding error message instead of calling model. Original: '{message}'"
                )
                full_response_text = "To protect your privacy, we cannot process requests that appear to contain personal or sensitive information. Please rephrase your question without including any personal details and try again."
                error_chunk = json.dumps(
                    {"answer_chunk": full_response_text, "error": True}
                )
                yield f"data: {error_chunk}\n\n"

            else:
                if len(history):
                    summary = services._generate_history_summary_sync(history)

                    if summary:
                        processed_history = [
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "text": f"PREVIOUS CONVERSATION SUMMARY: {summary}"
                                    }
                                ],
                            }
                        ]
                    else:
                        logger.warning(
                            "History summarization failed. Proceeding with the original history."
                        )

                primary_response_generator = services.generate_primary_response(
                    masked_message, processed_history, vector_store_instance
                )

                for chunk_type, data in primary_response_generator:
                    if chunk_type == "text":
                        full_response_text += data
                        yield f"data: {json.dumps({'answer_chunk': data})}\n\n"
                    elif chunk_type == "sources":
                        source_urls = data
                        yield f"event: sources\ndata: {json.dumps({'source_urls': data})}\n\n"

                result_queue = queue.Queue()
                threads = []

                logger.info(
                    "Starting background threads for contact info and follow-ups"
                )
                contact_thread = threading.Thread(
                    target=services.extract_and_generate_contact_titles,
                    args=(full_response_text, result_queue, source_urls),
                )
                contact_thread.start()
                threads.append(contact_thread)
                current_turn_history = masked_message + " " + full_response_text

                follow_up_thread = threading.Thread(
                    target=services.generate_follow_ups_async,
                    args=(current_turn_history, result_queue),
                )
                follow_up_thread.start()
                threads.append(follow_up_thread)

                for _ in range(len(threads)):
                    try:
                        event_type, data = result_queue.get(timeout=20)
                        yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    except queue.Empty:
                        logger.warning(
                            "A background task for supplementary info timed out."
                        )

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            error_msg = "Cherokee Nation Chat is not available right now, please try again later."
            yield f"data: {json.dumps({'answer_chunk': error_msg, 'error': True})}\n\n"
        finally:
            logger.info("Stream finished. Sending 'done' event.")
            yield "event: done\ndata: {}\n\n"

    return Response(generate_response_stream(), mimetype="text/event-stream")
