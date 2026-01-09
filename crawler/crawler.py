import os
import requests
import collections
import hashlib
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import logging
import vertexai
from vertexai.preview import rag
from google.cloud import storage
from urllib.parse import quote as url_encode
import time
from dotenv import load_dotenv
load_dotenv()

def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False 

    return logger

logger = setup_logger()

# ==============================================================================
# SECTION 1: SELF-CONTAINED DEPENDENCY CLASSES
# ==============================================================================

class Scraper:
    """Parses HTML and extracts content only from specific known class containers."""
    TARGET_CLASSES = ["content-text-full", "right-content"]

    @staticmethod
    def parse_html_content(html_content: bytes) -> tuple[str | None, str | None]:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Try to extract content only from specific class containers
            selected_elements = []
            for class_name in Scraper.TARGET_CLASSES:
                selected_elements.extend(soup.find_all(class_=class_name))

            if selected_elements:
                text_parts = [el.get_text(separator="\n", strip=True) for el in selected_elements]
                raw_text = "\n".join(text_parts)
            else:
                # Fallback: scrape the full HTML body without scripts/styles/navs/etc.
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()
                raw_text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in raw_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            return clean_text, None

        except Exception as e:
            return None, f"Error parsing HTML content: {e}"

class VectorStore:
    """Manages all interactions with Vertex AI RAG and Google Cloud Storage."""
    def __init__(self, project_id: str, region: str, bucket_name: str, corpus_display_name: str):
        if not all([project_id, region, bucket_name, corpus_display_name]):
            raise ValueError("GCP Config (Project, Region, Bucket, Corpus) must be provided.")
        vertexai.init(project=project_id, location=region)
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name
        self.rag_corpus = self._get_or_create_corpus(corpus_display_name)
    
    def _get_or_create_corpus(self, display_name: str):
        for corpus in rag.list_corpora():
            if corpus.display_name == display_name:
                logger.info(f"Found existing corpus: {corpus.name}")
                return corpus
        return rag.create_corpus(display_name=display_name)

    def _upload_bytes_to_gcs(self, content: bytes, blob_name: str, content_type: str) -> str:
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type=content_type)
        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
        logger.info(f"Uploaded to GCS: {gcs_uri}")
        return gcs_uri
        
    def _delete_existing_rag_file(self, file_display_name: str):
        """
        Helper to delete a RAG file by its unique display name.
        This now uses a reliable, exact match.
        """
        try:
            for file in rag.list_files(self.rag_corpus.name):
                try:
                    if str(file.display_name).split("|")[0] == file_display_name:
                        logger.info(f"Found and deleting existing RAG file: {file.name}")
                        rag.delete_file(file.name)
                        break
                except AttributeError as inner_e:
                    logger.exception(f"Skipped a file due to missing attributes: {inner_e}")
        except Exception as e:
            logger.warning(f"Could not check/delete existing file. Info: {e}")

    # --- NEW HELPER FUNCTION TO CREATE SAFE, HUMAN-READABLE FILENAMES ---
    def _create_safe_gcs_blob_name(self, file_display_name: str, document_url: str) -> str:
        encoded_url = url_encode(document_url, safe='')
        return f"{file_display_name}|{encoded_url}.txt"


    def upsert_text_content(self, content: str, document_url: str):
        """
        This is the final ingestion step for ALL content. It now uses the
        safe filename format and correctly attaches metadata.
        """
        # 1. The display_name for the RAG Corpus REMAINS the clean, naked hash.
        file_display_name = hashlib.sha256(document_url.encode('utf-8')).hexdigest()
        self._delete_existing_rag_file(file_display_name)
        
        # 2. Create the new, human-readable, and safe blob name for the GCS file.
        gcs_blob_name = self._create_safe_gcs_blob_name(file_display_name, document_url)

        # 3. Upload the content to GCS using the new, safe blob name.
        gcs_uri = self._upload_bytes_to_gcs(
            content.encode('utf-8'), gcs_blob_name, 'text/plain; charset=utf-8'
        )
        
        rag.import_files(
            self.rag_corpus.name,
            [gcs_uri],
            chunk_overlap=200,
            chunk_size=1024,
            max_embedding_requests_per_min=900
        )
        logger.info(f"--- Successfully started text ingestion for: {document_url} ---")

    def upsert_pdf_with_llm_parser(self, pdf_content: bytes, document_url: str):
        """
        Uploads a raw PDF to GCS and ingests it using the native LLM Parser.
        """
        logger.info(f"--- Starting PDF upsert with LLM Parser for: {document_url} ---")
        if not pdf_content:
            logger.warning("No PDF content provided. Skipping upsert.")
            return

        file_display_name = hashlib.sha256(document_url.encode('utf-8')).hexdigest()
        self._delete_existing_rag_file(file_display_name)
        gcs_blob_name = self._create_safe_gcs_blob_name(file_display_name, document_url) 

        gcs_uri = self._upload_bytes_to_gcs(pdf_content, gcs_blob_name, "application/pdf")

        llm_parser_config = rag.LlmParserConfig(model_name="gemini-2.5-flash")
        transformation_config = rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=1024, chunk_overlap=150),
        )
        
        logger.info(f"Starting RAG import job for GCS URI: {gcs_uri}")

        rag.import_files(
            self.rag_corpus.name,
            [gcs_uri],
            llm_parser=llm_parser_config,
            transformation_config=transformation_config,
            max_embedding_requests_per_min=900
        )
        logger.info(f"Successfully started LLM Parser ingestion job for: {document_url}")

# ==============================================================================
# SECTION 2: THE CRAWLER
# ==============================================================================

class Crawler:
    """
    Crawls and processes websites sequentially (one by one).
    """
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    BLACKLISTED_EXTENSIONS = [
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.7z', 
        '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.mp3', 
        '.wav', '.ogg', '.mp4', '.avi', '.mov', '.wmv', '.css', '.js'
    ]

    def __init__(self, vector_store: VectorStore):
        """Initializes the crawler without any threading components."""
        self.vector_store = vector_store
        self.scraper = Scraper()

    def process_url(self, url: str) -> bytes | None:
        """
        Downloads, identifies, ingests content, and returns the raw HTML content
        if applicable, so it can be parsed for links without a second download.
        """
        logger.info(f"Processing URL: {url}")
        try:
            with requests.get(url, headers=self.HEADERS, timeout=30, stream=True, verify=False) as response:
                response.raise_for_status()
                content = response.content
                content_type = response.headers.get('Content-Type', '').lower()

                if 'application/pdf' in content_type:
                    logger.info(f"Found PDF. Routing to Gemini text extraction...")
                    self.vector_store.upsert_pdf_with_llm_parser(content, document_url=url)
                    time.sleep(5)
                    return None
                elif 'text/html' in content_type:
                    logger.info(f"Found HTML page. Submitting to text ingestion...")
                    time.sleep(5)
                    clean_text, error = self.scraper.parse_html_content(content)
                    if error:
                        logger.error(f"Could not parse HTML for {url}. Reason: {error}"); return None
                    self.vector_store.upsert_text_content(clean_text, document_url=url)
                    return content
                else:
                    logger.error(f"Skipping unsupported content type '{content_type}' for URL: {url}")
                    return None
        except Exception as e:
            logger.error(f"Failed to process URL '{url}'. Reason: {e}")
            return None

    def crawl_domain(self, start_url: str):
        """
        Discovers and processes all links on a domain sequentially.
        """
        try:
            domain_netloc = urlparse(start_url).netloc
            if not domain_netloc:
                logger.error(f"Invalid start URL, cannot determine domain: {start_url}"); return
        except Exception as e:
            logger.error(f"Could not parse start URL '{start_url}': {e}"); return

        urls_to_visit = collections.deque([start_url])
        visited_urls = {start_url}
        logger.info(f"--- Starting Sequential Crawl for domain: {domain_netloc} ---")

        while urls_to_visit:
            current_url = urls_to_visit.popleft()
            
            html_content = self.process_url(current_url)

            if html_content:
                try:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        try:
                            absolute_link = urljoin(current_url, link['href'])
                            parsed_link = urlparse(absolute_link)
                            clean_link = parsed_link._replace(query="", fragment="").geturl()
                            if (parsed_link.netloc == domain_netloc and clean_link not in visited_urls):
                                visited_urls.add(clean_link)
                                urls_to_visit.append(clean_link)
                        except Exception as e:
                            logger.warning(f"Could not process a link on page {current_url}. Reason: {e}")
                except Exception as e:
                    logger.warning(f"Could not parse links from '{current_url}'. Reason: {e}")
        
        logger.info(f"--- Finished crawl for {domain_netloc}. Processed {len(visited_urls)} unique URLs. ---")

    def run_ingestion_from_env(self):
        """Runs the entire ingestion process sequentially using the hybrid strategy."""
        logger.info("--- SEQUENTIAL INGESTION SCRIPT STARTED ---")
        
        url_list_str = os.getenv("CRAWL_DOMAINS")
        if not url_list_str:
            logger.critical("CRITICAL: CRAWL_DOMAINS environment variable not set. Exiting."); return

        all_start_points = [url.strip() for url in url_list_str.split(',') if url.strip()]
        for start_point in all_start_points:
            if not start_point.startswith(('http://', 'https://')):
                start_point = 'https://' + start_point

            if start_point.lower().endswith('.pdf'):
                logger.info(f"Found direct PDF link: {start_point}")
                # This is now a direct, blocking call.
                self.process_url(start_point)
            else:
                logger.info(f"Found domain/start page: {start_point}")
                # This is also a direct, blocking call.
                self.crawl_domain(start_point)

        # The executor and future-waiting logic is removed.
        logger.info("--- INGESTION SCRIPT FINISHED ---")
        
# ==============================================================================
# SECTION 3: SCRIPT ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    
    try:
        GCP_PROJECT_ID = str(os.getenv("GCP_PROJECT_ID"))
        GCP_REGION = str(os.getenv("GCP_REGION"))
        GCS_BUCKET_NAME = str(os.getenv("GCS_BUCKET_NAME"))
        CORPUS_DISPLAY_NAME = str(os.getenv("CORPUS_DISPLAY_NAME"))
        
        print(f"Project: {GCP_PROJECT_ID}, Region: {GCP_REGION}, Bucket: {GCS_BUCKET_NAME}, Corpus: {CORPUS_DISPLAY_NAME}")

        vector_store_instance = VectorStore(
            project_id=GCP_PROJECT_ID, region=GCP_REGION,
            bucket_name=GCS_BUCKET_NAME, corpus_display_name=CORPUS_DISPLAY_NAME
        )
        
        crawler_instance = Crawler(vector_store=vector_store_instance)
        crawler_instance.run_ingestion_from_env()

    except Exception as e:
        logger.critical(f"The crawler could not start. Reason: {e}")
        exit(1)