import hashlib
import vertexai
from vertexai import rag
from google.cloud import storage
from urllib.parse import quote as url_encode
import requests
from bs4 import BeautifulSoup
import warnings 
from . import config
warnings.filterwarnings("ignore")

data = config.proj_config()
logger = config.setup_logger()

GEMINI_MODEL_NAME=data.get("GEMINI_MODEL_NAME")

HEADERS = {
    'User-Agent': data.get("USER_AGENT")
}

class VectorStore:
    """Manages interactions with Vertex AI RAG, Google Cloud Storage, and web scraping."""
    
    def __init__(self, project_id: str = None, region: str = None, 
                 bucket_name: str = None, corpus_display_name: str = None):
        """Initializes the VectorStore with GCP and RAG configuration."""
        self.project_id = project_id 
        self.region = region
        self.bucket_name = bucket_name 
        self.corpus_display_name = corpus_display_name 
        
        if not all([self.project_id, self.region, self.bucket_name, self.corpus_display_name]):
            logger.critical("GCP Config (Project, Region, Bucket, Corpus) must be provided.")
            raise ValueError("GCP Config (Project, Region, Bucket, Corpus) must be provided.")
        
        logger.info(f"Initializing Vertex AI for project '{self.project_id}' in '{self.region}'.")
        vertexai.init(project=self.project_id, location=self.region)
        self.storage_client = storage.Client()
        self.rag_corpus = self._get_corpus()

    def _get_corpus(self):
        """Finds a RAG corpus by its display name or creates it if it doesn't exist."""
        if not self.corpus_display_name:
            raise ValueError("CORPUS_DISPLAY_NAME environment variable not set or is empty.")
        logger.debug(f"Searching for RAG corpus with display name: '{self.corpus_display_name}'")
        corpora = rag.list_corpora()
        for corpus in corpora:
            if corpus.display_name == self.corpus_display_name:
                logger.info(f"Found existing corpus '{corpus.display_name}' with ID: {corpus.name}")
                return corpus
        logger.warning(f"Corpus '{self.corpus_display_name}' not found. Creating a new one.")
        return rag.create_corpus(display_name=self.corpus_display_name)

    def _upload_to_gcs(self, content: bytes, destination_blob_name: str, content_type: str = "text/plain; charset=utf-8") -> str:
        """Uploads content to a GCS bucket and returns the GCS URI."""
        gcs_uri = f"gs://{self.bucket_name}/{destination_blob_name}"
        logger.debug(f"Attempting to upload {len(content)} bytes to {gcs_uri}")
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_string(content, content_type=content_type)
            logger.info(f"Successfully uploaded content to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            logger.error(f"Failed to upload to GCS at '{gcs_uri}'. Error: {e}", exc_info=True)
            raise


    def _create_safe_gcs_blob_name(self, file_display_name: str, document_url: str) -> str:
        """Creates a GCS-safe blob name that includes the original URL."""
        encoded_url = url_encode(document_url, safe='')
        blob_name = f"{file_display_name}|{encoded_url}.txt"
        logger.debug(f"Created GCS-safe blob name: '{blob_name}' for URL: '{document_url}'")
        return blob_name

    def _delete_existing_rag_file(self, file_display_name: str):
        """Deletes a RAG file by its unique display name hash."""
        logger.debug(f"Checking for existing RAG file with display name hash: {file_display_name}")
        try:
            for file in rag.list_files(self.rag_corpus.name):
                try:
                    if str(file.display_name).split("|")[0] == file_display_name:
                        rag.delete_file(file.name)
                        logger.info(f"Successfully deleted RAG file: {file.name}")
                        break
                except AttributeError as inner_e:
                    logger.warning(f"Skipped a file due to missing attributes: {inner_e}")
            logger.info(f"No existing RAG file found with hash '{file_display_name}'.")
        except Exception as e:
            logger.error(f"Error during RAG file deletion check for hash '{file_display_name}': {e}")

    def _delete_from_gcs(self, blob_name: str):
        """Deletes a blob from the GCS bucket if it exists."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                logger.info(f"Successfully deleted blob '{blob_name}' from GCS.")
            else:
                logger.warning(f"Blob '{blob_name}' not found in GCS. Nothing to delete.")
        except Exception as e:
            logger.error(f"Failed to delete blob '{blob_name}' from GCS. Error: {e}", exc_info=True)
    
    def delete_document_by_url(self, document_url: str):
        """
        Finds and deletes a document from RAG and its corresponding file in GCS
        based on its original source URL.
        """
        logger.info(f"Starting deletion process for URL: {document_url} ")
        if not document_url:
            logger.warning("No document URL provided. Skipping deletion.")
            return

        file_display_name_hash = hashlib.sha256(document_url.encode('utf-8')).hexdigest()
        gcs_blob_name = self._create_safe_gcs_blob_name(file_display_name_hash, document_url)

        # Delete the file from the Vertex AI RAG corpus.
        self._delete_existing_rag_file(file_display_name_hash)

        # Delete the corresponding source file from Google Cloud Storage.
        self._delete_from_gcs(gcs_blob_name)

        logger.info(f" Successfully completed deletion process for URL: {document_url} ")
        
    def parse_html_content(self, html_content: bytes) -> tuple[str | None, str | None]:
        """Parses HTML and extracts content from specific class containers."""
        TARGET_CLASSES = ["content-text-full", "right-content"]
        logger.debug(f"Parsing {len(html_content)} bytes of HTML content.")
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            selected_elements = []
            for class_name in TARGET_CLASSES:
                selected_elements.extend(soup.find_all(class_=class_name))

            if selected_elements:
                logger.info(f"Extracting text from {len(selected_elements)} targeted HTML elements.")
                text_parts = [el.get_text(separator="\n", strip=True) for el in selected_elements]
                raw_text = "\n".join(text_parts)
            else:
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()
                raw_text = soup.get_text()

            lines = (line.strip() for line in raw_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            logger.info(f"Successfully parsed HTML, extracted {len(clean_text)} characters of clean text.")
            return clean_text, None
        except Exception as e:
            logger.error(f"Failed to parse HTML content: {e}", exc_info=True)
            return None, f"Error parsing HTML content: {e}"

    def scrape_website(self, url: str) -> tuple[str | None, str | None]:
        """Fetches a URL and extracts clean text using parse_html_content."""
        logger.info(f"Starting scrape for URL: {url}")
        try:
            response = requests.get(url, timeout=20, headers=HEADERS, verify=False)
            response.raise_for_status()
            logger.info(f"Successfully fetched URL '{url}' with status code {response.status_code}.")
            return self.parse_html_content(response.content)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL '{url}'. Error: {e}", exc_info=True)
            return None, f"Error fetching URL '{url}': {e}"

    def upsert_pdf_with_llm_parser(self, pdf_content: bytes, document_url: str):
        """
        Uploads a raw PDF to GCS and ingests it using the native LLM Parser.
        """
        logger.info(f"Starting PDF upsert with LLM Parser for: {document_url} ")
        if not pdf_content:
            logger.warning("No PDF content provided. Skipping upsert.")
            return

        file_display_name = hashlib.sha256(document_url.encode('utf-8')).hexdigest()
        self._delete_existing_rag_file(file_display_name)
        gcs_blob_name = self._create_safe_gcs_blob_name(file_display_name, document_url)

        gcs_uri = self._upload_to_gcs(pdf_content, gcs_blob_name, "application/pdf")

        llm_parser_config = rag.LlmParserConfig(model_name=GEMINI_MODEL_NAME)
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

    def delete_all_corpus_content(self):
        """Deletes all RagFiles in the RAG corpus."""
        # logger.warning(f"--- Preparing to delete ALL content from corpus: '{self.rag_corpus.display_name}' ({self.rag_corpus.name}) ---")
        try:
            existing_files = list(rag.list_files(self.rag_corpus.name))
            if not existing_files:
                logger.info("Corpus is already empty. No files to delete.")
                return

            logger.info(f"Found {len(existing_files)} files to delete...")
            for file in existing_files:
                logger.debug(f"Submitting deletion request for file: {file.display_name} (ID: {file.name})")
                rag.delete_file(file.name)

            logger.info(f"--- Successfully deleted all {len(existing_files)} files from the corpus. ---")
        except Exception as e:
            logger.critical(f"A critical error occurred while deleting corpus content: {e}", exc_info=True)
            raise e

    def upsert_text_content(self, content: str, document_url: str):
        """
        Handles ingesting text from any source (PDF, scraped HTML, etc.).
        """
        logger.info(f"--- Starting text content upsert for: {document_url} ---")
        if not content:
            # logger.warning(f"No content provided for {document_url}. Skipping.")
            return

        file_display_name = hashlib.sha256(document_url.encode('utf-8')).hexdigest()
        self._delete_existing_rag_file(file_display_name)
        gcs_blob_name = self._create_safe_gcs_blob_name(file_display_name, document_url)
        
        gcs_uri = self._upload_to_gcs(content.encode('utf-8'), gcs_blob_name)
        logger.info(f"Starting RAG import job for GCS URI: {gcs_uri}")

        rag.import_files(
            self.rag_corpus.name,
            [gcs_uri],
            transformation_config=rag.TransformationConfig(
                rag.ChunkingConfig(chunk_size=1024, chunk_overlap=150)
            ),
            max_embedding_requests_per_min=900
        )
        logger.info(f"--- Successfully started ingestion job for: {document_url} ---")

    def upsert_scraped_url(self, url: str):
        """Performs the complete upsert operation for a given URL."""
        logger.info(f"--- Starting full upsert process for: {url} ---")
        content, error = self.scrape_website(url)
        if error:
            logger.error(f"Failed to scrape {url}. Aborting upsert. Error: {error}")
            return
        self.upsert_text_content(content, document_url=url)
