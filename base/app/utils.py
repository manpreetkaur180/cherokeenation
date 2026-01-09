import re
import json
# from dotenv import load_dotenv
from . import config
from google.genai import types
from . import prompt_template
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, retry_if_exception_type, wait_random_exponential
from typing import Dict, List, Set

logger = config.setup_logger()


url_pattern = re.compile(
    r'(?:https?://|www\.)[^\s/$.?#].[^\s]*?'
    r'(?:(?:\.[a-z]{2,5})(?:\.txt)?)(?=$|\s|[\"\'<>])|'
    r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}/[^\s]*?'
    r'(?:(?:\.[a-z]{2,5})(?:\.txt)?)(?=$|\s|[\"\'<>])|'
    r'\b(?:[a-zA-Z0-9-]+\.){2,}[a-zA-Z]{2,}(?:\/[^\s]*)?\b',
    re.IGNORECASE
)

def clean_txt_in_urls(text: str) -> str:
    def repl(match: re.Match) -> str:
        url = match.group(0)
        # Remove .txt if it follows another extension or is at the end/query/fragment
        url = re.sub(r'(?i)(\.[a-z0-9]{1,6})\.txt(?=(?:$|\?|#|&))', r'\1', url)
        return url

    return url_pattern.sub(repl, text)


PROMPT_INJECTION_KEYWORDS = config.PROMPT_INJECTION_KEYWORDS

def parse_json_from_markdown(markdown_string: str):
    match = re.search(r"```json\s*\n(.*?)\n\s*```", markdown_string, re.DOTALL)
    json_string = match.group(1).strip() if match else markdown_string.strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        logger.warning(f"Warning: Failed to decode JSON from string: {json_string}")
        return None


def clean_text(message: str) -> str:
    CLEANING_REGEX_SIMPLE = re.compile(r"[^a-zA-Z0-9\s.?-]")
    cleaned_message = CLEANING_REGEX_SIMPLE.sub("", message)
    return cleaned_message

def find_contacts_with_regex(text: str, source_urls) -> Dict[str, Set[str]]:
    """Finds all unique URLs, emails, and phone numbers in a given text using regex."""
    markdown_link_pattern = re.compile(r'\[(?:[^\]]*)\]\(([^)]+)\)')
    cleaned_text = markdown_link_pattern.sub(r'\1', text)
    url_pattern = re.compile(
        r'(?:https?://|www\.)[^\s/$.?#].[^\s]*?'
        r'(?:(?:\.[a-z]{2,5})(?:\.txt)?)(?=$|\s|[\"\'<>])|'
        r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}/[^\s]*?'
        r'(?:(?:\.[a-z]{2,5})(?:\.txt)?)(?=$|\s|[\"\'<>])|'
        r'\b(?:[a-zA-Z0-9-]+\.){2,}[a-zA-Z]{2,}(?:\/[^\s]*)?\b',
        re.IGNORECASE | re.MULTILINE
    )
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phone_pattern = re.compile(r'\(?\b[2-9][0-9]{2}\)?[-.\s]?[2-9][0-9]{2}[-.\s]?[0-9]{4}\b')

    initial_urls = set(url_pattern.findall(cleaned_text))
    found_emails = set(email_pattern.findall(text))
    found_phones = set(phone_pattern.findall(text))

    processed_urls = set()
    for url in initial_urls:
        url = url.rstrip('./')
        url = re.sub(r'(?i)(\.[a-z0-9]{1,6})\.txt(?=(?:$|\?|#|&))', r'\1', url)
        cleaned_url = re.sub(r'(?i)\.txt(?=(?:$|\?|#|&))', '', url)
        if not url.lower().startswith(('http://', 'https://')):
            processed_urls.add('https://' + cleaned_url)
        else:
            processed_urls.add(cleaned_url)
            
    return {
        "urls": source_urls[:3] if source_urls else sorted(list(processed_urls))[:4],
        "emails": sorted(list(found_emails))[:4],
        "phones": sorted(list(found_phones))[:4]
    }
    