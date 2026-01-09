import re
import json
from urllib.parse import unquote
from google.genai import types as genai_types
from google import genai
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from . import prompt_template, config
from . import config
from .utils import clean_txt_in_urls, find_contacts_with_regex, parse_json_from_markdown
from typing import Dict, Set

logger = config.setup_logger()
data = config.proj_config()

# Initialize the GenAI client once
genai_client = genai.Client(
    vertexai=True, project=data["GCP_PROJECT_ID"], location=data["GENAI_CLIENT"]
)
GEMINI_MODEL_NAME = data["GEMINI_MODEL_NAME"]


def generate_primary_response(message, history, vector_store_instance):
    """
    Generates the main chat response and streams text and sources.
    """
    logger.info("Generating primary response from model")
    contents = [
        genai_types.Content(
            role=item.get("role", "user"),
            parts=[
                genai_types.Part.from_text(
                    text=item.get("parts", [{}])[0].get("text", "")
                )
            ],
        )
        for item in history
    ]
    contents.append(
        genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=message)]
        )
    )

    rag_tool = genai_types.Tool(
        retrieval=genai_types.Retrieval(
            vertex_rag_store=genai_types.VertexRagStore(
                rag_resources=[
                    genai_types.VertexRagStoreRagResource(
                        rag_corpus=vector_store_instance.rag_corpus.name
                    )
                ],
                similarity_top_k=20,
            )
        )
    )

    system_prompt_str = prompt_template.new_rag_prompt_datetime()
    system_instruction_part = genai_types.Part.from_text(text=system_prompt_str)

    generation_config = genai_types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=2048,
        safety_settings=[
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
            ),
        ],
        tools=[rag_tool],
        system_instruction=[system_instruction_part],
        thinking_config=genai_types.ThinkingConfig(
            thinking_budget=0,
        ),
    )

    @retry(
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
    )
    def generate_with_retry():
        """
        This function wraps the SDK call with a retry decorator to handle
        429 ResourceExhausted errors.
        """
        return genai_client.models.generate_content_stream(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=generation_config,
        )

    response_stream = generate_with_retry()

    all_ordered_sources = []
    CITATION_REGEX = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")

    for chunk in response_stream:
        if chunk.text:
            text = chunk.text
            text = CITATION_REGEX.sub("", chunk.text)
            text = clean_txt_in_urls(text)
            text = re.sub(
                r"\[\s*\d*(?:,\s*\d*)*\s*$", "", text
            )  # Incomplete citation at the end
            text = re.sub(
                r"^\s*\d*(?:,\s*\d*)*\s*\]", "", text
            )  # Incomplete citation at the start
            yield "text", text

        if (candidate := next(iter(chunk.candidates), None)) and (
            metadata := getattr(candidate, "grounding_metadata", None)
        ):
            if metadata.grounding_chunks is not None:
                for grounding_chunk in metadata.grounding_chunks:
                    context = grounding_chunk.retrieved_context
                    if context and context.title and "|" in context.title:
                        try:
                            encoded_url = context.title.split("|", 1)[1]
                            decoded_url = unquote(encoded_url).replace(".txt", "")
                            if decoded_url and decoded_url not in all_ordered_sources:
                                all_ordered_sources.append(decoded_url)
                        except IndexError:
                            continue

    yield "sources", all_ordered_sources


def _generate_history_summary_sync(history: list) -> str:
    """(Internal Function) Summarizes a conversation history to conserve tokens. This function performs the actual blocking API call with retry logic."""
    if not history:
        return ""

    transcript = "\n\n".join(
        f"{'User' if item.get('role') == 'user' else 'Model'}: {item.get('parts', [{}])[0].get('text', '')}"
        for item in history
    )
    system_prompt_str = prompt_template.summarization_prompt_template()
    system_instruction = genai_types.Part.from_text(text=system_prompt_str)
    contents = [genai_types.Part.from_text(text=transcript)]
    config = genai_types.GenerateContentConfig(
        temperature=0.2, max_output_tokens=1024, system_instruction=[system_instruction]
    )
    response = genai_client.models.generate_content(
        model=GEMINI_MODEL_NAME, contents=contents, config=config
    )

    summary = response.text.strip()
    logger.info(f"Generated summary of length: {len(summary)}")
    return summary


def _generate_titles_for_contacts_batch(
    contacts: Dict[str, Set[str]], context_text: str
) -> Dict[str, str]:
    """Generates context-aware titles for all contacts in a single API call."""

    items_to_title = set()
    items_to_title.update(contacts.get("urls", set()))
    items_to_title.update(contacts.get("emails", set()))
    items_to_title.update(contacts.get("phones", set()))
    all_contacts_list = sorted(list(items_to_title))

    if not all_contacts_list:
        return {}

    try:
        instruction_prompt = prompt_template.batched_contact_title_prompt(
            all_contacts_list, context_text
        )

        @retry(
            retry=retry_if_exception_type(google_exceptions.ResourceExhausted),
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(3),
        )
        def generate_with_retry():
            logger.debug(
                "Attempting single model call for all contextual contact titles..."
            )
            return genai_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[genai_types.Part.from_text(text=instruction_prompt)],
                config={"response_mime_type": "application/json"},
            )

        response = generate_with_retry()
        title_map = parse_json_from_markdown(response.text)

        if not isinstance(title_map, dict):
            logger.error("AI did not return a valid JSON object for contact titles.")
            return {}
        return title_map

    except Exception as e:
        logger.error(f"Failed to generate titles in batch: {e}", exc_info=True)
        return {}


def extract_and_generate_contact_titles(
    text: str, result_queue, source_urls: list
) -> None:
    """A single pipeline that finds contacts and generates all their context-aware titles with a single, batched API call."""
    empty_contacts = {"urls": [], "emails": [], "phones": []}

    try:
        found_contacts = find_contacts_with_regex(text, source_urls)
        if not any(found_contacts.values()):
            result_queue.put(("contact_info", empty_contacts))
            return

        title_map = _generate_titles_for_contacts_batch(found_contacts, text)
        structured_output = {"urls": [], "emails": [], "phones": []}

        for url in sorted(list(found_contacts.get("urls", set()))):
            structured_output["urls"].append(
                {"url": url, "title": title_map.get(url, "Visit Link")}
            )

        for email in sorted(list(found_contacts.get("emails", set()))):
            default_email_title = (
                f"Email {email.split('@')[0].replace('.', ' ').title()}"
            )
            structured_output["emails"].append(
                {"email": email, "title": title_map.get(email, default_email_title)}
            )

        for phone in sorted(list(found_contacts.get("phones", set()))):
            structured_output["phones"].append(
                {"phone": phone, "title": title_map.get(phone, f"Call {phone}")}
            )

        result_queue.put(("contact_info", structured_output))

    except Exception as e:
        logger.error(f"An error occurred in the contact pipeline: {e}", exc_info=True)
        result_queue.put(("contact_info", empty_contacts))


def generate_follow_ups_async(text: str, result_queue) -> None:
    """
    Worker function to generate follow-up questions, ensuring a JSON response.
    """
    logger.info("Generating follow-up question")
    try:
        json_prompt = prompt_template.json_followup_prompt_template()
        contents = [
            genai_types.Part.from_text(text=text),
            genai_types.Part.from_text(text=json_prompt),
        ]
        generation_config = genai_types.GenerateContentConfig(
            temperature=0.7, response_mime_type="application/json"
        )
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL_NAME, contents=contents, config=generation_config
        )
        parsed_response = parse_json_from_markdown(response.text)
        questions = parsed_response.get("questions", [])
        if not isinstance(questions, list):
            logger.warning(
                "AI returned a 'questions' key, but it was not a list. Defaulting to empty."
            )
            questions = []

        result_queue.put(("follow_up", {"follow_up_questions": questions}))

    except json.JSONDecodeError:
        logger.error(
            "Failed to decode JSON from the model's response for follow-ups.",
            exc_info=True,
        )
        result_queue.put(("follow_up", {"follow_up_questions": []}))
    except Exception as e:
        logger.error(f"Failed to generate follow-up questions: {e}", exc_info=True)
        result_queue.put(("follow_up", {"follow_up_questions": []}))
