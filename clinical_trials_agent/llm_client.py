import logging
from typing import Any, Optional
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

from .config import get_llm_config
from .utils import safe_json_loads

logger = logging.getLogger(__name__)

# Configure logging
llm_logger = logging.getLogger('llm_responses')
llm_handler = logging.FileHandler('llm_responses.log')
llm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
llm_logger.addHandler(llm_handler)
llm_logger.setLevel(logging.INFO)


class LLMClient:
    """Unified client for interacting with LLM APIs (sync and async)."""
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        llm_config = get_llm_config()
        self.model = model or llm_config["models"]["study_processing"]
        self.api_key = llm_config["api_key"]
        self.temperature = temperature
        
        # Sync client
        try:
            self._client = genai.Client(api_key=self.api_key)
            self._aio_client = genai.Client(
                api_key=self.api_key,
                http_options=HttpOptions(api_version="v1")
            )
            logger.info(f"LLMClient initialized with model: '{self.model}'")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google GenAI client: {e}")

    def generate_text(self, prompt: str) -> str:
        """Synchronous text generation."""
        logger.info(f"Making text generation request to model: {self.model}")
        llm_logger.info(f"TEXT REQUEST PROMPT:\n{prompt[:100000]}...")

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=GenerateContentConfig(
                    temperature=self.temperature,
                    candidate_count=1,
                    max_output_tokens=64000,
                )
            )

            if not response.candidates:
                llm_logger.warning("RESPONSE BLOCKED: No candidates returned")
                return ""

            text = self._extract_response_text(response)
            llm_logger.debug(f"EXTRACTED TEXT:\n{text}")
            return text.strip() if text else ""

        except Exception as e:
            logger.error(f"LLM text generation failed: {e}")
            llm_logger.error(f"TEXT GENERATION EXCEPTION: {e}")
            return ""

    async def generate_text_async(self, prompt: str) -> str:
        """Asynchronous text generation."""
        logger.debug(f"Making ASYNC text generation request to model: {self.model}")
        llm_logger.debug(f"ASYNC REQUEST PROMPT:\n{prompt[:100000]}...")

        try:
            response = await self._aio_client.aio.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=GenerateContentConfig(
                    temperature=self.temperature,
                    candidate_count=1,
                    max_output_tokens=64000,
                )
            )

            if not response.candidates:
                llm_logger.warning("ASYNC RESPONSE BLOCKED: No candidates returned")
                return ""

            text = self._extract_response_text(response)
            llm_logger.debug(f"ASYNC EXTRACTED TEXT (first 500): {text[:5000]}...")
            return text.strip() if text else ""

        except Exception as e:
            logger.error(f"LLM async generation failed: {e}")
            llm_logger.error(f"ASYNC GENERATION EXCEPTION: {e}")
            return ""

    def _extract_response_text(self, response) -> str:
        """Extract text from response object (works for sync and async)."""
        try:
            if hasattr(response, 'text') and response.text:
                return response.text

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and 
                    candidate.content and 
                    hasattr(candidate.content, 'parts') and 
                    candidate.content.parts):
                    part = candidate.content.parts[0]
                    if hasattr(part, 'text'):
                        return part.text
                    elif isinstance(part, dict) and 'text' in part:
                        return part['text']

            # Fallback: try model_dump or to_dict
            if hasattr(response, 'model_dump'):
                dump = response.model_dump()
                try:
                    return dump['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError):
                    pass

            llm_logger.error("Could not extract text from response")
            return ""

        except Exception as e:
            llm_logger.error(f"Error extracting text: {e}")
            return ""