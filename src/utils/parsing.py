import json
import logging # Use logging instead of print

logger = logging.getLogger(__name__)

def parse_json_response(response_content: str) -> dict | None: # Renamed for clarity and added type hint
    """Parses a JSON string and returns a dictionary. Logs errors instead of printing."""
    if not isinstance(response_content, str):
        logger.error(f"Invalid response type for JSON parsing. Expected string, got {type(response_content).__name__}. Response: {repr(response_content)}")
        return None
    try:
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}. Response: {repr(response_content)}")
        return None
    except Exception as e: # Catch any other unexpected errors during parsing
        logger.error(f"Unexpected error while parsing JSON response: {e}. Response: {repr(response_content)}")
        return None
