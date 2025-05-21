"""Helper functions for LLM"""

import json
import re # Added import for regular expressions
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory=None,
) -> T:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider)

    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result is None:
                    # Check if result.content exists before printing
                    if hasattr(result, 'content'):
                        print(f"LLM_RESPONSE_ERROR: Raw response content that caused error: {result.content}")
                    raise ValueError("Failed to parse JSON from LLM response")
                return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            # Check if result and result.content exist before printing
            if 'result' in locals() and hasattr(result, 'content'):
                print(f"LLM_RESPONSE_ERROR: Raw response content that caused error: {result.content}")
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> Optional[dict]:
    """
    Extracts JSON from markdown-formatted response.
    Removes <think>...</think> blocks and then attempts to parse JSON.
    """
    # Remove <think>...</think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    try:
        # Attempt to find JSON within ```json ... ``` code blocks
        json_start = content.find("```json")
        if json_start != -1:
            # Adjust json_start to be after "```json"
            json_text_start = content[json_start + 7 :] 
            json_end = json_text_start.find("```")
            if json_end != -1:
                json_text = json_text_start[:json_end].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from ```json ... ``` block: {e}")
                    # Fall through to try parsing the whole content if this fails
                    pass # Add this pass to explicitly show we are falling through

        # Fallback: if no ```json ... ``` block or if parsing it failed, try to parse the entire content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from the entire content: {e}")
            return None

    except Exception as e: # General exception catch for unexpected errors
        print(f"Unexpected error in extract_json_from_response: {e}")
    return None
