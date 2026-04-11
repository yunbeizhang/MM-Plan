import base64
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


# ============================================================================
# OpenAI API Functions (GPT-4o, GPT-4.5)
# ============================================================================

def _prepare_openai_content(prompt: str, image_path: Optional[str] = None) -> List[dict]:
    """Prepare content list for OpenAI API with optional image."""
    content = [{"type": "text", "text": prompt}]
    
    if image_path:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Determine image format from file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(ext, "image/jpeg")
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }
        })
    
    return content


# Specific OpenAI model functions
def invoke_gpt_4o(
    prompt: str,
    image_path: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single-turn GPT-4o API call.
    
    Args:
        prompt: The user prompt/question
        image_path: Optional path to image file
        api_key: OpenAI API key (if not set, will use OPENAI_API_KEY env var)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        system_prompt: Optional system prompt
    
    Returns:
        Dictionary containing the API response with usage information
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    content = _prepare_openai_content(prompt, image_path)
    messages.append({"role": "user", "content": content})
    
    # Call API
    timestamp = datetime.utcnow().isoformat()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Extract message content
    message_content = response.choices[0].message.content
    if not message_content:
        if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
            message_content = f"[REFUSAL] {response.choices[0].message.refusal}"
        else:
            message_content = "[EMPTY RESPONSE]"
    
    return {
        "output": {
            "message": {
                "content": [{"text": message_content}]
            }
        },
        "stopReason": response.choices[0].finish_reason,
        "raw_response": response
    }


def invoke_gpt_4o_messages(
    messages: List[dict],
    *,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multi-turn GPT-4o API call with message history.
    
    Args:
        messages: List of message dicts in OpenAI format
        api_key: OpenAI API key
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        system_prompt: Optional system prompt (prepended to messages)
    
    Returns:
        Dictionary containing the API response with usage information
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    
    # Normalize messages to ensure proper format
    normalized_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Normalize content to proper format
        if isinstance(content, str):
            # Simple string content
            normalized_content = content
        elif isinstance(content, list):
            # Content is a list of items - ensure each text field is a string
            normalized_items = []
            for item in content:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    # Ensure text field is a string, not an array
                    if "text" in item_copy and isinstance(item_copy["text"], list):
                        item_copy["text"] = " ".join(str(x) for x in item_copy["text"])
                    elif "text" in item_copy and not isinstance(item_copy["text"], str):
                        item_copy["text"] = str(item_copy["text"])
                    normalized_items.append(item_copy)
            normalized_content = normalized_items if normalized_items else ""
        else:
            normalized_content = str(content)
        
        normalized_messages.append({"role": role, "content": normalized_content})
    
    # Prepend system prompt if provided
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(normalized_messages)
    
    # Call API
    timestamp = datetime.utcnow().isoformat()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    # Log usage
    usage = response.usage
    
    # Extract message content
    message_content = response.choices[0].message.content
    if not message_content:
        if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
            message_content = f"[REFUSAL] {response.choices[0].message.refusal}"
        else:
            message_content = "[EMPTY RESPONSE]"
    
    return {
        "output": {
            "message": {
                "content": [{"text": message_content}]
            }
        },
        "usage": {
            "inputTokens": usage.prompt_tokens,
            "outputTokens": usage.completion_tokens,
            "totalTokens": usage.total_tokens
        },
        "stopReason": response.choices[0].finish_reason,
        "raw_response": response
    }


def invoke_gpt_5(
    prompt: str,
    image_path: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single-turn GPT-5 API call.
    Note: GPT-5 uses max_completion_tokens parameter and doesn't support custom temperature.
    
    Args:
        prompt: The user prompt/question
        image_path: Optional path to image file
        api_key: OpenAI API key (if not set, will use OPENAI_API_KEY env var)
        max_tokens: Maximum tokens in response (internally converted to max_completion_tokens)
        temperature: Sampling temperature (ignored - GPT-5 only supports default)
        system_prompt: Optional system prompt
    
    Returns:
        Dictionary containing the API response with usage information
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    content = _prepare_openai_content(prompt, image_path)
    messages.append({"role": "user", "content": content})
    
    # Call API - GPT-5 uses max_completion_tokens, not max_tokens
    # GPT-5 doesn't support temperature parameter (only default value)
    timestamp = datetime.utcnow().isoformat()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        max_completion_tokens=max_tokens  # Use max_completion_tokens for GPT-5
        # temperature not included - GPT-5 only supports default
    )
    
    # Log usage
    usage = response.usage
    
    # Extract message content
    message_content = response.choices[0].message.content
    if not message_content:
        if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
            message_content = f"[REFUSAL] {response.choices[0].message.refusal}"
        else:
            message_content = "[EMPTY RESPONSE]"
    
    return {
        "output": {
            "message": {
                "content": [{"text": message_content}]
            }
        },
        "usage": {
            "inputTokens": usage.prompt_tokens,
            "outputTokens": usage.completion_tokens,
            "totalTokens": usage.total_tokens
        },
        "stopReason": response.choices[0].finish_reason,
        "raw_response": response
    }


def invoke_gpt_5_messages(
    messages: List[dict],
    *,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multi-turn GPT-5 API call with message history.
    Note: GPT-5 uses max_completion_tokens parameter and doesn't support custom temperature.
    
    Args:
        messages: List of message dicts in OpenAI format
        api_key: OpenAI API key
        max_tokens: Maximum tokens in response (internally converted to max_completion_tokens)
        temperature: Sampling temperature (ignored - GPT-5 only supports default)
        system_prompt: Optional system prompt (prepended to messages)
    
    Returns:
        Dictionary containing the API response with usage information
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    
    # Normalize messages to ensure proper format
    normalized_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Normalize content to proper format
        if isinstance(content, str):
            # Simple string content
            normalized_content = content
        elif isinstance(content, list):
            # Content is a list of items - ensure each text field is a string
            normalized_items = []
            for item in content:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    # Ensure text field is a string, not an array
                    if "text" in item_copy and isinstance(item_copy["text"], list):
                        item_copy["text"] = " ".join(str(x) for x in item_copy["text"])
                    elif "text" in item_copy and not isinstance(item_copy["text"], str):
                        item_copy["text"] = str(item_copy["text"])
                    normalized_items.append(item_copy)
            normalized_content = normalized_items if normalized_items else ""
        else:
            normalized_content = str(content)
        
        normalized_messages.append({"role": role, "content": normalized_content})
    
    # Prepend system prompt if provided
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(normalized_messages)
    
    # Call API - GPT-5 uses max_completion_tokens, not max_tokens
    # GPT-5 doesn't support temperature parameter (only default value)
    timestamp = datetime.utcnow().isoformat()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=full_messages,
        max_completion_tokens=max_tokens  # Use max_completion_tokens for GPT-5
        # temperature not included - GPT-5 only supports default
    )
    
    # Log usage
    usage = response.usage
    
    # Extract message content
    message_content = response.choices[0].message.content
    if not message_content:
        if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
            message_content = f"[REFUSAL] {response.choices[0].message.refusal}"
        else:
            message_content = "[EMPTY RESPONSE]"
    
    return {
        "output": {
            "message": {
                "content": [{"text": message_content}]
            }
        },
        "usage": {
            "inputTokens": usage.prompt_tokens,
            "outputTokens": usage.completion_tokens,
            "totalTokens": usage.total_tokens
        },
        "stopReason": response.choices[0].finish_reason,
        "raw_response": response
    }


# ============================================================================
# Google Gemini API Functions (Gemini 2.5 Pro, Flash)
# ============================================================================

def invoke_gemini(
    prompt: str,
    model_id: str = "gemini-2.5-pro",
    image_path: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single-turn Gemini API call.
    
    Args:
        prompt: The user prompt/question
        model_id: Gemini model ID (e.g., "gemini-2.5-pro", "gemini-2.5-flash")
        image_path: Optional path to image file
        api_key: Google API key (if not set, will use GOOGLE_API_KEY env var)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        system_prompt: Optional system prompt
    
    Returns:
        Dictionary containing the API response with usage information
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Please install google-genai: pip install -q -U google-genai")
    
    # Create client with API key
    client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""))
    
    # Prepare content - for new API, we just pass the text directly
    contents = prompt
    
    # Handle image if provided
    if image_path:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Determine mime type from file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(ext, "image/jpeg")
        
        # Create multimodal content
        contents = [
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_data))
        ]
    
    # Prepare config
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_prompt if system_prompt else None,
    )
    
    # Call API
    timestamp = datetime.utcnow().isoformat()
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config
    )
    
    # Extract usage metadata
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None) or 0
        total_tokens = getattr(response.usage_metadata, 'total_token_count', None) or 0
    else:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

    
    # Extract text from response - handle blocked/empty responses
    response_text = None
    
    # Try to get text from response.text first
    try:
        if hasattr(response, 'text'):
            text = response.text
            if text:
                response_text = text
    except Exception:
        pass  # response.text might raise an exception, continue to other methods
    
    # Try to extract from candidates if text is not directly available
    if not response_text and hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        
        # Try to get text from candidate.content.parts
        if hasattr(candidate, 'content'):
            content = candidate.content
            if content is not None and hasattr(content, 'parts') and content.parts is not None:
                parts_text = []
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        parts_text.append(part.text)
                if parts_text:
                    response_text = "".join(parts_text)
            
            # Try accessing text directly from content if parts didn't work
            if not response_text and hasattr(content, 'text') and content.text:
                response_text = content.text
        
        # If still no text, check finish_reason
        if not response_text and hasattr(candidate, 'finish_reason'):
            finish_reason = str(candidate.finish_reason)
            if 'SAFETY' in finish_reason.upper() or 'BLOCK' in finish_reason.upper():
                response_text = f"[BLOCKED BY SAFETY FILTERS: {finish_reason}]"
            else:
                response_text = f"[NO TEXT RETURNED: {finish_reason}]"
    
    # Final fallback
    if not response_text:
        response_text = str(response) if response else "[EMPTY RESPONSE]"
    
    # Return in similar format to bedrock
    return {
        "output": {
            "message": {
                "content": [{"text": response_text}]
            }
        },
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens
        },
        "stopReason": response.candidates[0].finish_reason if (hasattr(response, 'candidates') and response.candidates) else None,
        "raw_response": response
    }


def invoke_gemini_messages(
    *,
    messages: List[dict],
    model_id: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multi-turn Gemini API call with message history.
    
    Args:
        messages: List of message dicts in format:
                  [{"role": "user"|"model", "parts": [{"text": "..."}, ...]}]
                  or OpenAI format (will be converted)
        model_id: Gemini model ID
        api_key: Google API key
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        system_prompt: Optional system prompt
    
    Returns:
        Dictionary containing the API response with usage information
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Please install google-genai: pip install -q -U google-genai")
    
    # Create client with API key
    client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""))
    
    # Convert messages to new API format
    contents_list = []
    for msg in messages:
        role = msg.get("role", "user")
        # Map OpenAI roles to Gemini roles
        if role == "assistant":
            role = "model"
        elif role == "system":
            continue  # Skip system messages (handled by system_instruction)
        
        # Handle both formats: "content" (OpenAI) or "parts" (Gemini)
        parts = []
        if "parts" in msg:
            # Already in Gemini format
            for part in msg["parts"]:
                if "text" in part:
                    parts.append(types.Part(text=part["text"]))
                elif "inline_data" in part:
                    parts.append(types.Part(
                        inline_data=types.Blob(
                            mime_type=part["inline_data"]["mime_type"],
                            data=part["inline_data"]["data"]
                        )
                    ))
        elif "content" in msg:
            content = msg["content"]
            if isinstance(content, str):
                parts.append(types.Part(text=content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(types.Part(text=item.get("text", "")))
                        elif item.get("type") == "image_url":
                            # Handle base64 images from OpenAI format
                            url = (item.get("image_url") or {}).get("url", "")
                            if url.startswith("data:"):
                                try:
                                    header, b64 = url.split(",", 1)
                                    mime_type = header.split(";")[0].split(":")[1]
                                    import base64 as b64lib
                                    data = b64lib.b64decode(b64)
                                    parts.append(types.Part(
                                        inline_data=types.Blob(mime_type=mime_type, data=data)
                                    ))
                                except Exception:
                                    pass
        
        if parts:
            contents_list.append(types.Content(role=role, parts=parts))
    
    # Prepare config
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_prompt if system_prompt else None,
    )
    
    # Call API with message history
    timestamp = datetime.utcnow().isoformat()
    response = client.models.generate_content(
        model=model_id,
        contents=contents_list,
        config=config
    )
    
    # Extract usage metadata
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None) or 0
        total_tokens = getattr(response.usage_metadata, 'total_token_count', None) or 0
    else:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
    
    # Extract text from response - handle blocked/empty responses
    response_text = None
    
    # Try to get text from response.text first
    if hasattr(response, 'text') and response.text:
        response_text = response.text
    # Try to extract from candidates if text is not directly available
    elif hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        # Try to get text from candidate.content.parts
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
            parts_text = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    parts_text.append(part.text)
            if parts_text:
                response_text = "".join(parts_text)
        
        # If still no text, check finish_reason
        if not response_text and hasattr(candidate, 'finish_reason'):
            finish_reason = str(candidate.finish_reason)
            if 'SAFETY' in finish_reason.upper() or 'BLOCK' in finish_reason.upper():
                response_text = f"[BLOCKED BY SAFETY FILTERS: {finish_reason}]"
            else:
                response_text = f"[NO TEXT RETURNED: {finish_reason}]"
    
    # Final fallback
    if not response_text:
        response_text = str(response) if response else "[EMPTY RESPONSE]"
    
    return {
        "output": {
            "message": {
                "content": [{"text": response_text}]
            }
        },
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens
        },
        "stopReason": response.candidates[0].finish_reason if (hasattr(response, 'candidates') and response.candidates) else None,
        "raw_response": response
    }


# Convenience wrappers for specific Gemini models
def invoke_gemini_2_5_pro(
    prompt: str,
    image_path: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-turn Gemini 2.5 Pro call."""
    return invoke_gemini(
        prompt=prompt,
        model_id="gemini-2.5-pro",
        image_path=image_path,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def invoke_gemini_2_5_pro_messages(
    messages: List[dict],
    *,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-turn Gemini 2.5 Pro call."""
    return invoke_gemini_messages(
        messages=messages,
        model_id="gemini-2.5-pro",
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def invoke_gemini_3_0_pro(
    prompt: str,
    image_path: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-turn Gemini 3.0 Pro call (Preview)."""
    return invoke_gemini(
        prompt=prompt,
        # Correct Model ID for the preview release
        model_id="gemini-3-pro-preview", 
        image_path=image_path,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def invoke_gemini_3_0_pro_messages(
    messages: List[dict],
    *,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-turn Gemini 3.0 Pro call (Preview)."""
    return invoke_gemini_messages(
        messages=messages,
        # Correct Model ID for the preview release
        model_id="gemini-3-pro-preview",
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of the API functions."""
    
    # Example 1: Single-turn GPT-4o call with image
    print("=" * 60)
    print("Example 1: Single-turn GPT-4o")
    print("=" * 60)
    
    try:
        response = invoke_gpt_4o(
            prompt="What is in this image? Please describe what you see.",
            image_path="test.jpg",  # Optional
            system_prompt="You are a helpful AI assistant.",
            max_tokens=1024,
            temperature=0.7,
        )
        
        output_text = response["output"]["message"]["content"][0]["text"]
        print(f"Response: {output_text}")
        print(f"Usage: {response['usage']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Multi-turn GPT-4o conversation
    print("\n" + "=" * 60)
    print("Example 2: Multi-turn GPT-4o")
    print("=" * 60)
    
    try:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello! Can you help me?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Of course! How can I assist you?"}]},
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
        ]
        
        response = invoke_gpt_4o_messages(
            messages=messages,
            system_prompt="You are a helpful math tutor.",
            max_tokens=512,
            temperature=0.7,
        )
        
        output_text = response["output"]["message"]["content"][0]["text"]
        print(f"Response: {output_text}")
        print(f"Usage: {response['usage']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Single-turn Gemini 2.5 Pro call
    print("\n" + "=" * 60)
    print("Example 3: Single-turn Gemini 2.5 Pro")
    print("=" * 60)
    
    try:
        response = invoke_gemini_2_5_pro(
            prompt="Explain quantum computing in simple terms.",
            system_prompt="You are a science educator.",
            max_tokens=512,
            temperature=0.7,
        )
        
        output_text = response["output"]["message"]["content"][0]["text"]
        print(f"Response: {output_text}")
        print(f"Usage: {response['usage']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Multi-turn Gemini 2.5 Flash call
    print("\n" + "=" * 60)
    print("Example 4: Multi-turn Gemini 2.5 Flash")
    print("=" * 60)
    
    try:
        messages = [
            {"role": "user", "parts": [{"text": "What is the capital of France?"}]},
            {"role": "model", "parts": [{"text": "The capital of France is Paris."}]},
            {"role": "user", "parts": [{"text": "What is its population?"}]},
        ]
        
        response = invoke_gemini_2_5_pro_messages(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        
        output_text = response["output"]["message"]["content"][0]["text"]
        print(f"Response: {output_text}")
        print(f"Usage: {response['usage']}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
