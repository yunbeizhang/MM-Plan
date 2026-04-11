import boto3
from typing import Dict, Any, Optional, List
import base64


def invoke_bedrock(
    prompt: str,
    model_id: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    # Create Bedrock Runtime client
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region)

    # Prepare the request parameters
    content: list[dict] = [
        {
            "text": prompt,
        }
    ]
    if image_path:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        content.append(
            {
                "image": {
                    "format": image_format,
                    "source": {
                        "bytes": image_data,
                    },
                },
            }
        )
    request_params = {
        "modelId": model_id,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "inferenceConfig": {"maxTokens": max_tokens, 
                            "temperature": temperature
                            },
    }

    # Add system prompt as a separate parameter if provided
    if system_prompt:
        request_params["system"] = [{"text": system_prompt}]

    # Invoke the model using Converse API
    response = bedrock_runtime.converse(**request_params)

    return response

# from invoke_bedrock write a funtion invoke_claude_sonet_4_5
def invoke_claude_sonnet_4_5(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_claude_haiku_4_5(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_claude_sonnet_4(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="global.anthropic.claude-sonnet-4-20250514-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_claude_sonnet_3_5(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_claude_sonnet_3_7(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_llama_3_2_11b(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="us.meta.llama3-2-11b-instruct-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_claude_haiku_3(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_claude_opus_4(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="us.anthropic.claude-opus-4-20250514-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

def invoke_claude_opus_4_1(
    prompt: str,
    image_path: Optional[str] = None,
    image_format: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return invoke_bedrock(
        prompt=prompt,
        model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
        image_path=image_path,
        image_format=image_format,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

# Multi-turn conversation

def _oai_messages_to_bedrock(messages: List[dict]) -> List[dict]:
    """
    Convert your OpenAI-style messages:
      {role: "user"|"assistant"|"system",
       content: [{type: "text"| "image_url", ...}, ...]}
    into Bedrock Converse `messages`.

    Supports:
      - {"type":"text","text": "..."}
      - {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<...>"}}
        (data: URLs only; remote HTTP URLs are ignored by design here)
    """
    br_messages: List[dict] = []

    for m in messages:
        role = m.get("role", "user")
        oai_content = m.get("content") or []

        # Map role
        br_role = role
        if role == "system":
            # Move system to 'system' param on the API (handled by caller); skip here
            # We'll just treat 'system' as user in the messages array if needed
            br_role = "user"

        br_content: List[dict] = []
        for item in oai_content:
            t = item.get("type")
            if t == "text":
                text = item.get("text", "")
                if text:
                    br_content.append({"text": text})
            elif t == "image_url":
                url = (item.get("image_url") or {}).get("url") or ""
                # Only accept data URLs (already base64). Example:
                #   data:image/jpeg;base64,<...>
                if url.startswith("data:"):
                    try:
                        header, b64 = url.split(",", 1)
                        # best-effort parse format (jpeg/png/webp/gif)
                        image_format = "jpeg"
                        if ";base64" in header and ":" in header:
                            mime = header.split(";")[0].split(":")[1]
                            if "/" in mime:
                                image_format = mime.split("/")[1]
                        br_content.append({
                            "image": {
                                "format": image_format,
                                "source": {"bytes": base64.b64decode(b64)}
                            }
                        })
                    except Exception:
                        # ignore malformed data URL
                        pass

        if br_content:
            br_messages.append({"role": br_role, "content": br_content})

    return br_messages


def invoke_bedrock_messages(
    *,
    messages: List[dict],
    model_id: str,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Multi-turn Bedrock Converse call that accepts the entire messages history.
    """
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region)

    br_messages = _oai_messages_to_bedrock(messages)

    request_params: Dict[str, Any] = {
        "modelId": model_id,
        "messages": br_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_prompt:
        request_params["system"] = [{"text": system_prompt}]

    return bedrock_runtime.converse(**request_params)


# Convenience wrappers for Sonnet & Haiku (4.5)
def invoke_claude_sonnet_4_5_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def invoke_claude_sonnet_4_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="global.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def invoke_claude_sonnet_3_5_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def invoke_claude_sonnet_3_7_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def invoke_claude_haiku_4_5_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def invoke_claude_haiku_3_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def invoke_claude_opus_4_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="us.anthropic.claude-opus-4-20250514-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )
def invoke_claude_opus_4_1_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def invoke_llama_3_2_11b_messages(
    messages: List[dict],
    *,
    system_prompt: Optional[str] = None,        
    region: str = "us-west-2",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    return invoke_bedrock_messages(
        messages=messages,
        model_id="us.meta.llama3-2-11b-instruct-v1:0",
        system_prompt=system_prompt,
        region=region,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def main():
    prompt = "What is in this image? Please describe what you see."
    system_prompt = "You are a helpful AI assistant that explains complex topics clearly."

    response = invoke_bedrock(
        prompt=prompt,
        system_prompt=system_prompt,
        image_path="test.jpg",
        image_format="jpeg",  # png | jpeg | gif | webp
        max_tokens=1024,
        temperature=0.7,
    )

    # Extract the response text
    output_text = response["output"]["message"]["content"][0]["text"]
    print(output_text)


if __name__ == "__main__":
    main()
