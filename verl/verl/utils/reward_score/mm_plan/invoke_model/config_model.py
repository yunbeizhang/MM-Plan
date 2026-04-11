"""
config_model.py - Model configuration and endpoint mappings.

Maps model names to their corresponding invocation functions.
"""

from .invoke_bedrock_model import (
    # Bedrock multi-turn functions
    invoke_claude_sonnet_4_5_messages,
    invoke_claude_sonnet_4_messages,
    invoke_claude_sonnet_3_5_messages,
    invoke_claude_sonnet_3_7_messages,
    invoke_claude_haiku_4_5_messages,
    invoke_claude_haiku_3_messages,
    invoke_claude_opus_4_messages,
    invoke_claude_opus_4_1_messages,
)

from .invoke_api_model import (
    # API multi-turn functions
    invoke_gpt_4o_messages,
    invoke_gpt_5_messages,
    invoke_gemini_2_5_pro_messages,
    invoke_gemini_3_0_pro_messages,
)

# SageMaker endpoint name mappings
# Replace values with your own SageMaker endpoint names.
ENDPOINT_MODEL_ZOO = {
    "Qwen3-VL-8B": "YOUR_QWEN3_VL_8B_ENDPOINT",
    "InternVL3-8B": "YOUR_INTERNVL3_8B_ENDPOINT",
    "Llama-3.2-11B-Vision": "YOUR_LLAMA_3_2_11B_ENDPOINT",
}

# AWS Bedrock model function mappings
BEDROCK_MODEL_ZOO = {
    "Claude-Sonnet-4.5": invoke_claude_sonnet_4_5_messages,
    "Claude-Sonnet-4": invoke_claude_sonnet_4_messages,
    "Claude-Sonnet-3.5": invoke_claude_sonnet_3_5_messages,
    "Claude-Sonnet-3.7": invoke_claude_sonnet_3_7_messages,
    "Claude-Haiku-4.5": invoke_claude_haiku_4_5_messages,
    "Claude-Haiku-3": invoke_claude_haiku_3_messages,
    "Claude-Opus-4": invoke_claude_opus_4_messages,
    "Claude-Opus-4.1": invoke_claude_opus_4_1_messages,
}

# External API model function mappings (OpenAI, Google)
API_MODEL_ZOO = {
    "GPT-4o": invoke_gpt_4o_messages,
    "GPT-5": invoke_gpt_5_messages,
    "Gemini-2.5-pro": invoke_gemini_2_5_pro_messages,
    "Gemini-3-pro": invoke_gemini_3_0_pro_messages,
}
