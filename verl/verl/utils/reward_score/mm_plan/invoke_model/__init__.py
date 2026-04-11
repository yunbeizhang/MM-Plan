"""
invoke_model - Model invocation wrappers for different providers.

This subpackage provides:
- invoke_api_model: OpenAI (GPT) and Google (Gemini) API calls
- invoke_bedrock_model: AWS Bedrock model calls (Claude, Llama)
- invoke_endpoint_model: SageMaker endpoint calls
"""

# Import from bedrock model
from .invoke_bedrock_model import (
    invoke_bedrock,
    invoke_bedrock_messages,
    invoke_claude_sonnet_4_5,
    invoke_claude_sonnet_4,
    invoke_claude_sonnet_3_5,
    invoke_claude_sonnet_3_7,
    invoke_claude_haiku_4_5,
    invoke_claude_haiku_3,
    invoke_claude_opus_4,
    invoke_claude_opus_4_1,
    invoke_llama_3_2_11b,
    invoke_claude_sonnet_4_5_messages,
    invoke_claude_sonnet_4_messages,
    invoke_claude_sonnet_3_5_messages,
    invoke_claude_sonnet_3_7_messages,
    invoke_claude_haiku_4_5_messages,
    invoke_claude_haiku_3_messages,
    invoke_claude_opus_4_messages,
    invoke_claude_opus_4_1_messages,
    invoke_llama_3_2_11b_messages,
)

# Import from API model (OpenAI, Gemini)
from .invoke_api_model import (
    invoke_gpt_4o,
    invoke_gpt_4o_messages,
    invoke_gpt_5,
    invoke_gpt_5_messages,
    invoke_gemini,
    invoke_gemini_messages,
    invoke_gemini_2_5_pro,
    invoke_gemini_2_5_pro_messages,
    invoke_gemini_3_0_pro,
    invoke_gemini_3_0_pro_messages,
)

# Import from endpoint model (SageMaker)
from .invoke_endpoint_model import (
    get_sagemaker_client,
    call_sagemaker_endpoint,
)

# Import model zoo configurations
from .config_model import (
    ENDPOINT_MODEL_ZOO,
    BEDROCK_MODEL_ZOO,
    API_MODEL_ZOO,
)

__all__ = [
    # Bedrock single-turn
    "invoke_bedrock",
    "invoke_claude_sonnet_4_5",
    "invoke_claude_sonnet_4",
    "invoke_claude_sonnet_3_5",
    "invoke_claude_sonnet_3_7",
    "invoke_claude_haiku_4_5",
    "invoke_claude_haiku_3",
    "invoke_claude_opus_4",
    "invoke_claude_opus_4_1",
    "invoke_llama_3_2_11b",
    # Bedrock multi-turn
    "invoke_bedrock_messages",
    "invoke_claude_sonnet_4_5_messages",
    "invoke_claude_sonnet_4_messages",
    "invoke_claude_sonnet_3_5_messages",
    "invoke_claude_sonnet_3_7_messages",
    "invoke_claude_haiku_4_5_messages",
    "invoke_claude_haiku_3_messages",
    "invoke_claude_opus_4_messages",
    "invoke_claude_opus_4_1_messages",
    "invoke_llama_3_2_11b_messages",
    # API single-turn
    "invoke_gpt_4o",
    "invoke_gpt_5",
    "invoke_gemini",
    "invoke_gemini_2_5_pro",
    "invoke_gemini_3_0_pro",
    # API multi-turn
    "invoke_gpt_4o_messages",
    "invoke_gpt_5_messages",
    "invoke_gemini_messages",
    "invoke_gemini_2_5_pro_messages",
    "invoke_gemini_3_0_pro_messages",
    # SageMaker
    "get_sagemaker_client",
    "call_sagemaker_endpoint",
    # Model zoo configurations
    "ENDPOINT_MODEL_ZOO",
    "BEDROCK_MODEL_ZOO",
    "API_MODEL_ZOO",
]
