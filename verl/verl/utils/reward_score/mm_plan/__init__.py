"""
mm_plan - Multi-modal attack planning and evaluation module.

This package provides:
- Attack plan generation prompts
- Plan parsing and execution
- Judge-based evaluation
- Model invocation utilities (Bedrock, API, SageMaker endpoints)

Main entry points:
- training_reward.compute_score: Reward function for VERL training
- evaluate_attacker_agent.main: Standalone evaluation script for trained attackers

Package structure:
- custom_prompt/: Attacker and judge prompt templates
- process_plan/: Plan parsing, execution, and evaluation
- invoke_model/: Model invocation utilities
"""

# Main reward function for VERL training
from .training_reward import (
    parse_response,
    reward_by_judge,
    compute_score,
)

# Evaluation utilities for trained attacker agents
from .evaluate_attacker_agent import (
    read_parquet_exact,
    load_qwen3_vl,
    generate_plan_with_local_model,
    binary_eval_by_judge,
)

# Process plan functions
from .process_plan import (
    execute_plan,
    call_target_model,
    call_judge_model,
    build_binary_judge_prompt,
    build_granular_judge_prompt,
    apply_operation,
)

# Prompts
from .custom_prompt import (
    ATTACKER_SYSTEM_PROMPT,
    ATTACKER_USER_PROMPT,
    JUDGE_PROMPT_BINARY,
    JUDGE_PROMPT_GRANULAR,
)

# Model invocation
from .invoke_model import (
    ENDPOINT_MODEL_ZOO,
    BEDROCK_MODEL_ZOO,
    API_MODEL_ZOO,
    invoke_claude_sonnet_4_5,
    invoke_claude_sonnet_4_5_messages,
    get_sagemaker_client,
)

__all__ = [
    # Main entry points
    "compute_score",
    "reward_by_judge",
    "parse_response",
    # Evaluation
    "read_parquet_exact",
    "load_qwen3_vl",
    "generate_plan_with_local_model",
    "binary_eval_by_judge",
    # Process plan
    "execute_plan",
    "call_target_model",
    "call_judge_model",
    "build_binary_judge_prompt",
    "build_granular_judge_prompt",
    "apply_operation",
    # Prompts
    "ATTACKER_SYSTEM_PROMPT",
    "ATTACKER_USER_PROMPT",
    "JUDGE_PROMPT_BINARY",
    "JUDGE_PROMPT_GRANULAR",
    # Model invocation
    "ENDPOINT_MODEL_ZOO",
    "BEDROCK_MODEL_ZOO",
    "API_MODEL_ZOO",
    "invoke_claude_sonnet_4_5",
    "invoke_claude_sonnet_4_5_messages",
    "get_sagemaker_client",
]
