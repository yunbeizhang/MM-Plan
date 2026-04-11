"""
process_plan - Plan parsing, execution, and evaluation.

This subpackage handles:
- parse_plan: Parse planner JSON output
- execute_plan: Execute attack plans against target models
- evaluate_plan: Build judge prompts and compute rewards
- image_ops: Image manipulation operations
"""

from .parse_plan import parse_response, _normalize_op

from .execute_plan import (
    execute_plan,
    call_target_model,
)

from .evaluate_plan import (
    call_judge_model,
    format_conversation_for_judge,
    build_judge_prompt,
    build_binary_judge_prompt,
    build_granular_judge_prompt,
    append_judge_log,
    compute_reward,
    OUTPUT_DIR,
)

from .image_ops import (
    apply_operation,
    create_figstep_image,
    get_figstep_image_path,
)

__all__ = [
    # Parsing
    "parse_response",
    "_normalize_op",
    # Execution
    "execute_plan",
    "call_target_model",
    # Evaluation
    "call_judge_model",
    "format_conversation_for_judge",
    "build_judge_prompt",
    "build_binary_judge_prompt",
    "build_granular_judge_prompt",
    "append_judge_log",
    "compute_reward",
    "OUTPUT_DIR",
    # Image ops
    "apply_operation",
    "create_figstep_image",
    "get_figstep_image_path",
]
