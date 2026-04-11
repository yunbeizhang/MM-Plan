"""
prompt - Prompt templates for attackers and judges.

Contains:
- attacker_prompt: System and user prompts for attack planner models
- judge_prompt: Templates for binary and granular evaluation
"""

from .attacker_prompt import ATTACKER_SYSTEM_PROMPT, ATTACKER_USER_PROMPT
from .judge_prompt import JUDGE_PROMPT_BINARY, JUDGE_PROMPT_GRANULAR

__all__ = [
    "ATTACKER_SYSTEM_PROMPT",
    "ATTACKER_USER_PROMPT",
    "JUDGE_PROMPT_BINARY",
    "JUDGE_PROMPT_GRANULAR",
]
