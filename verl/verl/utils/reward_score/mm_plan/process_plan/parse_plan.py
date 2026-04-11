"""
parse_plan.py - Parse generated attack plans into a fixed format.

This module handles parsing of planner JSON output with robustness for various schema variations.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _normalize_op(op: str) -> str:
    """Normalize image_operation values to standard form."""
    o = (op or "").strip().lower()
    if o in ("no_op", "noop", "no-operation"):
        return "no_operation"
    if o in ("noimage", "noimage_turn"):
        return "no_image"
    return o


def parse_response(solution_str: str) -> Dict[str, Any]:
    """
    Parse planner JSON output with robustness for various schema variations.
    
    Supports multiple formats:
    1. conversation_plan nested inside strategy
    2. conversation_plan at top level (sibling to strategy)
    3. Partial plans (strategy only, no conversation_plan)
    
    Returns a canonical shape:
        {
          "image_size": {...},
          "attack_strategy": "<string built from persona/context/approach/turns_needed>",
          "attack_plan": [
              { "turn": int|str, "image_operation": str, "parameters": dict, "attack_instruction": str },
              ...
          ],
          "parse_warning": "<optional error message>"
        }

    Returns empty dict {} for critical failures (no strategy, invalid JSON, etc.)
    """
    if not solution_str or not solution_str.strip():
        return {}
    
    # Strip code fences if present (```json ... ```)
    s = solution_str.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r',\s*([}\]])', r'\1', s)
    
    # Parse JSON
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        print(f"[parse_response] JSON decode error: {e}")
        return {}
    except Exception as e:
        print(f"[parse_response] Unexpected parsing error: {e}")
        return {}

    if not isinstance(obj, dict):
        print(f"[parse_response] Root object is not a dict: {type(obj)}")
        return {}

    # Validate strategy field
    strat = obj.get("strategy")
    if not isinstance(strat, dict):
        print(f"[parse_response] 'strategy' field missing or invalid: {type(strat)}")
        return {}

    # Build attack_strategy string from strategy fields
    persona = str(strat.get("persona", "") or "").strip()
    context = str(strat.get("context", "") or "").strip()
    approach = str(strat.get("approach", "") or "").strip()
    turns_needed = str(strat.get("turns_needed", "") or "").strip()

    parts = []
    if persona: parts.append(f"Persona: {persona}")
    if context: parts.append(f"Context: {context}")
    if approach: parts.append(f"Approach: {approach}")
    if turns_needed: parts.append(f"Turns: {turns_needed}")
    attack_strategy = " | ".join(parts)

    # Find conversation_plan (multiple locations for robustness)
    plan = None
    parse_warning = None
    
    if "conversation_plan" in strat and isinstance(strat["conversation_plan"], dict):
        plan = strat["conversation_plan"]
    elif "conversation_plan" in obj and isinstance(obj["conversation_plan"], dict):
        plan = obj["conversation_plan"]
    else:
        parse_warning = "conversation_plan not found in JSON (incomplete generation)"
        plan = {}

    # Parse turns from conversation_plan
    def _order_key(k: str):
        """Sort turns: turn_1, turn_2, ..., final_turn"""
        if k == "final_turn":
            return (10**9, k)
        m = re.match(r"turn_(\d+)$", str(k).lower())
        if m:
            return (int(m.group(1)), k)
        return (10**8, k)

    attack_plan: List[Dict[str, Any]] = []
    for k in sorted(plan.keys(), key=_order_key):
        turn_data = plan.get(k)
        if not isinstance(turn_data, dict):
            continue
        
        op = turn_data.get("image_operation", "no_operation")
        if not op or not isinstance(op, str):
            op = "no_operation"
        
        params = turn_data.get("parameters")
        if not isinstance(params, dict):
            params = {}
        
        text = turn_data.get("text", "")
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        if not text.strip():
            print(f"[parse_response] Turn '{k}' has empty text, skipping")
            continue
        
        m = re.match(r"turn_(\d+)$", str(k).lower())
        turn_id: Any = int(m.group(1)) if m else k
        
        attack_plan.append({
            "turn": turn_id,
            "image_operation": op,
            "parameters": params,
            "attack_instruction": text,
        })

    result = {
        "image_size": obj.get("image_size"),
        "attack_strategy": attack_strategy,
        "attack_plan": attack_plan,
    }
    
    if parse_warning:
        result["parse_warning"] = parse_warning
    
    return result


def format_op_context_text(op: str, params: dict) -> str:
    """
    Render a compact text stub that summarizes the image operation and target.
    Example: "[image_op] op=crop region='trigger mechanism' bbox=[200,150,500,500]"
    """
    op = _normalize_op(op)
    params = params or {}
    region = params.get("region") or params.get("query")
    bbox = params.get("bbox") or ((params.get("detection") or {}).get("bbox"))
    
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox_str = f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"
    else:
        bbox_str = "None"
    region_str = f"'{region}'" if isinstance(region, str) else "None"
    return f"[image_op] op={op} region={region_str} bbox={bbox_str}"
