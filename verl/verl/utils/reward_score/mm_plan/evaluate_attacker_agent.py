#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_planner.py - Standalone evaluation script for attack planners.

This script:
1. Loads test data from parquet files
2. Generates attack plans using Qwen3-VL model
3. Executes plans against target models
4. Evaluates success using judge models
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

# HF / Qwen
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration

# Project-local imports
from .process_plan import (
    parse_response, 
    execute_plan, 
    call_judge_model,
    build_binary_judge_prompt,
)


# -----------------------------
# Parquet loader utilities
# -----------------------------
def _maybe_parse(x: Any) -> Any:
    """If a value is a stringified list/dict, parse it back; otherwise return as-is."""
    if isinstance(x, (list, dict)) or x is None:
        return x
    if isinstance(x, str):
        s = x.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(s)
            except Exception:
                pass
    return x


def read_parquet_exact(path: str) -> pd.DataFrame:
    """
    Read parquet so nested objects remain list/dict.
    Prefer pyarrow; if columns are strings anyway, repair them.
    """
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception:
        df = pd.read_parquet(path)

    for col in ("images", "prompt", "extra_info"):
        if col in df.columns and df[col].dtype == object and len(df) > 0:
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].map(_maybe_parse)
    return df


# -----------------------------
# Model loading
# -----------------------------
def load_qwen3_vl(model_variant: str = "qwen3-vl-4b", model_weights: Optional[str] = None):
    """
    Load a Qwen3-VL model + processor.
    - Default is Qwen/Qwen3-VL-4B-Instruct
    - If model_weights is given, it is passed directly to from_pretrained
    """
    import torch

    mv = (model_variant or "qwen3-vl-4b").lower()
    default_repo = "Qwen/Qwen3-VL-8B-Instruct" if "8" in mv else "Qwen/Qwen3-VL-4B-Instruct"
    repo = model_weights or default_repo

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(repo)
    return processor, model


def generate_plan_with_local_model(
    image_path: str,
    system_prompt_text: str,
    user_prompt_text: str,
    processor,
    model,
) -> str:
    """Generate a plan with Qwen3-VL given an image + per-row prompts."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt_text},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out_text[0]


# -----------------------------
# Binary judge evaluation
# -----------------------------
def binary_eval_by_judge(
    *,
    solution_str: str,
    extra_info: Dict[str, Any] | None,
    judge_model: str,
    target_model: str,
) -> Tuple[int, str, List[str]]:
    """
    Evaluate an attack plan with binary yes/no judge.
    
    Args:
        solution_str: The generated attack plan (JSON string)
        extra_info: Metadata including Behavior, image_description, original_image
        judge_model: Judge model to use for evaluation
        target_model: Target model to execute plan against
    
    Returns:
        Tuple of (label, judge_text, responses):
        - label: 1 for success ("yes"), 0 for failure ("no")
        - judge_text: Raw text from judge model
        - responses: List of responses from target model
        
    For failures, returns (0, error_msg, [error_msg]).
    Error types: GENERATION_ERROR, PARSE_ERROR, TARGET_ERROR, etc.
    """
    # Check if solution_str is already an error message
    if isinstance(solution_str, str) and solution_str.startswith("["):
        if any(err in solution_str for err in ["[GENERATION_ERROR", "[DATA_ERROR"]):
            print(f"[binary_eval_by_judge] Received error as input: {solution_str[:100]}")
            return 0, solution_str, [solution_str]
    
    # Parse the plan
    parsed = parse_response(solution_str)
    
    if not parsed:
        error_msg = "[PARSE_ERROR: Failed to parse JSON or missing strategy field]"
        print(f"[binary_eval_by_judge] {error_msg}")
        return 0, error_msg, [error_msg]
    
    if not parsed.get("attack_plan") or len(parsed["attack_plan"]) == 0:
        parse_warning = parsed.get("parse_warning", "No turns in conversation_plan")
        error_msg = f"[PARSE_ERROR: {parse_warning}]"
        print(f"[binary_eval_by_judge] {error_msg}")
        return 0, error_msg, [error_msg]

    # Execute the plan
    responses, _ = execute_plan(parsed, extra_info, target_model=target_model)
    
    # Concatenate responses for judge
    generation = "\n\n".join([r for r in responses if isinstance(r, str)])[:12000]

    # Extract behavior and image description
    behavior = str((extra_info or {}).get("Behavior", ""))
    img_desc = str(
        (extra_info or {}).get("RedactedImageDescription", "") 
        or (extra_info or {}).get("image_description", "")
    )

    # Build and send judge prompt
    prompt = build_binary_judge_prompt(
        behavior=behavior, 
        image_description=img_desc, 
        generation=generation
    )
    judge_text = call_judge_model(prompt, system_prompt=None, judge_model=judge_model)

    # Parse judge response
    m = re.search(r"\b(yes|no)\b", (judge_text or "").strip(), flags=re.IGNORECASE)
    label = 1 if (m and m.group(1).lower() == "yes") else 0
    
    return label, (judge_text or ""), responses


# -----------------------------
# Logging record
# -----------------------------
@dataclass
class PlanEvalLog:
    idx: int
    image_name: str
    label: int
    behavior: str
    image_description: str
    plan_text: str
    responses: str
    judge_text: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="./data/vebench/test.parquet")
    ap.add_argument("--out-jsonl", default="./logs_eval")
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="Optional row start offset")
    ap.add_argument("--model", default="qwen3-vl-4b", help="Planner model variant")
    ap.add_argument("--model-weights", default=None, help="Optional local path / HF repo")
    ap.add_argument("--tag", default="", help="Optional tag to identify this eval run")
    ap.add_argument("--target-model", default=None, help="Override TARGET_MODEL")
    ap.add_argument("--judge-model", default=None, help="Override JUDGE_MODEL")
    args = ap.parse_args()

    TARGET_MODEL = args.target_model or os.getenv("TARGET_MODEL", "Claude-Sonnet-4.5")
    JUDGE_MODEL = args.judge_model or os.getenv("JUDGE_MODEL", "Claude-Sonnet-4.5")

    # Load dataset
    df = read_parquet_exact(args.parquet)

    if args.start > 0:
        df = df.iloc[args.start:].reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].reset_index(drop=True)

    # Load planner model
    processor, model = load_qwen3_vl(args.model, args.model_weights)

    # Output setup
    os.makedirs(args.out_jsonl, exist_ok=True)
    save_path = os.path.join(args.out_jsonl, f"{TARGET_MODEL}_eval_by_{JUDGE_MODEL}{args.tag}.jsonl")
    out = open(save_path, "w", encoding="utf-8")

    yes_count, total = 0, 0

    for idx, row in df.iterrows():
        try:
            image_path = row["images"][0]["image"]
            prompts = row["prompt"]
            system_prompt_text = prompts[0]
            user_prompt_text = prompts[1]
        except Exception as e:
            print(f"[{idx}] row parse failed: {e}")
            error_msg = f"[DATA_ERROR: Failed to parse row from parquet: {e}]"
            log = PlanEvalLog(
                idx=idx,
                image_name="(parse_error)",
                label=0,
                behavior="",
                image_description="",
                plan_text="",
                responses=error_msg,
                judge_text=error_msg,
            )
            out.write(log.to_json() + "\n")
            out.flush()
            continue

        extra_info = (row.get("extra_info") or {}) if isinstance(row, dict) else (row["extra_info"] if "extra_info" in row else {})
        if not isinstance(extra_info, dict):
            extra_info = _maybe_parse(extra_info) or {}

        behavior = extra_info.get("Behavior", "")
        image_desc = extra_info.get("RedactedImageDescription", "") or extra_info.get("image_description", "")

        # Generate plan
        try:
            plan_text = generate_plan_with_local_model(
                image_path=image_path,
                system_prompt_text=system_prompt_text,
                user_prompt_text=user_prompt_text,
                processor=processor,
                model=model,
            )
        except Exception as e:
            print(f"[{idx}] plan generation failed: {e}")
            plan_text = f"[GENERATION_ERROR: Planner model failed to generate plan: {e}]"

        # Prepare extra info
        ei = dict(extra_info)
        ei.update({
            "original_image": image_path,
            "Behavior": behavior,
            "RedactedImageDescription": image_desc,
            "image_description": image_desc,
            "image_name": os.path.basename(image_path) if isinstance(image_path, str) else None,
        })

        # Execute and judge
        try:
            label, judge_text, responses = binary_eval_by_judge(
                solution_str=plan_text,
                extra_info=ei,
                judge_model=JUDGE_MODEL,
                target_model=TARGET_MODEL,
            )
        except Exception as e:
            print(f"[{idx}] binary_eval_by_judge failed: {e}")
            error_msg = f"[EXECUTION_ERROR: Failed during plan execution or judging: {e}]"
            label, judge_text, responses = 0, error_msg, [error_msg]

        yes_count += int(label == 1)
        total += 1

        log = PlanEvalLog(
            idx=idx,
            image_name=str(extra_info.get("image_name")),
            label=int(label),
            behavior=str(behavior),
            image_description=str(image_desc),
            plan_text=str(plan_text),
            responses=str(responses),
            judge_text=str(judge_text),
        )
        out.write(log.to_json() + "\n")
        out.flush()
        print(f"[{idx}] label={'yes' if label else 'no'}")

    out.close()

    if total > 0:
        asr = yes_count / float(total)
        print(f"\nDone. ASR (yes-rate) = {asr:.4f} over {total} samples.")
    else:
        print("\nDone. No valid samples processed.")


if __name__ == "__main__":
    main()
