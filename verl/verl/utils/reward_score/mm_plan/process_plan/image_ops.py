
from __future__ import annotations

import os
import cv2  # type: ignore
import json
import hashlib
import numpy as np
from typing import Tuple, Optional, Dict, Any

# For FigStep image generation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

__all__ = ["apply_operation", "create_figstep_image"]

# Save all operated images here
OUTPUT_DIR = f"./logs_images/{os.getenv('PROJECT_NAME', 'unknown_project')}/{os.getenv('EXP_NAME', 'unknown_exp')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Helpers: paths & serialization
# -----------------------------
def _ensure_outdir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _basename_noext(p: str) -> str:
    b = os.path.basename(p)
    return os.path.splitext(b)[0] if b else "image"

def _sanitize_token(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s) or "none"

def _bbox_token(params: Dict[str, Any], w: int = None, h: int = None) -> str:
    # prefer params['bbox'], else params['detection']['bbox']
    bb = None
    if isinstance(params.get("bbox"), (list, tuple)) and len(params["bbox"]) == 4:
        bb = params["bbox"]
    elif isinstance(params.get("detection"), dict):
        det = params["detection"]
        if isinstance(det.get("bbox"), (list, tuple)) and len(det["bbox"]) == 4:
            bb = det["bbox"]
    if bb is None:
        return "none"
    x1, y1, x2, y2 = [int(v) for v in bb]
    return f"{x1}-{y1}-{x2}-{y2}"

def _region_token(params: Dict[str, Any]) -> str:
    # prefer 'region' (if you still use this), else 'query', else 'none'
    region = params.get("region") or params.get("query") or "none"
    return _sanitize_token(str(region))

def _output_path(image_path: str, op: str, params: Dict[str, Any]) -> str:
    _ensure_outdir()
    stem = _basename_noext(image_path)
    region_tok = _region_token(params or {})
    bbox_tok = _bbox_token(params or {})
    op_tok = _sanitize_token(op or "op")
    return os.path.join(OUTPUT_DIR, f"{stem}_{op_tok}_{region_tok}_{bbox_tok}.jpg")

# -----------------------------
# BBox helpers
# -----------------------------
def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def _get_bbox(params: Dict[str, Any], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Δ CHANGED: prefer params['bbox']; else params['detection']['bbox'] (new SYSTEM_PROMPT contract).
    """
    if not isinstance(params, dict):
        return None
    if isinstance(params.get("bbox"), (list, tuple)) and len(params["bbox"]) == 4:
        x1, y1, x2, y2 = params["bbox"]
        return _clip_bbox(x1, y1, x2, y2, w, h)
    det = params.get("detection") or {}
    if isinstance(det.get("bbox"), (list, tuple)) and len(det["bbox"]) == 4:
        x1, y1, x2, y2 = det["bbox"]
        return _clip_bbox(x1, y1, x2, y2, w, h)
    return None


# -----------------------------
# Operations (bbox-aware)
# -----------------------------
def _crop(image_path: str, params: Dict[str, Any]) -> str:
    out_path = _output_path(image_path, "crop", params or {})
    if os.path.exists(out_path):
        return out_path
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    bb = _get_bbox(params or {}, w, h)
    if bb is None:
        # fallback: central crop if no bbox 
        x1, y1 = int(0.25 * w), int(0.25 * h)
        x2, y2 = int(0.75 * w), int(0.75 * h)
    else:
        x1, y1, x2, y2 = bb

    # for qwen3
    if abs(x1-x2) <= 3:
        x1 = max(0, x1-5)
        x2 = min (w, x2+5)
    if abs(y1-y2) <= 3:
        y1 = max(0, y1-5)
        y2 = min(h, y2+5)
        
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(out_path, crop)
    return out_path

def _mask(image_path: str, params: Dict[str, Any]) -> str:
    out_path = _output_path(image_path, "mask", params or {})
    if os.path.exists(out_path):
        return out_path
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    bb = _get_bbox(params or {}, w, h)
    masked = img.copy()
    if bb is None:
        # fallback: central rectangle
        x1, y1 = int(0.25 * w), int(0.25 * h)
        x2, y2 = int(0.75 * w), int(0.75 * h)
    else:
        x1, y1, x2, y2 = bb
    masked[y1:y2, x1:x2] = 0  # black rectangle
    cv2.imwrite(out_path, masked)
    return out_path

def _blur(image_path: str, params: Dict[str, Any]) -> str:
    out_path = _output_path(image_path, "blur", params or {})
    if os.path.exists(out_path):
        return out_path
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    bb = _get_bbox(params or {}, w, h)
    if bb is None:
        # blur whole image
        whole = cv2.GaussianBlur(img, (25, 25), 0)
        cv2.imwrite(out_path, whole)
        return out_path
    x1, y1, x2, y2 = bb
    out = img.copy()
    roi = out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (25, 25), 0)
    cv2.imwrite(out_path, out)
    return out_path


# -----------------------------
# Public API
# -----------------------------
def apply_operation(image_path: str, op: str, parameters: dict | None = None) -> str:
    """
    Apply an image operation and return the cached result path.

    Args:
      image_path: Path to the current image.
      op: One of 'crop', 'mask', 'blur', 'no_operation', 'no_image'.
           ('no_image' returns the input path; caller decides whether to attach it.)
      parameters: should contain 'bbox' or 'detection' {'bbox':[x1,y1,x2,y2]} under new SYSTEM_PROMPT.

    Returns:
      Path to the processed image. If the operation is NOP or fails, returns the original path.
    """
    op = (op or "no_operation").lower()
    params = parameters or {}

    if op in ("no_operation", "no_op"):
        return image_path
    if op == "no_image":
        return image_path  # caller chooses not to attach the image this turn

    if op == "crop":
        return _crop(image_path, params)
    if op == "mask":
        return _mask(image_path, params)
    if op == "blur":
        return _blur(image_path, params)

    # Unknown op → no change
    return image_path


# -----------------------------
# FigStep Image Generation
# -----------------------------

def _wrap_text_by_axes_width(text, ax, fontproperties, left_margin, right_margin):
    """
    Wrap `text` so that it fits between left_margin and (1 - right_margin)
    in *axes* coordinates, using actual rendered width.
    """
    fig = ax.figure

    # temporary text object used just for measuring
    temp = ax.text(left_margin, 0.5, "", fontproperties=fontproperties,
                   transform=ax.transAxes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    words = text.split()
    lines = []
    current = ""

    for w in words:
        candidate = (current + " " + w).strip()
        temp.set_text(candidate)
        bbox_display = temp.get_window_extent(renderer=renderer)
        bbox_axes = bbox_display.transformed(ax.transAxes.inverted())

        # if the right side goes past the allowed width, start a new line
        if bbox_axes.x1 > 1 - right_margin and current != "":
            lines.append(current)
            current = w
        else:
            current = candidate

    if current:
        lines.append(current)

    temp.remove()
    return "\n".join(lines)


def create_figstep_image(
    behavior_text: str,
    output_path: str,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    main_fontsize: int = 40,
    bullet_fontsize: int = 40,
) -> str:
    """
    Create a FigStep-style image with behavior text and empty bullet points using matplotlib.
    
    Args:
        behavior_text: The behavior description text
        output_path: Path to save the generated image
        figsize: Tuple of (width, height) for the figure in inches
        dpi: DPI (dots per inch) for the output image
        main_fontsize: Font size for the main behavior text
        bullet_fontsize: Font size for the bullet points
        
    Returns:
        Path to the generated image file
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()

    left_margin = 0.05
    right_margin = 0.05
    top_margin = 0.95

    # set up font properties for measuring & drawing (bold + DejaVu Sans)
    main_fp = FontProperties(size=main_fontsize, weight='bold', family='DejaVu Sans')
    bullet_fp = FontProperties(size=bullet_fontsize, weight='bold', family='DejaVu Sans')

    # --- 1) wrap the behavior text based on actual width ---
    wrapped = _wrap_text_by_axes_width(
        behavior_text, ax, main_fp, left_margin, right_margin
    )

    # draw the (wrapped) behavior text
    text_obj = ax.text(
        left_margin,
        top_margin,
        wrapped,
        ha="left",
        va="top",
        fontproperties=main_fp,
        transform=ax.transAxes,
    )

    # --- 2) figure out where the bottom of that text is ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_display = text_obj.get_window_extent(renderer=renderer)
    bbox_axes = bbox_display.transformed(ax.transAxes.inverted())
    text_bottom_y = bbox_axes.y0

    # --- 3) place bullets under the text ---
    gap = 0.1          # space between text and first bullet
    bullet_spacing = 0.2  # vertical space between bullets

    y = text_bottom_y - gap
    for i in range(3):
        ax.text(
            left_margin,
            y,
            f"{i+1}.",
            ha="left",
            va="top",
            fontproperties=bullet_fp,
            transform=ax.transAxes,
        )
        y -= bullet_spacing

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


def get_figstep_image_path(behavior_text: str, image_basename: str) -> str:
    """
    Generate and cache a FigStep image for a given behavior text.
    Uses the same OUTPUT_DIR as other image operations for consistency.
    
    Args:
        behavior_text: The text to render
        image_basename: Base name for the output file (without extension)
        
    Returns:
        Path to the generated FigStep image
    """
    _ensure_outdir()
    
    # Sanitize filename
    safe_basename = _sanitize_token(image_basename)
    output_path = os.path.join(OUTPUT_DIR, f"figstep_{safe_basename}.png")
    
    # Check if already exists (cache)
    if os.path.exists(output_path):
        return output_path
    
    # Generate the image
    try:
        create_figstep_image(
            behavior_text=behavior_text,
            output_path=output_path,
            figsize=(10, 8),
            dpi=200,
            main_fontsize=40,
            bullet_fontsize=40,
        )
        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to create FigStep image: {e}")
        # Return a placeholder or raise
        return output_path
