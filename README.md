# Visual Exclusivity Attacks: Automatic Multimodal Red Teaming via Agentic Planning

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-brightgreen)](https://yunbeizhang.github.io/MM-Plan/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/zybeich/VE-Safety)

**Yunbei Zhang, Yingqiang Ge, Weijie Xu, Yuhui Xu, Jihun Hamm, Chandan K. Reddy**


> **Note:** The code and dataset will be released soon once the arXiv submission hold is resolved. Stay tuned!

---

## Abstract

Current multimodal red teaming treats images as wrappers for malicious payloads via typography or adversarial noise. These attacks are structurally brittle, as standard defenses neutralize them once the payload is exposed.

We introduce **Visual Exclusivity (VE)**, a more resilient *Image-as-Basis* threat where harm emerges only through reasoning over visual content such as technical schematics. To systematically exploit VE, we propose **Multimodal Multi-turn Agentic Planning (MM-Plan)**, a framework that reframes jailbreaking from turn-by-turn reaction to global plan synthesis.

MM-Plan trains an attacker planner to synthesize comprehensive, multi-turn strategies, optimized via Group Relative Policy Optimization (GRPO), enabling self-discovery of effective strategies without human supervision. To rigorously benchmark this reasoning-dependent threat, we introduce **VE-Safety**, a human-curated dataset filling a critical gap in evaluating high-risk technical visual understanding.

MM-Plan achieves **46.3% attack success rate against Claude 4.5 Sonnet** and **13.8% against GPT-5**, outperforming baselines by 2-5x where existing methods largely fail.

## Visual Exclusivity: A New Threat Model

<p align="center">
  <img src="static/motivation_ve.png" alt="Image-as-Wrapper vs Image-as-Basis" width="600">
</p>

Unlike prior "wrapper-based" attacks where images merely conceal text payloads, **Visual Exclusivity (VE)** exploits the model's own visual reasoning capabilities:

- The text query appears innocuous (e.g., "How do I assemble this?")
- The image contains no adversarial noise or hidden typography
- Harm materializes only when the model correctly interprets spatial/functional relationships in the image

This dependency renders standard defenses largely ineffective: OCR cannot extract payloads that don't exist in text form, and caption-based screening cannot capture precise structural details required for harm.

## Method Overview

<p align="center">
  <img src="static/mm_plan_workflow.png" alt="MM-Plan Framework Overview" width="800">
</p>

**MM-Plan** reformulates multimodal jailbreaking as agentic planning. Given a harmful goal and image, the Attacker Planner generates complete multi-turn strategies in a single pass. Plans are sampled and executed against victim MLLMs, with rewards collected from a judge model. The policy is updated via GRPO based on relative plan performance.

### Why Agentic Planning?

| | Prior Approaches | MM-Plan (Ours) |
|---|---|---|
| **Strategy** | Sequential RL (myopic, optimizes immediate rewards) | Global planning (synthesizes complete strategy in one pass) |
| **Scaling** | Iterative search scales K<sup>N</sup> for N-turn dialogues | Linear scaling: only K x N steps for K sampled plans |
| **Attack Surface** | Wrapper-based (easily neutralized by OCR-aware filters) | Visual operations (exploits image reasoning, not text wrappers) |

## Main Results

Attack Success Rate (ASR %) across 8 frontier MLLMs on VE-Safety. MM-Plan significantly outperforms all baselines, especially on heavily defended proprietary models.

| Method | Llama-3.2-11B | InternVL3-8B | Qwen3-VL-8B | GPT-4o | GPT-5 | Sonnet 3.7 | Sonnet 4.5 | Gemini 2.5 Pro |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Direct Request | 13.4 | 27.2 | 11.9 | 5.0 | 0.6 | 4.7 | 8.4 | 9.7 |
| Direct Plan | 18.1 | 34.7 | 22.5 | 9.4 | 0.9 | 8.1 | 9.7 | 11.9 |
| FigStep | 23.8 | 44.4 | 33.1 | 6.6 | 0.6 | 13.4 | 24.4 | 11.3 |
| SI-Attack | 25.6 | 31.9 | 29.1 | 8.1 | 1.9 | 12.8 | 15.6 | 12.5 |
| SSA | 25.3 | 39.1 | 29.4 | 6.3 | 1.6 | 9.7 | 15.9 | 12.2 |
| Crescendo | 21.9 | 45.0 | 33.8 | 14.4 | 3.1 | 15.0 | 18.1 | 15.9 |
| **MM-Plan (Ours)** | **64.4** | **65.0** | **54.4** | **36.9** | **13.8** | **27.2** | **46.3** | **43.8** |

*Statistically significant improvement (p <= 0.05) over second-best method across all models.*

## VE-Safety Benchmark

We introduce **VE-Safety**, the first benchmark specifically targeting the *Image-as-Basis* threat model with real-world technical imagery.

- **440** human-curated instances across **15** safety categories
- **100%** real-world images (technical schematics, circuit diagrams, floor plans, chemical formulas)
- Designed for multi-turn attack evaluation

| Benchmark | Human-Curated | Image Type | Visual Role | Multi-Turn |
|---|:---:|---|---|:---:|
| FigStep | - | Typographic | Image-as-Wrapper | - |
| HADES | - | Typo. / Adv. Noise | Image-as-Wrapper | - |
| MM-SafetyBench | - | Typo. / SD | Image-as-Wrapper | - |
| HarmBench (MM) | - | SD / Real | Image-as-Basis | - |
| **VE-Safety (Ours)** | **Yes** | **Real** | **Image-as-Basis** | **Yes** |

## Contributions

1. **Visual Exclusivity (VE):** We formalize a new multimodal vulnerability where harmful goals require visual reasoning about image content, providing criteria that distinguish VE from wrapper-based attacks.

2. **VE-Safety Benchmark:** We construct the first benchmark targeting Image-as-Basis threats, comprising 440 human-curated instances across 15 safety categories with verified non-textual irreducibility.

3. **MM-Plan Framework:** We propose a multimodal agentic planning framework that achieves 2-5x higher attack success rates than search-based and turn-by-turn baselines across frontier MLLMs.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{zhang2025mmplan,
  title={Visual Exclusivity Attacks: Automatic Multimodal Red Teaming via Agentic Planning},
  author={Zhang, Yunbei and Ge, Yingqiang and Xu, Weijie and Xu, Yuhui and Hamm, Jihun and Reddy, Chandan K.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Warning

This repository contains research on AI safety vulnerabilities. The content is intended solely for academic research and responsible disclosure to improve the safety of multimodal AI systems.
