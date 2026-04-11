# Visual Exclusivity Attacks: Automatic Multimodal Red Teaming via Agentic Planning

<p align="center">
  <a href="https://agentwild-workshop.github.io/"><b>ICLR 2026 Workshop AIWILD (Oral)</b></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.20198"><img src="https://img.shields.io/badge/arXiv-2603.20198-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://agentwild-workshop.github.io/"><img src="https://img.shields.io/badge/ICLR%202026%20Workshop-Oral-blue?style=flat-square" alt="ICLR 2026 Workshop Oral"></a>
  <a href="https://yunbeizhang.github.io/MM-Plan/"><img src="https://img.shields.io/badge/Project%20Page-GitHub%20Pages-brightgreen?style=flat-square" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/zybeich/VE-Safety"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=flat-square" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <b>Yunbei Zhang, Yingqiang Ge, Weijie Xu, Yuhui Xu, Jihun Hamm, Chandan K. Reddy</b>
</p>

## News

- **Apr 10, 2026** — Code released.
- **Apr 06, 2026** — Selected as **Oral Presentation** at ICLR 2026 AIWILD Workshop.
- **Mar 24, 2026** — Selected as **Spotlight** at ICLR 2026 AIWILD Workshop.
- **Mar 01, 2026** — Accepted to ICLR 2026 Workshop [Agents in the Wild (AIWILD): Safety, Security, and Beyond](https://agentwild-workshop.github.io/).
- **Feb 05, 2026** — Paper released on [arXiv](https://arxiv.org/abs/2603.20198).

## TODO

- [ ] Release VE-Safety dataset on HuggingFace (under DUA)
- [x] Release code

## Abstract

Current multimodal red teaming treats images as wrappers for malicious payloads via typography or adversarial noise — attacks that are structurally brittle once standard defenses expose the payload. We introduce **Visual Exclusivity (VE)**, a more resilient *Image-as-Basis* threat where harm emerges only through reasoning over visual content such as technical schematics. To systematically exploit VE, we propose **Multimodal Multi-turn Agentic Planning (MM-Plan)**, a framework that reframes jailbreaking from turn-by-turn reaction to global plan synthesis. MM-Plan trains an attacker planner via **Group Relative Policy Optimization (GRPO)** to self-discover effective multi-turn strategies without human supervision. We also introduce **VE-Safety**, a human-curated benchmark of 440 instances across 15 safety categories. MM-Plan achieves **46.3% ASR against Claude 4.5 Sonnet** and **13.8% against GPT-5**, outperforming baselines by 2–5×.

## Visual Exclusivity: A New Threat Model

<p align="center">
  <img src="static/motivation_ve.png" alt="Image-as-Wrapper vs Image-as-Basis" width="600">
</p>

Unlike prior "wrapper-based" attacks where images merely conceal text payloads, **Visual Exclusivity (VE)** exploits the model's own visual reasoning capabilities. The text query appears innocuous (e.g., "How do I assemble this?"), the image contains no adversarial noise or hidden typography, and harm materializes only when the model correctly interprets spatial/functional relationships in the image. This makes standard defenses (OCR, caption-based screening) largely ineffective.

## Method Overview

<p align="center">
  <img src="static/mm_plan_workflow.png" alt="MM-Plan Framework Overview" width="800">
</p>

**MM-Plan** reformulates multimodal jailbreaking as agentic planning. Given a harmful goal and image, the Attacker Planner generates complete multi-turn strategies in a single pass. Plans are sampled and executed against victim MLLMs, with rewards collected from a judge model. The policy is updated via GRPO based on relative plan performance.

## Installation

```bash
git clone https://github.com/yunbeizhang/MM-Plan.git
cd MM-Plan
conda create -n mm_plan python=3.10 -y
conda activate mm_plan
bash verl/scripts/install_mm_plan.sh
```

## Data Preparation

The **VE-Safety** benchmark will be released on [HuggingFace](https://huggingface.co/datasets/zybeich/VE-Safety) under a Data Use Agreement (DUA). Please follow the instructions on the dataset page to request access.

After downloading, place the parquet files under `data/VE-Safety/` and update the `data_path` in the training scripts accordingly.

## Quick Start

Train the attacker planner with GRPO on VE-Safety:

```bash
bash verl/examples/mm_plan/Qwen3/vebench.sh
```

After training, merge the FSDP sharded checkpoint into HuggingFace format:

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/checkpoint/actor \
    --target_dir /path/to/checkpoint/merged_hf_model
```

Evaluate on the test set:

```bash
python -m verl.utils.reward_score.mm_plan.evaluate_attacker_agent \
    --parquet ./data/VE-Safety/test.parquet \
    --out-jsonl ./logs_eval/vesafety \
    --model qwen3-vl-4b \
    --model-weights /path/to/merged_hf_model \
    --target-model Qwen3-VL-8B \
    --judge-model Claude-Sonnet-4.5
```

## Acknowledgements

This repo is built upon the following prior works:

- [veRL](https://github.com/volcengine/verl) — RL training framework for LLMs.
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — Base vision-language model for the attacker planner.

We sincerely thank the authors for making their code and data publicly available.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2026mmplan,
  title={Visual Exclusivity Attacks: Automatic Multimodal Red Teaming via Agentic Planning},
  author={Zhang, Yunbei and Ge, Yingqiang and Xu, Weijie and Xu, Yuhui and Hamm, Jihun and Reddy, Chandan K.},
  journal={arXiv preprint arXiv:2603.20198},
  year={2026}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Disclaimer

This repository contains research on AI safety vulnerabilities. The content is intended solely for academic research and responsible disclosure to improve the safety of multimodal AI systems.
