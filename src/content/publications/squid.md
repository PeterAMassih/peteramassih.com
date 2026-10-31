---
title: "SQuID: A Benchmark for Quantitative Spatial Reasoning on Satellite Imagery"
description: "A benchmark of 2,000 quantitative questions on satellite images. The best zero-shot end-to-end VLM reaches 28.1%, and the QASR baseline reaches 42.0%."
pubDate: 2026-01-15
authors: ["P. A. Massih", "E. Cosatto", "M. R. Min"]
venue: "Preprint, arXiv:2601.13401, 2026. Dataset on Hugging Face."
links:
  - { label: "arXiv", href: "https://arxiv.org/abs/2601.13401" }
  - { label: "Dataset", href: "https://huggingface.co/datasets/PeterAM4/SQuID" }
  - { label: "Code", href: "https://github.com/PeterAMassih/SQuID" }
---

SQuID is a benchmark of 2,000 quantitative questions on satellite images. Numeric answers are scored against tolerance ranges informed by human annotations. SQuID also releases the reference masks behind its answers and SQuID-Train, 75,687 training question-answer pairs on separate images.

The best zero-shot end-to-end VLM reaches 28.1%. As a baseline, QASR has an LLM write a program that segments the image and measures the masks. It reaches 42.0%. On the 1,950 mask-derived questions, running its programs on the reference masks raises accuracy from 42.2% to 82.6%. Fine-tuning Qwen2.5-VL-7B on SQuID-Train raises its accuracy from 16.1% to 40.7%.

<details>
<summary>BibTeX</summary>

```bibtex
@misc{massih2026squid,
  title         = {{SQuID}: A Benchmark for Quantitative Spatial Reasoning on Satellite Imagery},
  author        = {Massih, Peter A. and Cosatto, Eric and Min, Martin Renqiang},
  year          = {2026},
  eprint        = {2601.13401},
  archivePrefix = {arXiv}
}
```

</details>
