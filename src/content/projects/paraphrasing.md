---
title: "Reasoning-Model Paraphraser"
description: "More than doubled diversity over BART (BLEU diversity 0.51 vs 0.23) at similar BERTScore (0.95 vs 0.96) by fine-tuning DeepSeek-R1-Distill-Qwen-7B with LoRA."
pubDate: 2025-07-29
context: "Personal project"
links:
  - { label: "GitHub", href: "https://github.com/PeterAMassih/paraphrasing" }
  - { label: "Model", href: "https://huggingface.co/PeterAM4/deepseek-paraphrase" }
---

On 212 test sentences, BERTScore is 0.9521 vs 0.9587 for BART and 0.9952 for T5, and BLEU diversity is 0.5135 vs 0.2347 for BART and 0.0416 for T5. BART and T5 score higher on BERTScore. The gain is diversity.
