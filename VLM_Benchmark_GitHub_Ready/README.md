# VLM Hallucination Benchmark

This repository contains the cleaned release version of the VLM hallucination benchmark used for the final experiments.

## Contents

- `dataset/Final_Dataset/`: final 8-category image benchmark dataset. Each category contains 100 images and an `annotations.json`.
- `annotations/`: final merged annotation files, including `final_annotations_800.json`.
- `benchmark/`: prompt policy and model runner scripts used to generate model outputs.
- `evaluation/`: final ASR and hallucination severity scoring scripts.
- `results/Final_results/`: final model outputs, ASR outputs, score outputs, summaries, and paper figures.
- `docs/`: original project documentation copied from the final experiment folder.

## Main Evaluation Scripts

- ASR evaluation: `evaluation/eval_asr_policy_0106.py`
- Severity scoring: `evaluation/score_hybrid_600_200.py`
- Prompt policy: `benchmark/prompt_policy.py`

## Dataset Categories

1. `01_text_blur`
2. `02_text_gibberish`
3. `03_text_blank`
4. `04_time_fix`
5. `05_analog_decoy`
6. `06_digital_decoy`
7. `07_scene_schema`
8. `08_human_intent`

## GitHub Note

The dataset is image-heavy. If pushing this repository to GitHub, use Git LFS for image and large CSV files, or uncomment `dataset/Final_Dataset/` in `.gitignore` and distribute the dataset separately.
