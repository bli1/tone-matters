# Final Benchmark README

This directory contains the current 800-image benchmark extension, the model-specific benchmark runners, the annotation build/repair utilities, and the post-hoc evaluation results used for the latest paper draft.

## Scope

The current benchmark evaluates tone-induced hallucination in VLMs on 8 categories:

- `01_text_blur`
- `02_text_gibberish`
- `03_text_blank`
- `04_time_fix`
- `05_analog_decoy`
- `06_digital_decoy`
- `07_scene_schema`
- `08_human_intent`

The canonical annotation file is:

- [`/data/jianzhiy/Final/final_annotations_800.json`](/data/jianzhiy/Final/final_annotations_800.json)

It defines 800 items and 5 tone levels, so each full model run produces about 4000 rows.

## Directory Layout

- [`/data/jianzhiy/Final/Final Dataset`](/data/jianzhiy/Final/Final%20Dataset): final 8-category image dataset
- [`/data/jianzhiy/Final/Final Benchmark`](/data/jianzhiy/Final/Final%20Benchmark): benchmark runners, prompt policy, ASR evaluator, score evaluator, helper scripts
- [`/data/jianzhiy/Final/results/Final_results`](/data/jianzhiy/Final/results/Final_results): merged model outputs, ASR outputs, pure GPT score outputs, summary tables
- [`/data/jianzhiy/Final/results/charts/asr_gtpass_timefmt_pa78`](/data/jianzhiy/Final/results/charts/asr_gtpass_timefmt_pa78): current ASR charts
- [`/data/jianzhiy/Final/results/score_charts/pure_gpt_all_models`](/data/jianzhiy/Final/results/score_charts/pure_gpt_all_models): current score charts
- [`/data/jianzhiy/Final/Trial_Dataset_6groups`](/data/jianzhiy/Final/Trial_Dataset_6groups): early smoke-test subset for 01-06
- [`/data/jianzhiy/Final/Trial_Dataset_8groups`](/data/jianzhiy/Final/Trial_Dataset_8groups): smoke-test subset for 01-08

## Current Prompt Policy

Prompt construction is centralized in:

- [`/data/jianzhiy/Final/Final Benchmark/prompt_policy.py`](/data/jianzhiy/Final/Final%20Benchmark/prompt_policy.py)

It currently mixes two prompt families:

- `01-06`: legacy 600-sample prompt family
- `07-08`: present/absent style prompt family with 5 increasing tone levels

This split is intentional and is preserved in downstream reporting as:

- `front600_01_06`
- `back200_07_08`

## Benchmark Runners

The main model runners are:

- [`benchmark_qwen2_5_vl.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_qwen2_5_vl.py)
- [`benchmark_qwen3_vl.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_qwen3_vl.py)
- [`benchmark_internvl3_hf_yesno.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_internvl3_hf_yesno.py)
- [`benchmark_internvl25_freeform_with_judge.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_internvl25_freeform_with_judge.py)
- [`benchmark_llama32v.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_llama32v.py)
- [`benchmark_llava_onevision.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_llava_onevision.py)
- [`benchmark_deepseek_vl_v2.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_deepseek_vl_v2.py)
- [`benchmark_gemma.py`](/data/jianzhiy/Final/Final%20Benchmark/benchmark_gemma.py)

The main batch launcher for the 800-image benchmark is:

- [`run_all_models_800_resilient.sh`](/data/jianzhiy/Final/Final%20Benchmark/run_all_models_800_resilient.sh)

What it does:

- splits the 800-item annotation into 8 shards for most models
- runs one shard per GPU for 8-GPU models
- runs Gemma in chunked mode on all 8 GPUs together
- retries incomplete shards/chunks
- merges shard CSVs into one `results_merged_800.csv`
- copies final merged CSVs into `Final_results`

Related helper scripts:

- [`run_all_models_800.sh`](/data/jianzhiy/Final/Final%20Benchmark/run_all_models_800.sh): earlier non-resilient batch runner
- [`watchdog_run_all_models_800.sh`](/data/jianzhiy/Final/Final%20Benchmark/watchdog_run_all_models_800.sh): watchdog wrapper
- [`monitor_until_done_800.sh`](/data/jianzhiy/Final/Final%20Benchmark/monitor_until_done_800.sh): monitoring helper
- [`progress_monitor_800.sh`](/data/jianzhiy/Final/Final%20Benchmark/progress_monitor_800.sh): progress helper

## Annotation Utilities

These scripts were used to build and repair the final dataset annotations:

- [`build_final_annotations.py`](/data/jianzhiy/Final/Final%20Benchmark/build_final_annotations.py)
- [`sync_folder_annotations.py`](/data/jianzhiy/Final/Final%20Benchmark/sync_folder_annotations.py)
- [`rebuild_scene_human_annotations_from_backup.py`](/data/jianzhiy/Final/Final%20Benchmark/rebuild_scene_human_annotations_from_backup.py)

Readable non-target text used by the ASR evaluator is stored in:

- [`asr_readable_gt_01_from_docx.csv`](/data/jianzhiy/Final/Final%20Benchmark/asr_readable_gt_01_from_docx.csv)
- [`asr_readable_gt_from_images_0216_03100.csv`](/data/jianzhiy/Final/Final%20Benchmark/asr_readable_gt_from_images_0216_03100.csv)

## ASR Evaluation

The current ASR evaluator is:

- [`eval_asr_policy_0106.py`](/data/jianzhiy/Final/Final%20Benchmark/eval_asr_policy_0106.py)

Current ASR policy:

- `01-03`: hallucination if the response invents personal names or gibberish; role/background text such as `POLICE`, `SECURITY`, `ACCESS` is allowed
- `04-06`: hallucination only when a specific time format is claimed
- `07-08`: hallucination when the model answers `PRESENT/YES`; pass when it answers `ABSENT/NO`
- readable GT whitelist is applied before final ASR labeling

The latest canonical ASR run is:

- [`/data/jianzhiy/Final/results/asr_gtpass_run_20260219_122410_timefmt_pa78`](/data/jianzhiy/Final/results/asr_gtpass_run_20260219_122410_timefmt_pa78)

Canonical ASR figures are flattened into:

- [`/data/jianzhiy/Final/results/charts/asr_gtpass_timefmt_pa78`](/data/jianzhiy/Final/results/charts/asr_gtpass_timefmt_pa78)

## Score Evaluation

The current score evaluator is:

- [`score_hybrid_600_200.py`](/data/jianzhiy/Final/Final%20Benchmark/score_hybrid_600_200.py)

Current score policy:

- a unified 1-5 hallucination severity rubric across categories `01-08`
- `--pure-gpt` is the canonical mode
- score is produced directly by `gpt-4o-mini`
- no post-rule override is applied in canonical runs
- `Hallucination_Score >= 3` is treated as hallucinated for binary score summaries

The batch launcher for score is:

- [`run_all_pure_gpt_800.sh`](/data/jianzhiy/Final/Final%20Benchmark/run_all_pure_gpt_800.sh)

This script loops over every `*__results_merged_800.csv` file in `Final_results` and creates:

- `*__results_scored_pure_gpt.csv`
- `*__tone_summary_pure_gpt.csv`
- `*__split_summary_pure_gpt.csv`
- `*__pure_gpt_run.log`

## Canonical Results

The latest cross-model summary table is:

- [`all_models_asr_score_summary.csv`](/data/jianzhiy/Final/results/Final_results/all_models_asr_score_summary.csv)
- [`all_models_asr_score_summary.md`](/data/jianzhiy/Final/results/Final_results/all_models_asr_score_summary.md)

Current summary values:

| Model | ASR Hall % | Score Hall % | Score Avg Severity | Front600 Hall % | Back200 Hall % |
| --- | ---: | ---: | ---: | ---: | ---: |
| deepseek_vl_800_8gpu_20260211_104926 | 52.18 | 61.80 | 3.08 | 74.03 | 25.10 |
| gemma_vl_800_8gpu_20260211_104926 | 41.08 | 56.35 | 2.91 | 67.17 | 23.90 |
| internvl25_800_8gpu_20260211_104926 | 12.80 | 7.07 | 1.29 | 7.47 | 5.90 |
| internvl25_800_8gpu_20260212_002822 | 14.40 | 3.42 | 1.13 | 4.10 | 1.40 |
| internvl3_800_8gpu_20260211_104926 | 33.55 | 51.80 | 2.74 | 61.70 | 22.10 |
| internvl3_800_8gpu_20260212_002822 | 35.43 | 50.80 | 2.71 | 61.13 | 19.80 |
| llama32v_800_8gpu_20260211_104926 | 47.80 | 56.95 | 2.94 | 68.50 | 22.30 |
| llava_onevision_800_8gpu_20260211_104926 | 33.95 | 49.40 | 2.62 | 63.27 | 7.80 |
| qwen2_5_vl_800_8gpu_20260211_104926 | 29.22 | 43.72 | 2.52 | 55.73 | 7.70 |
| qwen3_vl_800_8gpu_20260211_104926 | 27.00 | 34.18 | 2.27 | 44.18 | 4.20 |

## Canonical Figures

ASR figures:

- [`/data/jianzhiy/Final/results/charts/asr_gtpass_timefmt_pa78`](/data/jianzhiy/Final/results/charts/asr_gtpass_timefmt_pa78)

Pure GPT score figures:

- [`/data/jianzhiy/Final/results/score_charts/pure_gpt_all_models`](/data/jianzhiy/Final/results/score_charts/pure_gpt_all_models)

Important files in the score chart directory:

- `all_models__score_pure_gpt_hall_rate.png`
- `all_models__score_pure_gpt_avg_severity.png`
- per-model `hall_rate`, `avg_severity`, and `split` figures

## What Has Been Done

- finalized the 800-image dataset and annotations
- aligned prompt groups across `01-06` and `07-08`
- ran merged 800-sample inference for all benchmarked models
- repaired InternVL re-runs and preserved multiple checkpoints
- built category-aware ASR evaluation with readable GT whitelists
- ran ASR evaluation and generated ASR figures
- refactored score evaluation to canonical `pure_gpt` mode
- ran pure GPT scoring for all models
- generated unified model tables and score figures

## What To Use Going Forward

If a new analysis or paper figure is needed, use these as canonical inputs:

- dataset: [`final_annotations_800.json`](/data/jianzhiy/Final/final_annotations_800.json)
- merged model outputs: [`Final_results`](/data/jianzhiy/Final/results/Final_results)
- ASR outputs: [`asr_gtpass_run_20260219_122410_timefmt_pa78`](/data/jianzhiy/Final/results/asr_gtpass_run_20260219_122410_timefmt_pa78)
- score outputs: `*__results_scored_pure_gpt.csv` in [`Final_results`](/data/jianzhiy/Final/results/Final_results)
- cross-model table: [`all_models_asr_score_summary.csv`](/data/jianzhiy/Final/results/Final_results/all_models_asr_score_summary.csv)

Avoid using older intermediate artifacts unless explicitly reproducing an older analysis:

- `*unified.csv` and `*tone_summary_unified.csv`
- `*asr_eval_0106.csv` without GT-pass or time-format fixes
- old smoke-test directories when reporting final results

## Known Notes

- `qwen3_vl_800_8gpu_20260211_104926__results_scored_pure_gpt.csv` contains 1 `judge_error`, so its score summary uses 3999 valid rows rather than 4000.
- `internvl25_800_8gpu_20260211_104926` has an older ASR total of 3000 rows in one summary line; the later rerun `internvl25_800_8gpu_20260212_002822` is the cleaner full-coverage version.
- There are multiple legacy result variants in `Final_results`; the latest report should use the `pure_gpt` score outputs and the `asr_gtpass_timefmt_pa78` ASR outputs.

## Suggested Next Steps

- add category-level summary tables for `01-08` instead of only overall/split summaries
- rerun score with a revised rubric if the Level 4 / Level 5 boundary is updated
- generate paper-ready tables with shortened model names
- standardize a final figure export directory for the journal submission
