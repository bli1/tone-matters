# Evaluation

- `eval_asr_policy_0106.py`: rule-based ASR evaluator with whitelist support.
- `score_hybrid_600_200.py`: GPT-judge severity scorer used for final scoring.
- `asr_readable_gt_*.csv`: whitelist / readable non-target text used by the ASR policy.

Typical use depends on the model output CSV schema. See the benchmark scripts for expected columns.
