# Precision Prompting

**Model-Specific Error-Driven Prompt Optimization Improves Diagnostic Reasoning Across LLMs**

## Overview

This repository contains the data and prompts for the Precision Prompting framework. Each of 14 LLMs was evaluated on ICD-Bench (3,675 medical MCQs, 15 disease categories) and MedThink-Bench (500 expert-annotated cases). A single analyst model (GPT-5.4) classified each model's training-set errors into a 7-category taxonomy and generated a personalized system prompt. Each prompt was then applied to the model's test-set and validation-set errors with zero missing responses.

## Key Results

| Metric | Value |
|--------|-------|
| Models evaluated | 14 |
| Mean ICD-Bench test gain | **+10.9 pp** (95% CI 7.8–14.8) |
| Mean MedThink-Bench gain | **+11.5 pp** (95% CI 7.5–16.5) |
| Mean test flip rate | 33.2% |
| Mean MBT flip rate | 23.9% |
| Spearman ρ (baseline vs gain) | −0.73 (P = 0.003) |
| Cross-benchmark Pearson r | 0.56 (P = 0.036) |

## Repository Structure

```
├── prompts/                          # GPT-5.4-generated personalized system prompts
│   ├── gpt-oss-120b.txt
│   ├── mistral-small-24b.txt
│   ├── diffusiongemma-26b.txt
│   ├── deepseek-v4-flash.txt
│   ├── step-3.7-flash.txt
│   ├── GLM-5.2.txt
│   ├── qwen3.5-122b.txt
│   ├── qwen3.6-flash.txt
│   ├── minimax-m2.5.txt
│   ├── Baichuan-M3.txt
│   ├── medpsy-4b.txt
│   ├── GPT-5.txt
│   ├── MedSeek-V3.txt
│   └── hulu-med-flash-27b.txt
│
└── results/                          # Raw prediction data
    ├── all_results.csv               # Per-model summary (baseline, optimized, gain, flip rate)
    ├── baseline_predictions_icdbench.csv  # Per-question baseline predictions (train + test, 14 models)
    ├── baseline_predictions_mbt.csv       # Per-question baseline predictions (MedThink-Bench, 14 models)
    └── error_profiles.csv            # GPT-5.4 error classification (E1–E7 distribution per model)
```

## Data Description

### `results/all_results.csv`

Per-model summary with columns:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `icd_test_base_pct` | Baseline accuracy on ICD-Bench test set (735 questions) |
| `icd_test_opt_pct` | Optimized accuracy after applying personalized prompt |
| `icd_test_gain_pp` | Accuracy gain in percentage points |
| `icd_test_errors` | Number of baseline errors on test set |
| `icd_test_flipped` | Number of errors corrected by personalized prompt |
| `icd_test_flip_rate_pct` | Flip rate = flipped / errors × 100 |
| `mbt_base_pct` | Baseline accuracy on MedThink-Bench (500 questions) |
| `mbt_opt_pct` | Optimized accuracy on MedThink-Bench |
| `mbt_gain_pp` | MBT accuracy gain in percentage points |
| `mbt_errors` | Number of baseline MBT errors |
| `mbt_flipped` | Number of MBT errors corrected |
| `mbt_flip_rate_pct` | MBT flip rate |

### `results/baseline_predictions_icdbench.csv`

Per-question baseline predictions for all 14 models on ICD-Bench train (2,940) and test (735) splits:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `split` | "train" or "test" |
| `question_id` | Question index (0–3,674) |
| `model_answer` | Model's predicted letter (A–D) |
| `correct_answer` | Ground truth letter |
| `is_correct` | Whether the prediction matches the answer key |

### `results/baseline_predictions_mbt.csv`

Same format for MedThink-Bench (500 questions, answers A–J).

### `results/error_profiles.csv`

GPT-5.4 error classification distribution per model:

| Column | Description |
|--------|-------------|
| `model` | Model name |
| `total_errors` | Total training-set errors classified |
| `E1_pct` – `E7_pct` | Percentage of errors in each category |

**Error taxonomy:**
- **E1** Key Clue Neglect (cf. search-satisficing / premature closure)
- **E2** Common Disease Bias (cf. base-rate anchoring)
- **E3** Mechanism Confusion (no clinical analogue)
- **E4** Anchoring (cf. anchoring bias)
- **E5** Atypical Feature Discounting (cf. confirmation bias)
- **E6** Temporal Neglect (cf. age/tempo base-rate neglect)
- **E7** Question-Type Mismatch (no clinical analogue)

## 14 Models

| # | Model | ICD-Bench Base | Gain | MBT Gain |
|---|-------|---------------|------|----------|
| 1 | GLM-5.2 | 53.9% | +27.6 pp | +15.2 pp |
| 2 | diffusiongemma-26b | 54.4% | +25.7 pp | +26.0 pp |
| 3 | qwen3.5-122b | 67.2% | +13.5 pp | +16.4 pp |
| 4 | deepseek-v4-flash | 70.9% | +11.0 pp | +13.2 pp |
| 5 | Baichuan-M3 | 72.7% | +10.7 pp | +6.6 pp |
| 6 | mistral-small-24b | 70.6% | +10.2 pp | +11.4 pp |
| 7 | minimax-m2.5 | 70.7% | +7.6 pp | +4.6 pp |
| 8 | step-3.7-flash | 73.9% | +7.5 pp | +7.8 pp |
| 9 | MedSeek-V3 | 71.8% | +7.5 pp | +5.6 pp |
| 10 | medpsy-4b | 69.8% | +7.3 pp | +6.4 pp |
| 11 | qwen3.6-flash | 72.4% | +7.1 pp | +4.8 pp |
| 12 | hulu-med-flash-27b | 74.3% | +6.7 pp | +34.0 pp |
| 13 | gpt-oss-120b | 71.4% | +6.0 pp | +5.0 pp |
| 14 | GPT-5 | 75.2% | +4.2 pp | +4.6 pp |

## Citation

```bibtex
@article{jiang2025precision,
  title={Precision Prompting: Model-Specific Error-Driven Prompt Optimization Improves Diagnostic Reasoning Across Large Language Models},
  author={Jiang, Zhehan},
  journal={Nature Medicine},
  year={2025}
}
```
