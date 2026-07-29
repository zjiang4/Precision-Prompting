"""Configuration for Precision Prompting pipeline.

Defines model registry, API endpoints, and shared constants.
Replace API keys with your own before running.
"""

BASE_DIR = "/path/to/ICD_Benchmarks"
ICDBENCH_PATH = f"{BASE_DIR}/icdBench.parquet"
MBT_PATH = f"{BASE_DIR}/MedBenchThink_data.json"

# ===== API CONFIGURATION (replace with your keys) =====
NVIDIA_API_KEYS = ["nvapi-YOUR_KEY_1", "nvapi-YOUR_KEY_2", ...]
DASHSCOPE_API_KEY = "sk-ws-YOUR_DASHSCOPE_KEY"
BAICHUAN_API_KEY = "sk-YOUR_BAICHUAN_KEY"
PKU_API_KEYS = ["YOUR_PKU_KEY_1", "YOUR_PKU_KEY_2", "YOUR_PKU_KEY_3"]
STEP_API_KEY = "p3bk-YOUR_STEP_KEY"
DEEPSEEK_API_KEY = "sk-YOUR_DEEPSEEK_KEY"
LM_STUDIO_URL = "http://localhost:1234/api/v1/chat"

# ===== ANALYST MODEL (GPT-5.4) =====
ANALYST_URL = "https://chat.pku.edu.cn/v1/chat/completions"
ANALYST_MODEL = "gpt-5.4"

# ===== MODEL REGISTRY =====
# Format: {short_name: {provider, model_name, api_base}}
MODEL_REGISTRY = {
    "gpt-oss-120b":              {"provider": "nvidia",   "model": "openai/gpt-oss-120b"},
    "mistral-small-24b":         {"provider": "nvidia",   "model": "mistralai/mistral-small-4-119b-2603"},
    "diffusiongemma-26b":        {"provider": "nvidia",   "model": "google/diffusiongemma-26b-a4b-it"},
    "deepseek-v4-flash":         {"provider": "deepseek", "model": "deepseek-v4-flash"},
    "step-3.7-flash":            {"provider": "stepfun",  "model": "step-3.7-flash"},
    "glm-5.2":                   {"provider": "dashscope","model": "glm-5.2"},
    "qwen3.5-122b":              {"provider": "dashscope","model": "qwen3.5-122b-a10b"},
    "qwen3.6-flash":             {"provider": "dashscope","model": "qwen3.6-flash"},
    "minimax-m2.5":              {"provider": "dashscope","model": "MiniMax-M2.5"},
    "Baichuan-M3":               {"provider": "baichuan", "model": "Baichuan-M3"},
    "medpsy-4b":                 {"provider": "lmstudio", "model": "medpsy-4b"},
    "GPT-5":                     {"provider": "pku",      "model": "gpt-5"},
    "MedSeek-V3":                {"provider": "lmstudio", "model": "medseekv3-35b"},
    "hulu-med-flash-27b":        {"provider": "pku",      "model": "MedSeekV25"},
}

# ===== ERROR TAXONOMY =====
ERROR_TAXONOMY = {
    "E1": "Key Clue Neglect — overlooks a subtle but decisive lab finding or symptom",
    "E2": "Common Disease Bias — defaults to prevalent diagnosis despite atypical features",
    "E3": "Mechanism Confusion — selects correct disease but wrong pathophysiological mechanism",
    "E4": "Anchoring — fixates on one salient feature, discarding contradictory evidence",
    "E5": "Atypical Feature Discounting — dismisses features that don't fit the leading hypothesis",
    "E6": "Temporal Neglect — ignores onset age, progression speed, or treatment response timing",
    "E7": "Question-Type Mismatch — answers a different question than asked",
}

# ===== NEUTRAL BASELINE PROMPT =====
NEUTRAL_PROMPT = (
    "You are a medical expert. Answer the following multiple-choice question. "
    "First reason step-by-step, then output your final answer as a single letter: "
    "A, B, C, or D. Your final line MUST be exactly in the format: \"Answer: X\" "
    "where X is A, B, C, or D."
)
