"""Iterative prompt optimization via cross-model error comparison.

Core insight: For each error the target model makes, at least one other model
likely answered correctly.  By comparing the target model's (wrong) reasoning
with correct models' reasoning, GPT5.5 can diagnose the specific failure mode
and design a structured protocol to prevent it.

Workflow:
  1. Load ALL available model results (V3, V25, Baichuan-M3, NVIDIA)
  2. Split ICD-Bench 80/20
  3. For target model's TRAINING errors: find which other models got each right
  4. Build analyst prompt: show error Q + target's wrong reasoning + correct models' reasoning
  5. GPT5.5 outputs a structured diagnostic protocol (6-step style)
  6. Test protocol on TEST set errors, measure flip rate
  7. Iterate: feed remaining errors + refinement → GPT5.5
  8. Save best prompt
  9. Cross-validate on MedBenchThink errors

Usage:
    python analysis/optimize_prompt.py --model V3 --iterations 5
    python analysis/optimize_prompt.py --model V3 --analyst-only
    python analysis/optimize_prompt.py --model V3 --max-test-errors 30
"""
import argparse, os, sys, re, json, time, random, csv
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from benchmark_scripts.benchmark_config import (
    BASE_DIR, RESULTS_DIR, ICDBENCH_PATH, MBT_PATH, CLEANED_DIR,
    MODEL_REGISTRY, NVIDIA_API_KEYS, ANSWER_PATTERNS,
)

ANALYST_URL = "https://chat.pku.edu.cn/v1/chat/completions"
ANALYST_KEY = "JiangZheHan_nOeBhmuSIa5s"
ANALYST_MODEL = "MedSeekV25"

# Map filename-based model short names to MODEL_REGISTRY keys
FILENAME_TO_REGISTRY = {
    "deepseek-v4-flash": "deepseek-v4",
    "mistral-small-4-119b-2603": "mistral-small",
    "step-3.7-flash": "step-flash",
    "qwen3.5-122b-a10b": "qwen-3.5",
    "gpt-oss-120b": "gpt-oss",
    "diffusiongemma-26b-a4b-it": "diffusiongemma",
    "llama-4-maverick-17b-128e-instruct": "llama-4-maverick",
    "minimax-m3": "minimax-m3",
    "qwen3.6-flash": "qwen-3.6-flash",
    "glm-5.2": "glm-5.2",
    "MiniMax-M2.5": "minimax-m2.5",
    "minimax-m2.5": "minimax-m2.5",
    "gpt-5": "gpt-5",
}
REVERSE_MODEL_MAP = {v: k for k, v in FILENAME_TO_REGISTRY.items()}

# ===== HELPER: flush print =====
def pf(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ===== ANALYST CALL =====
def call_analyst(messages, temperature=0.3, timeout_sec=180):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ANALYST_KEY}",
    }
    data = {
        "model": ANALYST_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 12000,
        "temperature": temperature,
    }
    for attempt in range(3):
        try:
            resp = requests.post(ANALYST_URL, headers=headers, json=data, stream=True, timeout=timeout_sec)
            resp.raise_for_status()
            content = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line: continue
                if not line.startswith("data: "): continue
                payload = line[6:].strip()
                if payload == "[DONE]": break
                try:
                    chunk = json.loads(payload)
                    choices = chunk.get("choices")
                    if not choices: continue
                    delta = choices[0].get("delta", {})
                    c = delta.get("content", "")
                    if c: content += c
                    if choices[0].get("finish_reason") == "stop": break
                except: pass
            if "</think>" in content:
                content = content.split("</think>", 1)[1]
            result = content.strip()
            if result: return result
            pf(f"  [Analyst empty, attempt {attempt+1}]")
        except requests.exceptions.Timeout:
            pf(f"  [Analyst timeout, attempt {attempt+1}]")
        except Exception as e:
            pf(f"  [Analyst error, attempt {attempt+1}]: {e}")
        time.sleep(3)
    return None

def extract_letter(text, pattern=r'Answer:\s*([A-J])'):
    if not text or pd.isna(text): return None
    text = str(text)
    # Priority: most specific answer patterns first
    for p in [
        pattern,  # Answer: X
        r'[Cc]orrect [Aa]nswer[:\s]*[\(\*]*([A-J])',  # Correct Answer: X
        r'correct answer is\s*[\(\*]*([A-J])',  # correct answer is X
        r'[Aa]nswer\s+is\s+[\(\[]*([A-J])',  # answer is X
        r'[Cc]hoice\s+is\s+([A-J])',  # choice is X
        r'[Ff]inal [Oo]utput[:\s]*([A-J])',  # Final Output: X
        r'boxed\{([A-J])\}',  # \boxed{X}
        r'\\boxed\{([A-J])\}',  # \\boxed{X}
        r'\*\*([A-J])\*\*',  # **X**
        r'\*\*([A-J])[\)\.]',  # **X)** or **X.
    ]:
        m = re.search(p, text)
        if m: return m.group(1).upper()
    # Last resort: standalone letter near end of text only
    lines = text.strip().split('\n')
    for line in reversed(lines):
        m = re.search(r'\b([A-J])\b', line)
        if m: return m.group(1).upper()
    return None

# ===== LOAD ALL MODEL RESULTS =====
KNOWN_ICD_PATHS = {
    "V3": os.path.join(BASE_DIR, "results_v3.parquet"),
    "V25": os.path.join(BASE_DIR, "results_v25.parquet"),
    "Baichuan-M3": os.path.join(BASE_DIR, "results_baichuan.parquet"),
}

def load_all_icd_results():
    """Load results from ALL available models (legacy + NVIDIA)."""
    all_results = {}
    # Legacy models
    for name, path in KNOWN_ICD_PATHS.items():
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
            except Exception:
                pf(f"  Skipping corrupted legacy file: {path}")
                continue
            if 'predicted' not in df.columns:
                df['predicted'] = df['response'].apply(
                    lambda x: extract_letter(x, r'Answer:\s*([A-D])') if pd.notna(x) else None)
            df['key_letter'] = df['answer_key'].astype(str).str.extract(r'([A-D])')[0]
            df['correct'] = df['predicted'] == df['key_letter']
            all_results[name] = df
            pf(f"  Loaded {name}: {len(df)} rows, {(df['correct'].sum() if 'correct' in df.columns else 0)} correct")

    # NVIDIA models
    if os.path.exists(RESULTS_DIR):
        for fname in sorted(os.listdir(RESULTS_DIR)):
            if not fname.endswith('.parquet') or 'icdbench' not in fname:
                continue
            try:
                df = pd.read_parquet(os.path.join(RESULTS_DIR, fname))
            except Exception:
                pf(f"  Skipping corrupted file: {fname}")
                continue
            if 'predicted' not in df.columns:
                df['predicted'] = df['response'].apply(
                    lambda x: extract_letter(x, r'Answer:\s*([A-D])') if pd.notna(x) else None)
            if 'answer_key' in df.columns:
                df['key_letter'] = df['answer_key'].astype(str).str.extract(r'([A-D])')[0]
            elif 'index' in df.columns:
                # For in-progress files, merge answer key from ICD dataset
                icd_temp = pd.read_parquet(ICDBENCH_PATH)
                icd_temp['key_letter'] = icd_temp['answer'].astype(str).str.extract(r'([A-D])')[0]
                df['key_letter'] = icd_temp.loc[df['index'], 'key_letter'].values if len(icd_temp) >= len(df) else None
            df['correct'] = df['predicted'] == df['key_letter'] if 'predicted' in df.columns and 'key_letter' in df.columns else True
            # Use model_short as name
            model_short = fname.replace('results_', '', 1).replace('_icdbench.parquet', '')
            all_results[model_short] = df
            pf(f"  Loaded {model_short}: {len(df)} rows, {df['correct'].sum()} correct")

    return all_results

def resolve_registry_key(model_key):
    """Map filename-based short name or registry key to registry entry."""
    r = MODEL_REGISTRY.get(model_key)
    if not r and model_key in FILENAME_TO_REGISTRY:
        r = MODEL_REGISTRY.get(FILENAME_TO_REGISTRY[model_key])
    return r

def load_mbt_results(model_key):
    known_mbt = {
        "V3": os.path.join(CLEANED_DIR, "results/raw/medbenchthink_v3_checkpoint.parquet"),
        "V25": os.path.join(CLEANED_DIR, "results/raw/medbenchthink_v25_checkpoint.parquet"),
        "Baichuan-M3": os.path.join(CLEANED_DIR, "results/raw/medbenchthink_m3_checkpoint.parquet"),
    }
    if model_key in known_mbt and os.path.exists(known_mbt[model_key]):
        return pd.read_parquet(known_mbt[model_key])

    # Try NVIDIA MBT results
    registry = resolve_registry_key(model_key)
    if registry:
        mshort = registry["model_name"].split("/")[-1]
    else:
        mshort = model_key
    nv_path = os.path.join(RESULTS_DIR, f"results_{mshort}_medbenchthink.parquet")
    if os.path.exists(nv_path):
        return pd.read_parquet(nv_path)
    return None

# ===== TARGET MODEL CALL =====
def call_target_model(model_key, prompt, question_text, benchmark="icdbench"):
    # Try direct registry key first, then filename-based mapping
    registry = MODEL_REGISTRY.get(model_key)
    if not registry and model_key in FILENAME_TO_REGISTRY:
        registry = MODEL_REGISTRY.get(FILENAME_TO_REGISTRY[model_key])
    if not registry:
        pf(f"  Unknown model: {model_key}")
        return None
    provider = registry["provider"]
    pattern = ANSWER_PATTERNS.get(benchmark, r'Answer:\s*([A-J])')

    if provider == "lmstudio":
        payload = {"model": registry["model_name"], "system_prompt": prompt, "input": question_text}
        try:
            resp = requests.post(f"{registry['api_base']}/api/v1/chat", json=payload, timeout=300)
            resp.raise_for_status()
            body = resp.json()
            for out in body.get("output", []):
                if out.get("type") in ("message", None):
                    return out.get("content", "")
            return ""
        except Exception as e:
            pf(f"  [V3 call error: {e}]")
            return None

    elif provider == "nvidia":
        keys = registry.get("api_keys", NVIDIA_API_KEYS) if "api_keys" in registry else NVIDIA_API_KEYS
        api_key = random.choice(keys)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        url = registry["api_base"].rstrip("/") + "/chat/completions"
        data = {"model": registry["model_name"],
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question_text}],
                "temperature": 0.3, "max_tokens": 12000, "stream": False}
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=300)
            resp.raise_for_status()
            body = resp.json()
            msg = body["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                content = f"<reasoning>\n{reasoning}\n</reasoning>\n<message>\n{content}\n</message>"
            return content
        except Exception as e:
            return None

    elif provider == "pku":
        keys = registry.get("api_keys", [ANALYST_KEY])
        api_key = random.choice(keys)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        url = registry["api_base"]
        is_gpt5 = registry.get("gpt5_mode", False)
        if is_gpt5:
            data = {"model": registry["model_name"],
                    "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question_text}],
                    "max_completion_tokens": 12000}
        else:
            data = {"model": registry["model_name"],
                    "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question_text}],
                    "temperature": 0.3, "max_tokens": 12000, "stream": False}
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=300)
            resp.raise_for_status()
            body = resp.json()
            msg = body["choices"][0]["message"]
            content = msg.get("content", "")
            if not is_gpt5:
                reasoning = msg.get("reasoning", msg.get("reasoning_content", ""))
                if reasoning:
                    content = f"<reasoning>\n{reasoning}\n</reasoning>\n<message>\n{content}\n</message>"
            return content
        except Exception as e:
            pf(f"  [PKU call error: {e}]")
            return None

    elif provider == "baichuan":
        api_key = registry["api_key"]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        url = registry["api_base"].rstrip("/") + "/chat/completions"
        data = {"model": registry["model_name"],
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question_text}],
                "temperature": 0.3, "max_tokens": 12000, "stream": False}
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=300)
            resp.raise_for_status()
            body = resp.json()
            msg = body["choices"][0]["message"]
            return msg.get("content", "")
        except Exception as e:
            pf(f"  [Baichuan call error: {e}]")
            return None

    elif provider == "dashscope":
        api_key = registry["api_key"]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        url = registry["api_base"].rstrip("/") + "/chat/completions"
        data = {"model": registry["model_name"],
                "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question_text}],
                "temperature": 0.3, "max_tokens": 12000, "stream": False}
        if "glm" in registry["model_name"]:
            data["extra_body"] = {"enable_thinking": True}
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=300)
            resp.raise_for_status()
            body = resp.json()
            msg = body["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning", msg.get("reasoning_content", ""))
            if reasoning:
                content = f"<reasoning>\n{reasoning}\n</reasoning>\n<message>\n{content}\n</message>"
            return content
        except Exception as e:
            pf(f"  [DashScope call error: {e}]")
            return None

    return None

def test_prompt_on_errors(model_key, prompt, error_df, id_col, benchmark="icdbench"):
    pattern = ANSWER_PATTERNS.get(benchmark, r'Answer:\s*([A-J])')
    flipped = 0; total = 0; results = []
    for idx, row in error_df.iterrows():
        qid = row.get(id_col, idx)
        question_text = row.get("question_text", "")
        correct_letter = row.get("key_letter_y", row.get("key_letter", ""))
        if isinstance(correct_letter, str):
            cm = re.search(r'([A-J])', correct_letter)
            correct_letter = cm.group(1) if cm else correct_letter
        response = call_target_model(model_key, prompt, question_text, benchmark)
        if response is None:
            results.append({"question_id": qid, "response": None, "predicted": None, "correct": None})
            continue
        pred = extract_letter(response, pattern)
        is_correct = 1 if (pred and pred == correct_letter) else 0
        total += 1
        if is_correct: flipped += 1
        results.append({"question_id": qid, "response": response, "predicted": pred, "correct": is_correct})
        time.sleep(0.3)
    flip_rate = (flipped / total * 100) if total > 0 else 0
    return flip_rate, results

# ===== ANALYST PROMPT CONSTRUCTION =====
SYSTEM_PROMPT_ANALYST = """You are an expert medical educator and prompt engineer.

Your task: analyze WHY a specific LLM makes diagnostic reasoning errors by comparing 
its (wrong) output with outputs from models that answered correctly on the SAME questions.

Then synthesize a STRUCTURED DIAGNOSTIC PROTOCOL as a system prompt that will prevent 
those errors.  The protocol should be step-by-step, concrete, and specific to the error 
patterns observed.

Output format:
- A complete system prompt with numbered steps
- Each step should be an actionable instruction (not vague advice)
- Include specific examples of traps to avoid
- The final line must require "Answer: X" format
- Do NOT include meta-commentary, just the prompt text itself"""


def build_analyst_initial_prompt(target_model, train_errors_df, icd_df, all_results, sample_n=8):
    """Build a comprehensive prompt for GPT5.5 that includes cross-model comparison."""
    error_examples = []
    target_results = all_results.get(target_model)
    if target_results is None:
        return None

    id_col = 'question_id' if 'question_id' in target_results.columns else 'index'

    for _, row in train_errors_df.head(sample_n).iterrows():
        qid = row.get(id_col, row.name)
        # Get the question text
        qrow = icd_df.loc[qid] if qid in icd_df.index else icd_df.iloc[0]
        question = str(qrow.get("question", ""))
        options = str(qrow.get("options", ""))
        correct_letter = str(qrow.get("key_letter", row.get("key_letter_y", "")))
        if isinstance(correct_letter, str):
            cm = re.search(r'([A-D])', correct_letter)
            correct_letter = cm.group(1).upper() if cm else correct_letter.upper()

        # Get target model's response
        target_resp = row.get("response", "")

        # Find models that answered correctly on this question
        correct_responses = {}
        for mname, mdf in all_results.items():
            if mname == target_model:
                continue
            if id_col in mdf.columns:
                mrow = mdf[mdf[id_col] == qid]
            else:
                mrow = mdf.loc[mdf.index == qid] if qid in mdf.index else pd.DataFrame()
            if len(mrow) > 0:
                mresp = mrow.iloc[0].get("response", "")
                mcorrect = mrow.iloc[0].get("correct", False)
                if mcorrect and mresp and pd.notna(mresp):
                    correct_responses[mname] = str(mresp)[:500]

        error_examples.append({
            "question_id": int(qid) if isinstance(qid, (int, np.integer)) else str(qid),
            "question": question[:300],
            "options": options[:300],
            "correct_answer": correct_letter,
            f"{target_model}_answer": str(row.get("predicted", "")),
            f"{target_model}_reasoning": str(target_resp)[:500] if pd.notna(target_resp) else "",
            "other_models_correct": correct_responses,
        })

    user_message = json.dumps({
        "task": f"Analyze and fix diagnostic errors made by LLM '{target_model}'",
        "method": "Compare each error against models that answered correctly",
        "total_training_errors": int(train_errors_df.shape[0]),
        "error_samples": error_examples,
        "instruction": (
            "For each error sample, compare the WRONG answer (target model) with "
            "the CORRECT answers (other models).  Identify the specific reasoning failure. "
            "Then synthesize a STRUCTURED DIAGNOSTIC PROTOCOL (numbered steps) as a system prompt "
            "that will correct these failures.  The protocol must be concrete and actionable."
        ),
    }, indent=2, ensure_ascii=False)

    return [
        {"role": "system", "content": SYSTEM_PROMPT_ANALYST},
        {"role": "user", "content": user_message},
    ]


def build_analyst_refine_prompt(target_model, current_prompt, remaining_errors_df, icd_df, all_results, sample_n=5):
    """Build a refinement prompt for GPT5.5 with remaining errors and current prompt."""
    error_examples = []
    target_results = all_results.get(target_model)
    id_col = 'question_id' if target_results is not None and 'question_id' in target_results.columns else 'index'

    for _, row in remaining_errors_df.head(sample_n).iterrows():
        qid = row.get(id_col, row.name)
        qrow = icd_df.loc[qid] if qid in icd_df.index else icd_df.iloc[0]
        correct_letter = str(qrow.get("key_letter", ""))
        cm = re.search(r'([A-D])', correct_letter)
        correct_letter = cm.group(1).upper() if cm else correct_letter
        target_resp = row.get("response", "")

        correct_responses = {}
        for mname, mdf in all_results.items():
            if mname == target_model: continue
            if id_col in mdf.columns:
                mrow = mdf[mdf[id_col] == qid]
            else:
                mrow = mdf.loc[mdf.index == qid] if qid in mdf.index else pd.DataFrame()
            if len(mrow) > 0 and mrow.iloc[0].get("correct", False):
                resp = mrow.iloc[0].get("response", "")
                if pd.notna(resp): correct_responses[mname] = str(resp)[:500]

        error_examples.append({
            "question": str(qrow.get("question", ""))[:300],
            "correct_answer": correct_letter,
            f"{target_model}_wrong_reasoning": str(target_resp)[:500],
            "other_models_correct": correct_responses,
        })

    user_message = json.dumps({
        "task": f"Refine the diagnostic protocol for LLM '{target_model}'",
        "current_protocol": current_prompt,
        "remaining_errors": error_examples,
        "instruction": (
            "The current protocol did NOT fix these remaining errors.  Analyze why the current "
            "protocol was insufficient for these specific cases, then output an IMPROVED protocol. "
            "Output ONLY the new protocol text — a numbered step-by-step system prompt."
        ),
    }, indent=2, ensure_ascii=False)

    return [
        {"role": "system", "content": SYSTEM_PROMPT_ANALYST},
        {"role": "user", "content": user_message},
    ]


# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--analyst-only", action="store_true")
    parser.add_argument("--max-test-errors", type=int, default=30)
    parser.add_argument("--train-idx", default=None)
    args = parser.parse_args()

    # Load split indices
    train_idx_path = args.train_idx or os.path.join(CLEANED_DIR, "analysis/train_idx.npy")
    test_idx_path = os.path.join(CLEANED_DIR, "analysis/test_idx.npy")
    if not os.path.exists(train_idx_path) or not os.path.exists(test_idx_path):
        pf("Run split_icdbench.py first."); return
    train_idx = set(np.load(train_idx_path))
    test_idx = set(np.load(test_idx_path))

    # Load ICD-Bench data
    icd_df = pd.read_parquet(ICDBENCH_PATH)
    icd_df["key_letter"] = icd_df["answer"].astype(str).str.extract(r'([A-D])')[0]
    icd_df["question_text"] = icd_df.apply(
        lambda r: f"Question: {r['question']}\n\nOptions:\n{r['options']}", axis=1)

    # Load ALL model results
    pf("\nLoading all model results...")
    all_results = load_all_icd_results()
    if args.model not in all_results:
        pf(f"Model '{args.model}' not found in results"); return

    target_results = all_results[args.model]
    id_col = 'question_id' if 'question_id' in target_results.columns else 'index'

    # Split train/test
    train_mask = target_results[id_col].isin(train_idx)
    test_mask = target_results[id_col].isin(test_idx)
    train_results = target_results[train_mask]
    test_results = target_results[test_mask]

    train_errors = train_results[~train_results['correct'] & train_results['predicted'].notna()]
    test_errors = test_results[~test_results['correct'] & test_results['predicted'].notna()]

    # Merge with icd question text
    icd_cols = ["question_text", "key_letter", "options", "question"]
    train_errors_raw = train_errors.drop(columns=["key_letter"], errors="ignore").merge(
        icd_df[icd_cols], left_on=id_col, right_index=True, how="left")
    test_errors_raw = test_errors.drop(columns=["key_letter"], errors="ignore").merge(
        icd_df[icd_cols], left_on=id_col, right_index=True, how="left")

    if args.max_test_errors and len(test_errors_raw) > args.max_test_errors:
        pf(f"  Limiting test errors: {len(test_errors_raw)} → {args.max_test_errors}")
        test_errors_raw = test_errors_raw.sample(n=args.max_test_errors, random_state=42)

    pf(f"\n{'='*60}")
    pf(f"Prompt Optimization for {args.model}")
    pf(f"{'='*60}")
    pf(f"  Train: {len(train_results)} Q, {len(train_errors)} errors")
    pf(f"  Test:  {len(test_results)} Q, {len(test_errors)} errors to test on")
    pf(f"  Reference models: {[m for m in all_results.keys() if m != args.model]}")
    pf(f"  Iterations: {args.iterations}")

    # Build initial analyst prompt with cross-model comparison
    pf(f"\n  --- Iteration 1/{args.iterations} ---")
    pf("  Calling GPT5.5 with cross-model error comparison...")
    analyst_msgs = build_analyst_initial_prompt(
        args.model, train_errors_raw, icd_df, all_results, sample_n=8)

    if args.analyst_only:
        pf("\n  [Dry run: generating protocol...]")
        initial_prompt = call_analyst(analyst_msgs)
        if initial_prompt:
            out_path = os.path.join(CLEANED_DIR, f"optimized_prompt_{args.model}_draft.txt")
            with open(out_path, "w") as f:
                f.write(initial_prompt)
            pf(f"\n{'-'*40}\n{initial_prompt}\n{'-'*40}")
            pf(f"Draft saved to {out_path}")
        else:
            pf("  Analyst returned nothing")
        return

    # ===== OPTIMIZATION LOOP =====
    NEUTRAL = ("You are a medical expert. Answer the multiple-choice question. "
               "First reason step-by-step, then output \"Answer: X\" (A/B/C/D).")
    best_prompt = NEUTRAL
    best_flip_rate = -1.0
    history = []

    for i in range(args.iterations):
        pf(f"\n  --- Iteration {i+1}/{args.iterations} ---")

        if i == 0:
            prompt = call_analyst(analyst_msgs)
            if not prompt:
                pf("    GPT5.5 failed, using neutral prompt")
                prompt = NEUTRAL
        else:
            # Refine: show remaining errors + current prompt
            last = history[-1]
            remaining_test_details = [r for r in last["details"] if not r.get("correct")]
            # Build refine prompt with remaining errors
            remaining_ids = [r["question_id"] for r in remaining_test_details]
            remaining_df = test_errors_raw[test_errors_raw[id_col].isin(remaining_ids)]
            
            pf(f"    Remaining errors: {len(remaining_df)}")
            if len(remaining_df) == 0:
                pf("    All errors fixed!  Using same prompt.")
                prompt = last["prompt"]
            else:
                refine_msgs = build_analyst_refine_prompt(
                    args.model, last["prompt"], remaining_df, icd_df, all_results, sample_n=5)
                prompt = call_analyst(refine_msgs)
                if not prompt:
                    pf("    GPT5.5 refine failed, reusing previous")
                    prompt = last["prompt"]

        pf(f"    Prompt preview: {prompt[:150]}...")

        # Test on test errors
        pf(f"    Testing on {len(test_errors_raw)} test errors (via {args.model})...")
        flip_rate, test_details = test_prompt_on_errors(
            args.model, prompt, test_errors_raw, id_col, "icdbench")
        pf(f"    Test flip rate: {flip_rate:.1f}% ({sum(1 for r in test_details if r.get('correct'))}/{len(test_details)})")

        history.append({"iteration": i+1, "prompt": prompt,
                        "flip_rate": flip_rate, "details": test_details})

        if flip_rate > best_flip_rate:
            best_flip_rate = flip_rate
            best_prompt = prompt
            pf("    ★ New best prompt!")

        time.sleep(2)

    # ===== SAVE (before MBT) =====
    best_iter = max(history, key=lambda h: h["flip_rate"])["iteration"]
    summary = (
        f"Model: {args.model}\n"
        f"Best prompt (iteration {best_iter}, flip rate {best_flip_rate:.1f}%):\n"
        f"{'='*50}\n"
        f"{best_prompt}\n"
        f"{'='*50}\n"
        f"ICD-Bench test flip rate: {best_flip_rate:.1f}%\n"
    )
    out_path = os.path.join(CLEANED_DIR, f"optimized_prompt_{args.model}.txt")
    with open(out_path, "w") as f:
        f.write(summary)
    pf(f"\nICD-Bench results saved to {out_path}")

    hist_path = os.path.join(CLEANED_DIR, f"optimization_history_{args.model}.csv")
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "flip_rate", "prompt_preview"])
        for h in history:
            w.writerow([h["iteration"], f"{h['flip_rate']:.1f}%", h["prompt"][:200]])
    pf(f"History saved to {hist_path}")

    # ===== CROSS-VALIDATE ON MBT =====
    pf(f"\n{'='*60}")
    pf(f"Cross-Validation on MedBenchThink")
    pf(f"{'='*60}")

    mbt = load_mbt_results(args.model)
    if mbt is not None:
        # Detect ID column
        for col in ['question_id', 'Index', 'index']:
            if col in mbt.columns: id_col_mbt = col; break
        else:
            id_col_mbt = mbt.index.name or 'index'

        pat_mbt = r'Answer:\s*([A-J])'
        if 'predicted' not in mbt.columns:
            mbt['predicted'] = mbt['response'].apply(
                lambda x: extract_letter(x, pat_mbt) if pd.notna(x) else None)
        if 'answer_key' in mbt.columns:
            mbt['key_letter'] = mbt['answer_key'].astype(str).str.extract(r'([A-J])')[0]
        else:
            with open(MBT_PATH) as f:
                mbt_data_key = json.load(f)
            mbt['key_letter'] = mbt['question_id'].astype(str).apply(
                lambda qid: re.search(r'([A-J])', str(next((d.get("answer", d.get("Answer", "")) for d in mbt_data_key if str(d.get("Index", "")) == str(qid)), ""))).group(1).upper() if re.search(r'([A-J])', str(next((d.get("answer", d.get("Answer", "")) for d in mbt_data_key if str(d.get("Index", "")) == str(qid)), ""))) else None)
            mbt['key_letter'] = mbt['key_letter'].fillna('')
        mbt['correct'] = mbt['predicted'] == mbt['key_letter']
        mbt_errors = mbt[~mbt['correct'] & mbt['predicted'].notna()]

        def get_mbt_text(qid):
            try:
                with open(MBT_PATH) as f:
                    data = json.load(f)
                for d in data:
                    if str(d.get("Index", "")) == str(qid):
                        return d.get("question", "")
            except: pass
            return ""

        mbt_errors.loc[:, "question_text"] = mbt_errors[id_col_mbt].apply(get_mbt_text)
        mbt_errors = mbt_errors[mbt_errors["question_text"] != ""]

        pf(f"  MBT errors to test: {len(mbt_errors)}")
        if len(mbt_errors) > 0:
            pf("  Testing best prompt on MBT errors (may take a while)...")
            # Test on a sample if too many
            test_sample = mbt_errors
            if args.max_test_errors and len(mbt_errors) > args.max_test_errors * 2:
                test_sample = mbt_errors.sample(n=min(args.max_test_errors * 2, len(mbt_errors)), random_state=42)
                pf(f"  Sampling {len(test_sample)} MBT errors for speed")
            mbt_flip, mbt_details = test_prompt_on_errors(
                args.model, best_prompt, test_sample, id_col_mbt, "medbenchthink")
            pf(f"  MBT cross-validation flip rate: {mbt_flip:.1f}%")
            with open(out_path, "a") as f:
                f.write(f"MBT cross-validation flip rate: {mbt_flip:.1f}%\n")
        else:
            pf("  No MBT errors found")
    else:
        pf("  No MBT results found for this model")

    pf(f"\n{'='*60}")
    pf(f"Done! Best prompt saved to {out_path}")
    pf(f"{'='*60}")


if __name__ == "__main__":
    main()
