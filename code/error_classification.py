"""Phase 2: Classify training-set errors using GPT-5.4 analyst.

For each model, loads its ICD-Bench training-set errors, sends them
in batches to GPT-5.4, and classifies each error into exactly one of
seven categories (E1-E7). Saves per-model error_profile.json.

Usage:
    python error_classification.py --model gpt-oss-120b
    python error_classification.py --model gpt-oss-120b --resume  # skip already classified
"""
import argparse, os, json, re, time, threading, requests
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (
    ANALYST_URL, ANALYST_MODEL, ERROR_TAXONOMY,
    PKU_API_KEYS, ICDBENCH_PATH, BASE_DIR,
)

OUT_DIR = os.path.join(BASE_DIR, "model_data")

# Analyst system prompt — "exactly one category" instruction
ANALYST_SYSTEM = (
    "You are an expert medical educator and cognitive bias specialist. "
    "Your task is to analyze diagnostic reasoning errors made by large language models "
    "and classify each error into EXACTLY ONE most appropriate category.\n\n"
    "## Error Taxonomy\n\n"
    + "\n\n".join(f"**{k}** — {v}" for k, v in ERROR_TAXONOMY.items())
    + "\n\n## Guidelines\n\n"
    "1. Assign EXACTLY ONE category per error — the SINGLE most diagnostic failure mode.\n"
    "2. E1 should ONLY be used when no other cognitive bias pattern applies.\n"
    "3. If the model fixated on one feature, use E4. If it dismissed atypical features, use E5.\n"
    "4. If the model got the right disease but wrong mechanism, use E3.\n\n"
    'Output: [{"id": <qid>, "category": "E4", "explanation": "..."}]'
)

_key_idx = [0]; _key_lock = threading.Lock()
def next_key():
    with _key_lock:
        k = PKU_API_KEYS[_key_idx[0] % len(PK_API_KEYS)]
        _key_idx[0] += 1
        return k

def call_analyst(messages, timeout=120):
    """Call GPT-5.4 with streaming."""
    key = next_key()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    data = {
        "model": ANALYST_MODEL,
        "messages": messages,
        "stream": True,
        "max_completion_tokens": 4000,
        "temperature": 0.3,
    }
    resp = requests.post(ANALYST_URL, headers=headers, json=data, stream=True, timeout=timeout)
    resp.raise_for_status()
    content = ""
    for line in resp.iter_lines():
        if not line: continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "): continue
        payload = decoded[6:].strip()
        if payload == "[DONE]": break
        try:
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("delta", {})
            c = delta.get("content", "")
            if c: content += c
        except: pass
    if "</think>" in content:
        parts = content.split("</think>", 1)
        content = parts[1] if len(parts) > 1 else parts[0]
    return content.strip()

def classify_batch(batch_data, model_name):
    """Classify a batch of ~10 errors. Returns list of classification dicts."""
    user_msg = json.dumps({
        "task": f"Classify errors made by {model_name} on medical diagnosis questions.",
        "errors": batch_data,
        "taxonomy": {k: v.split("—")[0].strip() for k, v in ERROR_TAXONOMY.items()},
        "output_format": "JSON array with id, category (single E1-E7), explanation",
    }, indent=2, ensure_ascii=False)
    
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    response = call_analyst(messages)
    if not response: return []
    
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return []
    return []

def classify_model(model_key, train_idx, icd_df, batch_size=10, workers=3, resume=False):
    """Classify all training errors for one model."""
    print(f"\nClassifying errors for: {model_key}")
    
    # Load model results, identify train-set errors
    results_path = os.path.join(OUT_DIR, model_key.replace("/","_"), "01_baseline/icdbench_train.parquet")
    if not os.path.exists(results_path):
        print(f"  No baseline results at {results_path}")
        return None
    df = pd.read_parquet(results_path)
    id_col = 'question_id' if 'question_id' in df.columns else 'index'
    errors = df[~df['correct'] & df['predicted'].notna()].copy()
    
    # Enrich with question text
    icd_df["question_text"] = icd_df.apply(
        lambda r: f"Question: {r['question']}\nOptions:\n{r['options']}", axis=1)
    errors = errors.merge(icd_df[["question_text"]], left_on=id_col, right_index=True, how="left")
    
    n_errors = len(errors)
    print(f"  Training errors: {n_errors}")
    
    # Batch and classify
    batches = [errors.iloc[i:i+batch_size] for i in range(0, len(errors), batch_size)]
    classifications = []
    
    def process_batch(batch):
        batch_data = []
        for _, row in batch.iterrows():
            batch_data.append({
                "id": int(row[id_col]),
                "model_answer": row.get("predicted", "?"),
                "correct_answer": row.get("key_letter", "?"),
                "question": str(row.get("question_text", ""))[:600],
            })
        return classify_batch(batch_data, model_key)
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_batch, b): i for i, b in enumerate(batches)}
        done = 0
        for f in as_completed(futs):
            classifications.extend(f.result())
            done += 1
            if done % 5 == 0:
                print(f"  Progress: {done}/{len(batches)} batches", flush=True)
    
    # Build profile
    from collections import Counter
    cat_counts = Counter()
    for c in classifications:
        cat = c.get("category", c.get("categories", [""])[0] if c.get("categories") else "")
        if cat: cat_counts[cat] += 1
    
    total = len(classifications)
    profile = {
        "model": model_key, "analyst": "gpt-5.4",
        "total_errors": n_errors, "total_classified": total,
        "category_counts": dict(sorted(cat_counts.items())),
        "category_percentages": {
            k: round(100 * cat_counts.get(k, 0) / max(total, 1), 1)
            for k in ERROR_TAXONOMY
        },
        "classifications": classifications,
    }
    
    out_path = os.path.join(OUT_DIR, model_key.replace("/","_"), "02_error_classification/error_profile.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    
    print(f"  Category distribution:")
    for cat in ERROR_TAXONOMY:
        count = cat_counts.get(cat, 0)
        pct = 100 * count / max(total, 1)
        print(f"    {cat}: {count:>4} ({pct:>5.1f}%)")
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    
    icd_df = pd.read_parquet(ICDBENCH_PATH)
    train_idx = set(np.load(os.path.join(BASE_DIR, "cleaned/analysis/train_idx.npy")))
    classify_model(args.model, train_idx, icd_df)
