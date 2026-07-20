"""Diagnose model-specific reasoning errors on the ICD-Bench training set.

For each model:
1. Load baseline results (neutral prompt) on ICD-Bench
2. Extract training set (80%) errors
3. Call MedSeekV25 ("GPT5.5") analyst to classify each error into taxonomy
4. Generate a summary of the model's characteristic error profile
5. Save error set + profile for downstream prompt optimization

Usage:
    python analysis/diagnose_errors.py --model V3
    python analysis/diagnose_errors.py --model mistral-small-4-119b-2603
    python analysis/diagnose_errors.py --model V3 --dry-run   # skip analyst calls
"""
import argparse, os, sys, re, json, time
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from benchmark_scripts.benchmark_config import (
    BASE_DIR, RESULTS_DIR, ICDBENCH_PATH, CLEANED_DIR, MODEL_REGISTRY,
)

# ===== ANALYST LLM (GPT5.5 = MedSeekV25) =====
ANALYST_URL = "https://chat.pku.edu.cn/v1/chat/completions"
ANALYST_KEY = "JiangZheHan_nOeBhmuSIa5s"
ANALYST_MODEL = "MedSeekV25"

ERROR_TAXONOMY = {
    "E1": "Key Clue Neglect — overlooks a subtle but decisive lab finding or symptom",
    "E2": "Common Disease Bias — defaults to prevalent diagnosis despite atypical features",
    "E3": "Mechanism Confusion — selects correct disease but wrong pathophysiological mechanism",
    "E4": "Anchoring — fixates on one salient feature, discarding contradictory evidence",
    "E5": "Atypical Feature Discounting — dismisses features that don't fit the leading hypothesis",
    "E6": "Temporal Neglect — ignores onset age, progression speed, or treatment response timing",
    "E7": "Question-Type Mismatch — answers a different question than asked (e.g., 'most likely' vs 'least likely')",
}

ANALYST_SYSTEM = f"""You are an expert medical educator evaluating LLM reasoning on diagnosis questions.
Classify each error into one or more of these categories:

{chr(10).join(f'{k}: {v}' for k, v in ERROR_TAXONOMY.items())}

Output JSON array: [{{"id": <qid>, "categories": ["E1", "E4"], "explanation": "..."}}]"""


def call_analyst(messages, temperature=0.3):
    """Call MedSeekV25 (GPT5.5) streaming."""
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
            resp = requests.post(ANALYST_URL, headers=headers, json=data, stream=True, timeout=120)
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
        except Exception as e:
            print(f"  [Analyst call failed, attempt {attempt+1}]: {e}")
            time.sleep(3)
    return None


def extract_letter(resp):
    if not resp or pd.isna(resp):
        return None
    for pat in [r'Answer:\s*([A-D])', r'answer\s+is\s+([A-D])',
                r'correct answer is ([A-D])', r'choice\s+is\s+([A-D])']:
        m = re.search(pat, str(resp), re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def load_results(model_key):
    known = {
        "V3": os.path.join(BASE_DIR, "results_v3.parquet"),
        "V25": os.path.join(BASE_DIR, "results_v25.parquet"),
        "Baichuan-M3": os.path.join(BASE_DIR, "results_baichuan.parquet"),
    }
    if model_key in known and os.path.exists(known[model_key]):
        return pd.read_parquet(known[model_key])

    # Check if model_key is a registry key
    registry = MODEL_REGISTRY.get(model_key)
    if registry:
        model_short = registry["model_name"].split("/")[-1]
    else:
        model_short = model_key

    nv_path = os.path.join(RESULTS_DIR, f"results_{model_short}_icdbench.parquet")
    if os.path.exists(nv_path):
        return pd.read_parquet(nv_path)

    print(f"  No results found at {nv_path}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key or short name")
    parser.add_argument("--train-idx", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip analyst calls")
    args = parser.parse_args()

    # Load split
    train_idx_path = args.train_idx or os.path.join(CLEANED_DIR, "analysis/train_idx.npy")
    if not os.path.exists(train_idx_path):
        print("No split found. Run split_icdbench.py first.")
        return
    train_idx = set(np.load(train_idx_path))

    # Load benchmark data
    icd_df = pd.read_parquet(ICDBENCH_PATH)

    # Load model results
    results = load_results(args.model)
    if results is None:
        return

    id_col = 'question_id' if 'question_id' in results.columns else 'index'
    if 'predicted' not in results.columns:
        results['predicted'] = results['response'].apply(extract_letter)
    results['key_letter'] = results['answer_key'].astype(str).str.extract(r'([A-D])')[0]
    results['correct'] = results['predicted'] == results['key_letter']

    # Filter to training set
    train_results = results[results[id_col].isin(train_idx)]
    errors = train_results[~train_results['correct'] & train_results['predicted'].notna()]

    print(f"\n{'='*60}")
    print(f"Error Diagnosis for {args.model}")
    print(f"{'='*60}")
    print(f"  Training set: {len(train_idx)} questions")
    valid_train = train_results[train_results['predicted'].notna()]
    print(f"  Training errors: {len(errors)}/{len(valid_train)} "
          f"({100*len(errors)/max(len(valid_train),1):.1f}%)")

    # Enrich with question text
    icd_df["question_text"] = icd_df.apply(
        lambda r: f"Question: {r['question']}\n\nOptions:\n{r['options']}", axis=1)
    errors_enriched = errors.merge(
        icd_df[["question_text", "key_letter"]], left_on=id_col, right_index=True, how="left")

    # Classify errors with analyst
    classifications = []
    error_batches = [errors_enriched.iloc[i:i+10] for i in range(0, len(errors_enriched), 10)]

    for batch_idx, batch in enumerate(error_batches):
        if args.dry_run:
            print(f"  Batch {batch_idx+1}/{len(error_batches)}: {len(batch)} errors (skipped, dry-run)")
            continue

        print(f"  Classifying batch {batch_idx+1}/{len(error_batches)} ({len(batch)} errors)...")
        batch_data = []
        for _, row in batch.iterrows():
            batch_data.append({
                "id": int(row[id_col]),
                "model_answer": row["predicted"],
                "correct_answer": row["key_letter"],
                "question": row["question_text"][:500],
            })

        analyst_msg = {
            "role": "user",
            "content": json.dumps({
                "task": f"Classify errors made by {args.model} on medical diagnosis questions",
                "errors": batch_data,
                "taxonomy": ERROR_TAXONOMY,
                "output_format": "JSON array with id, categories (list), explanation",
            }, indent=2),
        }

        response = call_analyst([
            {"role": "system", "content": ANALYST_SYSTEM},
            analyst_msg,
        ])

        if response:
            # Try to parse JSON
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    classifications.extend(parsed)
                    print(f"    Got {len(parsed)} classifications")
                except json.JSONDecodeError:
                    print(f"    Failed to parse JSON from response")
            else:
                print(f"    No JSON found in response")
        time.sleep(1)

    # Build error profile
    error_profile = {
        "model": args.model,
        "total_train": len(valid_train),
        "total_errors": len(errors),
        "error_rate": round(100 * len(errors) / max(len(valid_train), 1), 1),
        "classifications": classifications,
    }

    # Save error set and profile
    out_dir = os.path.join(CLEANED_DIR, f"analysis/errors_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    errors_enriched.to_parquet(os.path.join(out_dir, "train_errors.parquet"))

    with open(os.path.join(out_dir, "error_profile.json"), "w") as f:
        json.dump(error_profile, f, indent=2, ensure_ascii=False)

    # Print summary
    cat_counts = {}
    for c in classifications:
        for cat in c.get("categories", []):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    if cat_counts:
        print(f"\n  Error category distribution:")
        for cat_id in sorted(ERROR_TAXONOMY.keys()):
            count = cat_counts.get(cat_id, 0)
            pct = 100 * count / max(len(classifications), 1)
            bar = "█" * int(pct / 2)
            print(f"    {cat_id}: {count:>3} ({pct:>4.1f}%) {bar}")

    print(f"\n  Saved to {out_dir}/")
    if not args.dry_run and not classifications:
        print("  ⚠  No classifications — check analyst API key / connectivity")


if __name__ == "__main__":
    main()
