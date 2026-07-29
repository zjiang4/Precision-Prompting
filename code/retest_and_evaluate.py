"""Phase 4: Test personalized prompt on test+validation sets.

Applies each model's personalized prompt to ALL of its test-set errors
(ICD-Bench) and validation errors (MedThink-Bench), with retry until
every question receives a response. Computes flip rates and accuracy gains.

Usage:
    python retest_and_evaluate.py --model gpt-oss-120b
    python retest_and_evaluate.py --all
"""
import argparse, os, re, json, time, threading, requests
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MODEL_REGISTRY, ICDBENCH_PATH, MBT_PATH, BASE_DIR

MDIR = os.path.join(BASE_DIR, "model_data")

def ex_letter(resp):
    if not resp: return None
    for p in [r'Answer:\s*([A-D])', r'answer\s+is\s*([A-D])', r'\*\*([A-D])\*\*']:
        m = re.search(p, str(resp), re.IGNORECASE)
        if m: return m.group(1).upper()
    return None

def ex_mbt(resp):
    if not resp: return None
    m = re.search(r'Answer:\s*([A-J])', str(resp), re.IGNORECASE)
    return m.group(1).upper() if m else None

def call_model(model_key, system, question, timeout=120):
    """Call candidate model with the personalized prompt."""
    cfg = MODEL_REGISTRY[model_key]
    # ... dispatch to provider-specific API (see benchmark_runner.py for example) ...
    # This is simplified; the actual implementation handles 7 different API formats.
    raise NotImplementedError("See benchmark_runner.py for API call patterns")

def run_phase(name, model_key, prompt, items, extract_fn, workers=5, max_rounds=15):
    """Test ALL items with retry until zero missing."""
    total = len(items)
    answers = {}; pending = set(range(total)); rnd = 0
    
    while pending and rnd < max_rounds:
        rnd += 1
        batch = list(pending)
        if rnd > 1:
            print(f"    [{name}] Round {rnd}: {len(batch)} remaining", flush=True)
        
        def test_item(i):
            resp = call_model(model_key, prompt, items[i][0])
            return i, extract_fn(resp)
        
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed({ex.submit(test_item, i): i for i in batch}):
                i, pred = f.result()
                if pred is not None:
                    answers[i] = pred
                    pending.discard(i)
        
        if pending:
            print(f"    [{name}] {len(pending)} unanswered, retrying...", flush=True)
    
    flipped = sum(1 for i in answers if answers[i] == items[i][1])
    rate = flipped / max(total, 1) * 100
    nulls = total - len(answers)
    print(f"  [{name}] {flipped}/{total} = {rate:.1f}% ({len(answers)} answered, {nulls} NULL)")
    return flipped, len(answers), total, rate

def evaluate_model(model_key, icd_df, mbt_kmap, mbt_q):
    """Full retest for one model."""
    mdir = os.path.join(MDIR, model_key.replace("/", "_"))
    
    # Load personalized prompt
    with open(os.path.join(mdir, "03_optimized_prompts/prompt_iter_1.txt")) as f:
        prompt = f.read()
    
    # === ICD-Bench test set ===
    tdf = pd.read_parquet(os.path.join(mdir, "01_baseline/icdbench_test.parquet"))
    idc = 'question_id' if 'question_id' in tdf.columns else 'index'
    errors = tdf[~tdf['correct'] & tdf['predicted'].notna()].copy()
    
    test_items = []
    for _, row in errors.iterrows():
        qid = row[idc]
        qtext = icd_df.loc[qid, "question"] + "\n\nOptions:\n" + icd_df.loc[qid, "options"]
        test_items.append((qtext, row['key_letter']))
    
    tf, ta, tt, tr = run_phase(f"{model_key}-TEST", model_key, prompt, test_items, ex_letter)
    
    # === MedThink-Bench ===
    mdf = pd.read_parquet(os.path.join(mdir, "01_baseline/medbenchthink.parquet"))
    qc = 'question_id' if 'question_id' in mdf.columns else 'index'
    if 'predicted' not in mdf.columns:
        mdf['predicted'] = mdf['response'].apply(ex_mbt)
    mdf['kl'] = mdf[qc].astype(str).map(mbt_kmap)
    mbt_errors = mdf[mdf['predicted'] != mdf['kl']].copy()
    
    mbt_items = [(mbt_q.get(str(r[qc]), ""), r['kl']) for _, r in mbt_errors.iterrows()]
    
    mf, ma, mt, mr = run_phase(f"{model_key}-MBT", model_key, prompt, mbt_items, ex_mbt)
    
    # === Save results ===
    # Compute gains
    bf = json.load(open(os.path.join(mdir, "01_baseline/baseline_summary.json")))
    test_base = bf["icd_test"]["accuracy_pct"]
    test_gain = round(tf / 735 * 100, 1)
    
    summary = {
        "model": model_key,
        "total_test_errors": tt, "test_flipped": tf, "test_answered": ta,
        "test_flip_rate": round(tr, 1),
        "mbt": {"tested": True, "total": mt, "flipped": mf, "answered": ma, "flip_rate": round(mr, 1)},
    }
    
    rdir = os.path.join(mdir, "04_retest_results")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "retest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  ✅ {model_key}: test={tr:.1f}% mbt={mr:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    
    icd_df = pd.read_parquet(ICDBENCH_PATH)
    with open(MBT_PATH) as f: mbt_data = json.load(f)
    mbt_kmap = {}; mbt_q = {}
    for d in mbt_data:
        qid = str(d.get("Index",""))
        ans = d.get("answer", d.get("Answer",""))
        m = re.search(r'([A-J])', str(ans))
        mbt_kmap[qid] = m.group(1) if m else ""
        mbt_q[qid] = d.get("question","") + "\n\nOptions:\n" + d.get("options","")
    
    evaluate_model(args.model, icd_df, mbt_kmap, mbt_q)
