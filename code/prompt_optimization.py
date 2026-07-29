"""Phase 3: Generate personalized prompt for each model using GPT-5.4.

Reads each model's error_profile.json (from Phase 2), sends it to GPT-5.4
along with reference answers, and generates a structured system prompt with
countermeasure rules targeting the model's dominant error categories.

Usage:
    python prompt_optimization.py --model gpt-oss-120b
    python prompt_optimization.py --all
"""
import argparse, os, json, re, time, threading, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import ANALYST_URL, ANALYST_MODEL, ERROR_TAXONOMY, BASE_DIR, PKU_API_KEYS

MDIR = os.path.join(BASE_DIR, "model_data")

PROMPT_GEN_SYSTEM = (
    "You are an expert in prompt engineering for medical AI. "
    "Create a personalized system prompt that corrects a specific LLM's "
    "characteristic diagnostic reasoning errors.\n\n"
    "The prompt must:\n"
    "1. Be a structured, step-by-step diagnostic protocol\n"
    "2. Include specific 'Trap' sections targeting the model's most common error types\n"
    "3. Each Trap must include a concrete clinical example\n"
    "4. Be 400-800 words\n"
    "5. End with: 'Answer: X' where X is your final answer letter\n\n"
    "Output ONLY the prompt text."
)

_key_idx = [0]; _key_lock = threading.Lock()
def next_key():
    with _key_lock:
        k = PKU_API_KEYS[_key_idx[0] % len(PK_API_KEYS)]
        _key_idx[0] += 1
        return k

def call_gpt54(messages, max_tokens=4000, temperature=0.4, timeout=120):
    """Call GPT-5.4 streaming."""
    key = next_key()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    data = {
        "model": ANALYST_MODEL,
        "messages": messages,
        "stream": True,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
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

def generate_prompt(model_key, references):
    """Generate one personalized prompt for a model."""
    mdir = os.path.join(MDIR, model_key.replace("/", "_"))
    profile_path = os.path.join(mdir, "02_error_classification/error_profile.json")
    
    with open(profile_path) as f:
        profile = json.load(f)
    
    pcts = profile["category_percentages"]
    sorted_cats = sorted(pcts.items(), key=lambda x: -x[1])
    
    # Get top example errors for each dominant category
    from collections import defaultdict
    cat_examples = defaultdict(list)
    for c in profile.get("classifications", [])[:30]:
        cat = c.get("category", "")
        if cat and len(cat_examples[cat]) < 3:
            qid = c["id"]
            ref = references.get(str(qid), {})
            cat_examples[cat].append({
                "question_id": qid, "category": cat,
                "explanation": c["explanation"][:200],
                "correct_answer": ref.get("correct_answer", "?"),
                "reference_reasoning": (ref.get("reference_reasoning") or "")[:300],
            })
    
    error_summary = {
        "model": model_key,
        "total_errors": profile["total_classified"],
        "top_error_types": [
            {"category": cat, "percentage": pct, "examples": cat_examples.get(cat, [])}
            for cat, pct in sorted_cats[:3]
        ],
        "all_percentages": pcts,
    }
    
    user_msg = json.dumps({
        "task": f"Create a personalized diagnostic reasoning prompt for {model_key}.",
        "error_profile": error_summary,
        "format": "Structured system prompt with numbered steps and Trap sections",
    }, indent=2, ensure_ascii=False)
    
    messages = [
        {"role": "system", "content": PROMPT_GEN_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    
    prompt_text = call_gpt54(messages)
    
    prompt_dir = os.path.join(mdir, "03_optimized_prompts")
    os.makedirs(prompt_dir, exist_ok=True)
    prompt_path = os.path.join(prompt_dir, "prompt_iter_1.txt")
    with open(prompt_path, "w") as f:
        f.write(prompt_text)
    
    print(f"  {model_key}: prompt saved ({len(prompt_text)} chars)")
    return prompt_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Single model key")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    # Load reference answers (from Phase 0)
    ref_path = os.path.join(MDIR, "reference_answers.json")
    references = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            references = json.load(f)
    
    ALL_MODELS = list(os.listdir(MDIR))
    ALL_MODELS = [m for m in ALL_MODELS if os.path.isdir(os.path.join(MDIR, m)) and m != "reference_answers.json"]
    
    models = [args.model] if args.model else ALL_MODELS
    
    for model in models:
        try:
            generate_prompt(model, references)
        except Exception as e:
            print(f"  {model}: FAILED - {e}")

if __name__ == "__main__":
    main()
