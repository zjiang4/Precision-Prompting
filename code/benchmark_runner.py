"""Example: Run a candidate LLM on ICD-Bench questions.

This is a simplified example showing how each model was evaluated
with the neutral baseline prompt. The full pipeline evaluates all
14 models across ICD-Bench (3,675 questions) and MedThink-Bench (500 questions).

Usage:
    python benchmark_runner.py --model gpt-oss-120b --benchmark icdbench
"""
import argparse, os, json, re, time, requests, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import MODEL_REGISTRY, NEUTRAL_PROMPT, NVIDIA_API_KEYS, ICDBENCH_PATH

def extract_letter(resp):
    """Extract A-D from model response."""
    if not resp: return None
    for pat in [r'Answer:\s*([A-D])', r'answer\s+is\s*([A-D])', r'\*\*([A-D])\*\*']:
        m = re.search(pat, str(resp), re.IGNORECASE)
        if m: return m.group(1).upper()
    return None

def call_model(model_key, system_prompt, question, timeout=120):
    """Call a candidate model via its API. Returns response text."""
    cfg = MODEL_REGISTRY[model_key]
    provider = cfg["provider"]
    model_name = cfg["model"]
    
    # Example: NVIDIA API (adapt for other providers)
    if provider == "nvidia":
        key = NVIDIA_API_KEYS[0]
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question + "\n\nEnd with: Answer: X"},
                ],
                "max_tokens": 12000, "temperature": 0, "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    
    # Add similar blocks for "dashscope", "baichuan", "pku", "lmstudio", etc.
    # See config.py for endpoint URLs and key names.
    raise NotImplementedError(f"Provider {provider} not implemented in this example")

def run_benchmark(model_key, benchmark="icdbench", max_questions=None):
    """Run a model on benchmark questions with the neutral prompt."""
    df = pd.read_parquet(ICDBENCH_PATH)
    if max_questions:
        df = df.head(max_questions)
    
    df["question_text"] = df.apply(
        lambda r: f"{r['question']}\n\nOptions:\n{r['options']}", axis=1)
    df["answer_letter"] = df["answer"].astype(str).str.extract(r'([A-D])')[0]
    
    results = []
    questions = list(df["question_text"].values)
    correct_answers = list(df["answer_letter"].values)
    
    for i, (q, ans) in enumerate(zip(questions, correct_answers)):
        resp = call_model(model_key, NEUTRAL_PROMPT, q)
        pred = extract_letter(resp)
        results.append({
            "question_id": i,
            "model_answer": pred,
            "correct_answer": ans,
            "is_correct": pred == ans if pred else False,
        })
        if (i + 1) % 100 == 0:
            acc = sum(r["is_correct"] for r in results) / len(results) * 100
            print(f"  {model_key}: {i+1}/{len(questions)} accuracy={acc:.1f}%", flush=True)
    
    accuracy = sum(r["is_correct"] for r in results) / len(results) * 100
    print(f"\n{model_key} baseline accuracy: {accuracy:.1f}% ({sum(r['is_correct'] for r in results)}/{len(results)})")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="icdbench")
    parser.add_argument("--max", type=int, default=10, help="Max questions (demo)")
    args = parser.parse_args()
    run_benchmark(args.model, args.benchmark, args.max)
