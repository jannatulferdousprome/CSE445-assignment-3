# benchmark_llm_latency.py
"""
Measures local LLM latency/throughput for the Technical Report's latency table.

Ollama's /api/generate response (even in non-streaming mode) includes precise
timing fields in nanoseconds:
    load_duration        - time to load the model into memory (first call only,
                            usually ~0 on subsequent calls since it stays loaded)
    prompt_eval_duration  - time to process the input prompt (prefill)
    eval_duration         - time to generate the output tokens (decode)
    eval_count            - number of tokens generated

From these we derive:
    time_to_first_token  ~= load_duration + prompt_eval_duration
    full_response_time   =  total_duration
    tokens_per_second    =  eval_count / (eval_duration / 1e9)

Run:
    python3 benchmark_llm_latency.py
"""

import time
import platform
import subprocess
import statistics
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODELS_TO_TEST = ["llama3.2:3b", "mistral:7b"]  # add "mistral:7b" here too if you want a comparison row
NUM_RUNS = 5

# A representative prompt: similar length/shape to a real mid-loop ReAct prompt
TEST_PROMPT = """You are an expert Autonomous Machine Learning Assistant.
Thought: I need to check the iris dataset before training a model.
Action: load_dataset_summary
Action Input: {"dataset_name": "iris"}
"""


def get_hardware_info():
    info = {}
    info["platform"] = platform.platform()
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in cpu.splitlines():
            if line.startswith("Model name:"):
                info["cpu"] = line.split(":", 1)[1].strip()
            if line.startswith("CPU(s):"):
                info["cpu_cores"] = line.split(":", 1)[1].strip()
    except FileNotFoundError:
        info["cpu"] = platform.processor()

    try:
        mem = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout
        for line in mem.splitlines():
            if line.startswith("Mem:"):
                info["ram"] = line.split()[1]
    except FileNotFoundError:
        pass

    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True)
        if gpu.returncode == 0 and gpu.stdout.strip():
            info["gpu"] = gpu.stdout.strip()
        else:
            info["gpu"] = "None detected (CPU-only run)"
    except FileNotFoundError:
        info["gpu"] = "None detected (CPU-only run)"

    return info


def benchmark_model(model_name: str, num_runs: int = NUM_RUNS):
    ttft_list, total_time_list, tokens_per_sec_list = [], [], []

    for i in range(num_runs):
        payload = {
            "model": model_name,
            "prompt": TEST_PROMPT,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        wall_start = time.time()
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        wall_elapsed = time.time() - wall_start

        if resp.status_code != 200:
            print(f"  [run {i+1}] ERROR: {resp.text}")
            continue

        data = resp.json()
        load_ns = data.get("load_duration", 0)
        prompt_eval_ns = data.get("prompt_eval_duration", 0)
        eval_ns = data.get("eval_duration", 1)  # avoid div-by-zero
        eval_count = data.get("eval_count", 0)

        ttft_s = (load_ns + prompt_eval_ns) / 1e9
        tokens_per_sec = eval_count / (eval_ns / 1e9) if eval_ns else 0.0

        ttft_list.append(ttft_s)
        total_time_list.append(wall_elapsed)
        tokens_per_sec_list.append(tokens_per_sec)

        print(f"  [run {i+1}/{num_runs}] TTFT={ttft_s:.2f}s  "
              f"total={wall_elapsed:.2f}s  tok/s={tokens_per_sec:.1f}")

    if not ttft_list:
        return None

    return {
        "model": model_name,
        "avg_ttft": round(statistics.mean(ttft_list), 2),
        "avg_total_time": round(statistics.mean(total_time_list), 2),
        "avg_tokens_per_sec": round(statistics.mean(tokens_per_sec_list), 1),
    }


def main():
    print("Collecting hardware info...")
    hw = get_hardware_info()
    hw_str = f"{hw.get('cpu', 'Unknown CPU')}, {hw.get('ram', '?')} RAM, {hw.get('gpu', 'Unknown GPU')}"
    print(f"Hardware: {hw_str}\n")

    results = []
    for model in MODELS_TO_TEST:
        print(f"Benchmarking {model} ({NUM_RUNS} runs)...")
        r = benchmark_model(model)
        if r:
            r["hardware"] = hw_str
            results.append(r)
        print()

    print("=" * 70)
    print("MARKDOWN TABLE (copy this into report.md Section 3)")
    print("=" * 70)
    print("| Model | Avg. Time-to-First-Token (s) | Avg. Full Response Time (s) "
          "| Avg. Tokens/sec | Hardware |")
    print("|:-|:-|:-|:-|:-|")
    for r in results:
        print(f"| {r['model']} | {r['avg_ttft']} | {r['avg_total_time']} | "
              f"{r['avg_tokens_per_sec']} | {r['hardware']} |")


if __name__ == "__main__":
    main()