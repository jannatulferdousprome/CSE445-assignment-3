# CSE445 Assignment #3 — Autonomous Local LLM ML Agent

GitHub link: https://github.com/jannatulferdousprome/CSE445-assignment-3

## Student Info

- Student name: Jannatul Ferdous Prome

- Student ID: 2021770042

## About

An autonomous ReAct agent that runs entirely locally on Windows WSL2, using a
quantized LLM served by Ollama to reason over and orchestrate classical ML
(Scikit-Learn) and deep learning (PyTorch) tools.

## Files

| File | Purpose |
|:-|:-|
| `ml_tools.py` | All ML tools the agent can call (Task 1 + Task 2). |
| `react_agent.py` | ReAct controller: talks to Ollama, parses Thought/Action/Observation, self-heals tool failures (Task 3). |
| `benchmark_runner.py` | Deterministic multi-algorithm × multi-dataset benchmark, writes `benchmark_report.md` (Task 3). |
| `requirements.txt` | Python dependencies. |

## 1. Environment Setup (Windows WSL2)

```powershell
# In Windows PowerShell (Run as Administrator)
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

```bash
# In WSL Ubuntu Bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv curl build-essential git

mkdir -p ~/cse445_agent && cd ~/cse445_agent
python3 -m venv venv
source venv/bin/activate
```

## 2. Install and Serve the Local LLM (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b mistral:7b
```

Verify it's up:

```bash
curl http://127.0.0.1:11434/api/tags
```

## 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

(If you need a CUDA build of PyTorch for GPU passthrough, follow the
[official selector](https://pytorch.org/get-started/locally/) instead of the
generic `torch` pin in `requirements.txt`.)

## 4. Copy Project Files

Copy `ml_tools.py`, `react_agent.py`, and `benchmark_runner.py` into
`~/cse445_agent/`.

## 5. Run the Agent

```bash
python3 react_agent.py
```

This runs the default demo task defined in `if __name__ == "__main__"`. To
run your own natural-language task, edit `test_task` at the bottom of
`react_agent.py`, or import and call it from a Python shell:

```python
from react_agent import run_agent_loop
trace = run_agent_loop(
    "Evaluate decision_tree, random_forest, and a deep PyTorch classifier on "
    "both the wine and breast_cancer datasets using 5-fold cross-validation, "
    "then write a Markdown table comparing all results and recommend the "
    "best model for each dataset."
)
```

**Capture at least 3 such multi-step traces** (redirect stdout to a file) for
your submission's execution logs, e.g.:

```bash
python3 react_agent.py > execution_log_1.txt 2>&1
```

## 6. Run the Comprehensive Benchmark (Task 3)

```bash
python3 benchmark_runner.py
```

This produces `benchmark_report.md` with a Markdown comparison table across
3 algorithms × 2 datasets with 5-fold CV, plus a per-dataset "best model"
recommendation.

## Self-Healing Behavior (Task 3)

`react_agent.safe_tool_execute` wraps every tool call. If a tool raises an
exception, the controller:

1. Classifies the failure — `shape_mismatch`, `invalid_parameter`, or
   `nan_loss` (divergent training).
2. Applies a bounded, heuristic parameter correction (e.g., halving the
   learning rate on NaN loss, clamping `test_size` back into a valid range).
3. Retries the same tool call (up to `MAX_SELF_HEAL_RETRIES`, default 2).
4. If still failing, surfaces a clear diagnostic `Observation` to the LLM so
   it can reason about an alternative approach, instead of crashing the loop.

You can see this in action directly:

```python
from react_agent import safe_tool_execute
obs, heal_log = safe_tool_execute(
    "train_sklearn_model",
    {"dataset_name": "iris", "model_type": "decision_tree", "test_size": 0.9},
)
print(obs)
print(heal_log)
```

## Notes on Model Choice

`llama3.2:3b` is used by default for speed on modest hardware. Swap
`MODEL_NAME` in `react_agent.py` for `mistral:7b` or another instruction-tuned
model pulled via `ollama pull <model>` if you have more VRAM/RAM available —
larger models tend to follow the strict `Thought/Action/Action Input` format
more reliably.
