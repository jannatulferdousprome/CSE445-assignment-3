# react_agent.py
"""
ReAct (Reason + Act) controller for the Autonomous Local LLM ML Agent.
CSE445 Assignment #3 - Task 1 (baseline loop) + Task 3 (self-healing).

Talks to a local Ollama server (http://127.0.0.1:11434) running a quantized
model such as llama3.2:3b, parses Thought/Action/Action Input blocks, executes
the matching Python tool from ml_tools.AVAILABLE_TOOLS, and feeds the result
back as an Observation until the model emits a Final Answer.

Task 3 self-healing:
    If a tool call raises an exception, the controller does NOT just hand the
    raw traceback back to the model. It first tries to classify the failure
    (shape mismatch / bad parameter / divergence-NaN) and apply a bounded,
    automatic parameter correction, retrying the SAME tool a limited number of
    times before giving up and asking the LLM to reason about it. Every
    self-heal attempt is logged so it shows up in the execution trace.
"""

import re
import json
import copy
import requests

from ml_tools import AVAILABLE_TOOLS

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

MAX_SELF_HEAL_RETRIES = 2  # per tool call, before surfacing the error to the LLM

SYSTEM_PROMPT = """You are an expert Autonomous Machine Learning Assistant.
You solve machine learning problems by thinking step-by-step and invoking external tools.

You have access to the following tools:
1. load_dataset_summary(dataset_name: str) -> JSON summary of dataset (iris, wine, breast_cancer).
2. train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> JSON test/CV scores
   (model_type: decision_tree, logistic_regression, random_forest).
3. train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01)
   -> JSON PyTorch training and evaluation results.
4. tune_hyperparameters(dataset_name: str, model_type: str, search_type: str = "grid", cv: int = 5,
   n_iter: int = 10) -> JSON best hyperparameters and best CV score (model_type: svc, decision_tree).
5. reduce_dimensionality(dataset_name: str, n_components: int = 2) -> JSON PCA explained variance.
6. select_features(dataset_name: str, n_features_to_select: int = 3, direction: str = "forward")
   -> JSON of selected features and resulting CV accuracy.
7. train_deep_pytorch_classifier(dataset_name: str, hidden_dims: list = [64, 32], dropout: float = 0.3,
   use_batchnorm: bool = True, epochs: int = 100, lr: float = 0.01, scheduler_type: str = "step")
   -> JSON training/evaluation results for a regularized deep MLP.

To use a tool, you MUST strictly use this format:
Thought: Describe your reasoning about what to do next.
Action: <tool_name>
Action Input: {"param_name": "value"}

When you have received the observation and are ready to provide the complete answer to the user,
format your output as:
Thought: I have gathered all necessary experimental data.
Final Answer: <your complete, well-reasoned answer, including a short comparison / recommendation>

Begin!
"""


def query_local_llm(prompt: str) -> str:
    """Queries the local Ollama instance running in WSL."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "stop": ["Observation:"],
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")
    return response.json().get("response", "")


# ---------------------------------------------------------------------------
# Task 3: Self-healing tool execution
# ---------------------------------------------------------------------------
def _classify_error(exc: Exception) -> str:
    """Classifies a caught exception into one of a few known failure modes."""
    msg = str(exc).lower()
    if isinstance(exc, FloatingPointError) or "nan" in msg or "diverged" in msg:
        return "nan_loss"
    if "shape" in msg or "mismatch" in msg or "broadcast" in msg:
        return "shape_mismatch"
    if isinstance(exc, (ValueError, TypeError)):
        return "invalid_parameter"
    return "unknown"


def _auto_correct(tool_name: str, kwargs: dict, error_type: str, attempt: int) -> dict:
    """
    Applies a bounded, heuristic correction to kwargs based on the classified
    error type. Returns a NEW kwargs dict; never mutates the caller's dict.
    """
    corrected = copy.deepcopy(kwargs)

    if error_type == "nan_loss":
        # Loss diverged -> halve the learning rate each retry.
        current_lr = float(corrected.get("lr", 0.01))
        corrected["lr"] = round(current_lr / 2, 6)

    elif error_type == "invalid_parameter":
        # Common culprits across our tools: test_size, epochs, hidden_dim(s), n_components.
        if "test_size" in corrected:
            ts = float(corrected["test_size"])
            corrected["test_size"] = min(max(ts, 0.1), 0.4)
        if "epochs" in corrected and int(corrected.get("epochs", 1)) <= 0:
            corrected["epochs"] = 50
        if "hidden_dim" in corrected and int(corrected.get("hidden_dim", 1)) <= 0:
            corrected["hidden_dim"] = 32
        if "hidden_dims" in corrected:
            hd = corrected["hidden_dims"]
            if isinstance(hd, str):
                try:
                    hd = json.loads(hd)
                except json.JSONDecodeError:
                    hd = [64, 32]
            if not hd or any((not isinstance(h, (int, float)) or h <= 0) for h in hd):
                hd = [64, 32]
            corrected["hidden_dims"] = hd
        if "n_components" in corrected and int(corrected.get("n_components", 1)) <= 0:
            corrected["n_components"] = 2

    elif error_type == "shape_mismatch":
        # Most shape issues in our tools trace back to an out-of-range test_size
        # or n_components/n_features_to_select exceeding the feature count.
        if "test_size" in corrected:
            corrected["test_size"] = 0.2
        if "n_components" in corrected:
            corrected["n_components"] = 2
        if "n_features_to_select" in corrected:
            corrected["n_features_to_select"] = 2

    return corrected


def safe_tool_execute(tool_name: str, kwargs: dict, log_fn=print):
    """
    Executes a tool with self-healing retries.
    Returns (observation_str, heal_log) where heal_log is a list of strings
    describing every self-heal attempt (empty if the first call succeeded).
    """
    heal_log = []
    current_kwargs = kwargs

    for attempt in range(MAX_SELF_HEAL_RETRIES + 1):
        try:
            result = AVAILABLE_TOOLS[tool_name](**current_kwargs)
            if heal_log:
                log_fn(f"[self-heal] Recovered after {attempt} retry(ies).")
            return f"Observation: {result}\n", heal_log
        except Exception as e:  # noqa: BLE001 - we deliberately catch broadly here
            error_type = _classify_error(e)
            log_fn(f"[self-heal] Attempt {attempt} failed ({error_type}): {e}")

            if attempt == MAX_SELF_HEAL_RETRIES:
                # Give up auto-correcting; surface the reasoned failure to the LLM.
                observation = (
                    f"Observation: Tool '{tool_name}' failed after "
                    f"{MAX_SELF_HEAL_RETRIES} self-healing attempt(s). "
                    f"Last error ({error_type}): {e}. "
                    f"Please choose different parameters or a different tool.\n"
                )
                return observation, heal_log

            corrected_kwargs = _auto_correct(tool_name, current_kwargs, error_type, attempt)
            heal_log.append(
                f"attempt={attempt} error_type={error_type} "
                f"old_params={current_kwargs} new_params={corrected_kwargs}"
            )
            current_kwargs = corrected_kwargs


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_action(llm_output: str):
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)", llm_output)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", llm_output, re.DOTALL)
    if not (action_match and input_match):
        return None, None
    return action_match.group(1).strip(), input_match.group(1).strip()


# ---------------------------------------------------------------------------
# Main ReAct loop
# ---------------------------------------------------------------------------
def run_agent_loop(user_query: str, max_iterations: int = 8):
    print("\n=======================================================")
    print(f"USER QUERY: {user_query}")
    print("=======================================================\n")

    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"
    full_trace = []

    for step in range(1, max_iterations + 1):
        print(f"\n--- Step {step} ---")
        llm_output = query_local_llm(prompt)
        print(llm_output)
        prompt += llm_output
        full_trace.append({"step": step, "llm_output": llm_output})

        if "Final Answer:" in llm_output:
            print("\n>>> Task Completed Successfully!")
            break

        tool_name, raw_input = _parse_action(llm_output)

        if tool_name is None:
            observation = "\nObservation: Please respond with an Action and Action Input, or a Final Answer.\n"
            prompt += observation
            full_trace[-1]["observation"] = observation
            continue

        try:
            kwargs = json.loads(raw_input)
        except json.JSONDecodeError:
            observation = "\nObservation: Error parsing Action Input as JSON. Re-check the format.\n"
            prompt += observation
            full_trace[-1]["observation"] = observation
            continue

        if tool_name not in AVAILABLE_TOOLS:
            observation = f"\nObservation: Tool '{tool_name}' not recognized.\n"
            prompt += observation
            full_trace[-1]["observation"] = observation
            continue

        observation, heal_log = safe_tool_execute(tool_name, kwargs)
        print(observation)
        prompt += f"\n{observation}\n"
        full_trace[-1]["observation"] = observation
        full_trace[-1]["self_heal_log"] = heal_log

    return full_trace


if __name__ == "__main__":
    test_task = (
        "Analyze the breast_cancer dataset, train a Random Forest and a regularized "
        "deep PyTorch classifier on it, compare their accuracies, and recommend the "
        "best model for clinical screening."
    )
    run_agent_loop(test_task)
