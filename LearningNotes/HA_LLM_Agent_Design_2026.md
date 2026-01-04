# Hybrid Automaton LLM Agent Design (2026)

**Version:** 2.0  
**Date:** January 4, 2026  
**Status:** Design Draft  
**Target Framework:** Hugging Face `smolagents`  

---

## 1. Executive Summary

This document outlines the design for an autonomous **HA-LLM Agent** capable of inferring Hybrid Automata (HA) models from time-series data (`.npz`) and trajectory plots. 

**The Core Problem:** The current `DAINARX` pipeline relies on manual "priors" provided in a JSON configuration (`order`, `need_reset`, `self_loop`, `kernel`, `other_items`). This limits generalization.

**The Solution:** We propose a **closed-loop agentic workflow** inspired by "SR-Scientist" (Symbolic Regression Scientist). The agent uses an LLM to propose configuration hypotheses, executes the existing identification tools (`main.py` logic), simulates the result, and refines its hypothesis based on feedback metrics (RMSE, Event Match Rate).

---

## 2. Methodology & Architecture

We adopt a **Manager-Worker** architecture using the `smolagents` framework.

### 2.1 The Cycle of Discovery (SR-Scientist Approach)

1.  **Observation:** The agent analyzes the input data (statistics, derivatives) and visual plots (visual patterns).
2.  **Hypothesis Generation:** The LLM proposes a `config` (e.g., "This looks like a Bouncing Ball, so `need_reset=True`, `order=1`").
3.  **Experiment (Tool Execution):** The agent runs the `DAINARX` pipeline with this config.
4.  **Verification:** The resulting HA is simulated and compared against the ground truth data.
5.  **Refinement:** If the error is high, the agent analyzes *why* (e.g., "Missed the bounce" -> Increase `svm_c` or change `kernel`; "Poor curve fit" -> Add `x[?]**3` to `other_items`) and retries.

### 2.2 Dual-Path Strategy

The agent dynamically adapts its strategy based on the data signature:

| Feature | **Case A: Visual Switching** (e.g., Bouncing Ball) | **Case B: Latent/Subtle Switching** (e.g., Duffing) |
| :--- | :--- | :--- |
| **Visual Cues** | Sharp discontinuities, velocity reversals. | Smooth trajectories, oscillations. |
| **Key Configs** | `need_reset=True`, `self_loop=False` (usually). | `need_reset=False`, `self_loop=True` (often). |
| **Focus** | Accurate *ChangePoint* detection and *Guard* learning. | Accurate *Feature Extraction* (`other_items`) to capture non-linearity. |
| **Refinement** | Adjust `window_size` for segmentation. | Iteratively expand `other_items` (e.g., add `sin`, `x^3`). |

---

## 3. Tool Definition (smolagents)

The agent interacts with the `Dainarx-pq` codebase through a set of wrapped tools.

### 3.1 Core Tools

#### `DataAnalyzer`
*   **Input:** `.npz` file path.
*   **Function:** Computes basic stats (min, max, variance), derivative estimates (is velocity changing abruptly?), and visual density of "jumps".
*   **Output:** `DataProfile` (e.g., `{"has_jumps": True, "smoothness": 0.4, "suggested_modes": 2}`).

#### `RunIdentification`
*   **Input:** `config` dictionary (JSON).
*   **Function:** Wraps the `run()` function from `main.py`.
    *   Calls `find_change_point`, `slice_curve`, `clustering`, `guard_learning`, `build_system`.
*   **Output:** `HA_Model` object (or path to generated JSON) + `training_metrics`.

#### `SimulatorValidator`
*   **Input:** `HA_Model` (JSON path), `Test Data` (npz).
*   **Function:** Runs the simulation loop (predict next state) and computes metrics.
*   **Output:** `ScoreReport` (`RMSE`, `Mode_Transition_Accuracy`, `Frechet_Distance`).

#### `VisualCritic` (Multimodal)
*   **Input:** Image file path (plot).
*   **Function:** Uses a Vision-Language Model (VLM) to describe the system.
*   **Output:** Text description (e.g., "A particle bouncing under gravity," "A chaotic attractor with two lobes").

---

## 4. Detailed Workflow

### Phase 1: Initialization & Profiling
1.  **User Input:** `data/test.npz` (and optional `plot.png`).
2.  **Data Analysis:**
    *   Calculate `jump_score`: $\max(|\Delta x|)$. If high -> **Case A**.
    *   Calculate `non_linearity_score`: Linear regression residuals. If high -> **Case B** (needs `other_items`).

### Phase 2: Hypothesis Loop (The Agent)

The `ManagerAgent` maintains a `Thought Trace`:

> **Thought 1:** Data has sharp jumps (`jump_score > threshold`). Visuals show bouncing.
> **Action:** Propose `Config_v1`: `{order: 1, need_reset: True, other_items: '', kernel: 'linear'}`.
>
> **Observation 1:** `RunIdentification` successful. `SimulatorValidator` reports RMSE=0.05, but Transition Accuracy=60%. The model misses some bounces.
>
> **Thought 2:** The model detects the physics (linear gravity) but misses the exact switch timing. The `linear` kernel might be too simple for the guard, or `window_size` is too large. Let's try `rbf` kernel and reduce window.
> **Action:** Propose `Config_v2`: `{order: 1, need_reset: True, other_items: '', kernel: 'rbf', window_size: 5}`.
>
> **Observation 2:** RMSE=0.01, Transition Accuracy=95%. Excellent fit.

### Phase 3: Non-Linear Discovery (e.g., Duffing)

> **Thought 1:** Data is smooth (`jump_score` low). Linear fit is poor. Visuals show two lobes (Duffing).
> **Action:** Propose `Config_v1`: `{order: 2, other_items: '', self_loop: True}`.
>
> **Observation 1:** RMSE=0.8 (High). Linear model fails.
>
> **Thought 2:** Need non-linear terms. Visuals suggest cubic nature or oscillation. Let's search the function space.
> **Action:** Propose `Config_v2`: `{order: 2, other_items: 'x[?]**3; x[?]**2', self_loop: True}`.
>
> **Observation 2:** RMSE=0.02. `x[?]**3` term was significant.

---

## 5. Implementation Plan

### 5.1 Directory Structure
```
src/
  agent/
    __init__.py
    tools.py       # Wrappers for DAINARX functions
    prompts.py     # System prompts for hypothesis generation
    manager.py     # The smolagents orchestration logic
```

### 5.2 `smolagents` Tool Wrapper Example

```python
from smolagents import tool
from src.main import run, get_config_from_dict
from src.Evaluation import Evaluation

@tool
def identify_hybrid_system(config: dict, data_path: str) -> dict:
    """
    Runs the DAINARX identification pipeline with the given configuration.
    
    Args:
        config: A dictionary containing 'order', 'dt', 'need_reset', 'other_items', etc.
        data_path: Path to the .npz data file.
    """
    # Load data
    data_list, input_data = load_data_internal(data_path) 
    
    # Initialize Evaluation context
    eval_obj = Evaluation("agent_run")
    
    # Run pipeline
    try:
        sys, slice_data = run(data_list, input_data, config, eval_obj)
        
        # Quick simulation for validation score
        score = internal_simulation_score(sys, data_list, input_data)
        return {"status": "success", "score": score, "model_summary": str(sys)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### 5.3 Configuration Search Space (The "Priors")

The Agent will be restricted to proposing values from:
*   `order`: `[1, 2, 3]`
*   `window_size`: `[5, 10, 20]`
*   `kernel`: `['linear', 'rbf', 'poly']`
*   `other_items`: `['', 'x[?]**2', 'x[?]**3', 'sin(x[?])', 'cos(x[?])', 'x[?]**3;sin(x[?])']`
*   `need_reset`: `[True, False]`

---

## 6. Integration & Compatibility

1.  **Input Compatibility:** The agent accepts existing `data/*.npz` files without modification.
2.  **Output Compatibility:** The agent outputs a standard DAINARX JSON file (same as `automata/ATVA/*.json`).
3.  **Simulator Compatibility:** The resulting JSON is strictly validated against `HybridAutomata.from_json` in `src/HybridAutomata.py` to ensure it is runnable.

## 7. Future Work
*   **Parallel Hypothesis Testing:** Run multiple `RunIdentification` calls in parallel for different configs.
*   **Symbolic Regression Library:** Integrate `pysindy` more deeply to automatically suggest `other_items` instead of LLM guessing.

