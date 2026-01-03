# Hybrid Automaton LLM Agent: Complete Design Document

> **Objective**: Design an LLM agent that infers Hybrid Automata (HA) directly from npz data and/or trajectory plots, automatically determining configuration parameters (`order`, `need_reset`, `self_loop`, `kernel`, `other_items`) without manual prior specification.

---

## 1. Problem Statement

### 1.1 Current Limitations
The existing DAINARX pipeline requires manual specification of:
- `order`: NARX model history length (1-3)
- `need_reset`: Whether to learn reset functions
- `self_loop`: Whether to allow self-loop transitions
- `kernel`: SVM kernel for guard learning (`linear`, `rbf`, `poly`)
- `other_items`: Nonlinear term expressions (e.g., `x[?] ** 3`, `sin(x[?])`)

These parameters are currently read from JSON configuration, limiting generalization to new systems.

### 1.2 Key Observations
| System | Mode Switching | Visual Signature | Challenge |
|--------|---------------|------------------|-----------|
| **Bouncing Ball** | Clear (ground contact) | Sharp discontinuities, velocity reversals | Guard/reset detection |
| **Duffing** | Subtle or none | Smooth oscillations, no obvious breaks | Nonlinear term discovery |

### 1.3 Design Goals
1. **Automatic parameter inference** from data/plots
2. **Unified handling** of both clear-switching and subtle-switching systems
3. **Closed-loop verification** with simulation feedback
4. **Seamless integration** with existing DAINARX JSON format and simulator

---

## 2. Architecture Overview

### 2.1 SR-Scientist Inspired Design Principles

Following the SR-Scientist paradigm of **program synthesis with verification**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PLANNER AGENT (smolagent)                    │
│  - Orchestrates tool calls                                          │
│  - Manages beam search state                                        │
│  - Proposes hypotheses (K, lib_config, guard/reset forms)           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │   PROPOSE   │         │   EXECUTE   │         │  EVALUATE   │
   │ - K modes   │         │ - Segment   │         │ - RMSE      │
   │ - lib_config│         │ - Fit SINDy │         │ - Event MAE │
   │ - Guard form│         │ - Learn     │         │ - Switch F1 │
   │ - Reset form│         │   guards    │         │ - MDL       │
   └─────────────┘         └─────────────┘         └─────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
                          ┌─────────────┐
                          │   REFINE    │
                          │ - Error     │
                          │   analysis  │
                          │ - Adjust    │
                          │   params    │
                          └─────────────┘
```

### 2.2 Core Components

1. **Data Analyzer**: Extracts features, estimates derivatives, detects events
2. **Mode Discoverer**: Change point detection, clustering, mode count selection
3. **Equation Learner**: SINDy fitting with adaptive library configuration
4. **Guard/Reset Learner**: Boundary classification and reset function fitting
5. **Assembler & Validator**: JSON generation and simulation-based scoring
6. **Image Analyzer** (optional): Visual hints for weak priors

---

## 3. End-to-End Workflow

### 3.1 Pipeline Overview

```
npz Data ──┬──► [1] Load & Preprocess ──► [2] Derivative Estimation
           │                                        │
(Optional) │                                        ▼
Plot Image ──► [3] Image Analysis ──────► [4] Mode Discovery
                    (weak prior)                    │
                                                    ▼
                                    [5] Library Proposal (LLM)
                                                    │
                                                    ▼
                                    [6] SINDy Fitting (per mode)
                                                    │
                                                    ▼
                                    [7] Guard/Reset Learning
                                                    │
                                                    ▼
                                    [8] Assemble HA JSON
                                                    │
                                                    ▼
                                    [9] Simulate & Score
                                                    │
                                          ┌────────┴────────┐
                                          ▼                 ▼
                                    Score OK?          Score Bad?
                                          │                 │
                                          ▼                 ▼
                                    Output JSON       [10] Refine
                                                      (iterate)
```

### 3.2 Detailed Stage Descriptions

#### Stage 1: Load & Preprocess
```python
def load_npz(path: str) -> Dict:
    """
    Returns:
        - t: time array (N,)
        - X: state array (var_num, N)
        - U: input array (input_num, N) or None
        - mode_gt: ground truth modes (N,) if available
        - change_points_gt: ground truth change points if available
    """
```

#### Stage 2: Derivative Estimation
```python
def estimate_derivatives(t, X, method='savgol', params=None) -> Dict:
    """
    Methods:
        - 'savgol': Savitzky-Golay filter (default)
        - 'tvreg': Total Variation Regularization (noisy data)
        - 'finite': Simple finite differences

    Returns:
        - Xdot: first derivative (var_num, N)
        - Xddot: second derivative (var_num, N) if needed
        - best_params: auto-selected window/order via AIC
    """
```

#### Stage 3: Image Analysis (Optional, Weak Prior)
```python
def analyze_image(image_path: str) -> ImagePrior:
    """
    Returns:
        - rough_event_rate: float in [0, 1] (high = many mode switches)
        - has_floor_line: bool (detected ground plane)
        - symmetry: str ('bilateral', 'periodic', 'none')
        - trajectory_shape: str ('oscillatory', 'bouncing', 'smooth')
        - suggested_terms: List[str] (e.g., ['x^3', 'sin(x)'])
    """
```

#### Stage 4: Mode Discovery
```python
def discover_modes(X, Xdot, features, image_prior=None) -> ModeDiscoveryResult:
    """
    Steps:
        1. Compute features: [x, xdot, xddot, local_energy, residual]
        2. Detect change point candidates (PELT/BinSeg + dynamic programming)
        3. Cluster segments (KMeans/HDBSCAN)
        4. Select K via Silhouette + MDL/BIC
        5. Smooth labels (dynamic programming to minimize switch cost)

    Returns:
        - K: number of modes
        - labels: mode assignment per timestep
        - change_points: detected switching times
        - segments: list of Segment objects
    """
```

#### Stage 5: Library Proposal (LLM Decision)
```python
def propose_library(
    data_summary: DataSummary,
    mode_info: ModeDiscoveryResult,
    image_prior: Optional[ImagePrior]
) -> LibraryConfig:
    """
    LLM proposes term library based on:
        - Data statistics (range, variance, correlation)
        - Mode structure (K, segment lengths)
        - Image hints (if available)

    Returns:
        - poly_degree: int (1, 2, or 3)
        - include_trig: bool
        - include_cross: bool
        - specific_terms: List[str] (e.g., ['x[?]**3', 'x[?]*x_[?]'])
    """
```

#### Stage 6: SINDy Fitting
```python
def sindy_fit(
    X, Xdot, labels, lib_config: LibraryConfig, cv=5
) -> List[ModeEquation]:
    """
    Per-mode sparse regression using STRidge or LassoCV.

    Returns list of:
        - terms: List[str] (active terms)
        - coeffs: np.ndarray (coefficients)
        - scores: Dict (r2, rmse, mdl, cv_score)
    """
```

#### Stage 7: Guard/Reset Learning
```python
def learn_guard(boundary_samples, direction_samples, forms=['linear', 'quad']):
    """
    Fit guard condition at mode boundaries.
    Uses SVM or RANSAC-based hyperplane fitting.

    Returns:
        - expr: str (e.g., 'x1 <= 0')
        - direction: str (e.g., 'x2 < 0' for entry condition)
        - score: float
    """

def learn_reset(pre_states, post_states, forms=['identity', 'linear', 'affine']):
    """
    Fit reset function from pre/post state pairs.

    Returns:
        - reset_map: Dict[var, expr]
        - score: float
        - is_identity: bool
    """
```

#### Stage 8: Assemble HA JSON
```python
def assemble_ha_json(
    modes: List[ModeEquation],
    edges: List[Edge],
    meta: Dict
) -> Dict:
    """
    Produces JSON compatible with existing DAINARX format.
    See Section 6 for detailed schema.
    """
```

#### Stage 9: Simulate & Score
```python
def simulate_and_score(ha_json, t_ref, x_ref, x0) -> Metrics:
    """
    Simulate the learned HA and compare to reference data.

    Returns:
        - rmse: trajectory RMSE
        - event_mae: mean absolute error of switch times
        - switch_f1: precision/recall of mode switches
        - frechet: Frechet distance
        - mdl: model description length
        - total_score: weighted combination
    """
```

#### Stage 10: Refinement (Closed-Loop)
```python
def refine(metrics: Metrics, current_config: Config) -> Config:
    """
    Error attribution and parameter adjustment:

    - High segment residual → expand library (add terms)
    - Low switch F1 → adjust segmentation threshold
    - High MDL → prune terms, reduce K
    - Event MAE high → refine guard forms
    """
```

---

## 4. Tool Design for smolagent

### 4.1 Tool Registry

```python
from smolagent import Tool, Agent

TOOL_REGISTRY = {
    # Data & Features
    "load_npz": LoadNPZTool,
    "estimate_derivatives": DerivativeTool,
    "compute_features": FeatureTool,

    # Segmentation & Clustering
    "detect_changepoints": ChangepointTool,
    "select_num_modes": ModeSelectionTool,
    "assign_modes": ModeAssignmentTool,

    # Equation Learning
    "propose_library": LibraryProposalTool,  # LLM-powered
    "sindy_fit": SindyFitTool,
    "sr_symbolic_search": SymbolicSearchTool,  # optional

    # Guard & Reset
    "learn_guard": GuardLearningTool,
    "learn_reset": ResetLearningTool,

    # Assembly & Validation
    "assemble_ha_json": AssembleTool,
    "simulate_and_score": SimScoreTool,
    "validate_json": ValidateTool,

    # Image Analysis (optional)
    "describe_plot": ImageDescribeTool,
    "image_event_density": EventDensityTool,

    # Utility
    "read_json": ReadJSONTool,
    "write_json": WriteJSONTool,
    "plot_comparison": PlotTool,
}
```

### 4.2 Tool Interface Specifications

```python
class SindyFitTool(Tool):
    name = "sindy_fit"
    description = """
    Fit sparse dynamics equations using SINDy (Sparse Identification of
    Nonlinear Dynamics). Supports polynomial, trigonometric, and custom
    basis functions.

    Use this tool after mode discovery to fit equations for each mode.
    """
    inputs = {
        "X": {"type": "array", "description": "State data (var_num, N)"},
        "Xdot": {"type": "array", "description": "Derivative data"},
        "labels": {"type": "array", "description": "Mode labels per timestep"},
        "lib_config": {"type": "json", "description": "Library configuration"}
    }
    output_schema = {
        "modes": [{"terms": list, "coeffs": list, "scores": dict}]
    }

    def __call__(self, X, Xdot, labels, lib_config):
        # Implementation using STRidge/LassoCV
        pass


class LibraryProposalTool(Tool):
    name = "propose_library"
    description = """
    Propose a term library configuration for SINDy fitting based on
    data characteristics and mode structure.

    This is an LLM-powered tool that analyzes data summaries and
    suggests appropriate basis functions.
    """
    inputs = {
        "data_summary": {"type": "json", "description": "Statistics about the data"},
        "mode_info": {"type": "json", "description": "Mode discovery results"},
        "image_hints": {"type": "json", "description": "Optional image analysis hints"}
    }
    output_schema = {
        "poly_degree": int,
        "include_trig": bool,
        "include_cross": bool,
        "specific_terms": list
    }
```

### 4.3 DSL for Constrained Expression

To ensure safety and compatibility, LLM outputs are constrained to a DSL:

```python
# Allowed ODE terms
ODE_DSL = {
    "polynomial": ["1", "x[?]", "x[?]**2", "x[?]**3"],
    "cross_terms": ["x[?]*x_[?]", "x[?]**2*x_[?]"],
    "trigonometric": ["sin(x[?])", "cos(x[?])", "sin(w*t)", "cos(w*t)"],
    "nonsmooth": ["sign(x[?])", "abs(x[?])"],
    "velocity": ["xdot[?]", "xdot[?]**2"],
}

# Allowed guard forms
GUARD_DSL = {
    "linear": "a.T @ x + b <= 0",
    "quadratic": "x.T @ Q @ x + b.T @ x + c <= 0",
    "componentwise": "x[i] <= threshold",
}

# Allowed reset forms
RESET_DSL = {
    "identity": "x+ = x-",
    "linear": "x+ = A @ x-",
    "affine": "x+ = A @ x- + b",
    "componentwise": "x[i]+ = c * x[i]-",
}
```

---

## 5. Search and Scoring Strategy

### 5.1 Beam Search Configuration

```python
BEAM_CONFIG = {
    "width": 3,           # Number of candidates to keep
    "max_iterations": 5,  # Maximum refinement rounds
    "temperature": [0.7, 1.0, 1.3],  # For diverse candidate generation
}
```

### 5.2 Scoring Function

```python
def compute_score(metrics: Metrics, weights: Weights) -> float:
    """
    Total score = alpha * RMSE + beta * EventMAE + gamma * (1 - SwitchF1) + lambda * MDL

    Default weights for different scenarios:

    Clear Switching (Ball-like):
        alpha=0.30, beta=0.35, gamma=0.25, lambda=0.10

    Subtle Switching (Duffing-like):
        alpha=0.50, beta=0.10, gamma=0.05, lambda=0.35
    """
    return (
        weights.alpha * metrics.rmse +
        weights.beta * metrics.event_mae +
        weights.gamma * (1 - metrics.switch_f1) +
        weights.lambda_ * metrics.mdl
    )
```

### 5.3 MDL (Minimum Description Length) Calculation

```python
def compute_mdl(ha_json: Dict) -> float:
    """
    MDL = log(K) + sum_modes(log(n_terms)) + sum_edges(complexity(guard) + complexity(reset))

    Encourages simpler models with fewer modes and terms.
    """
    K = len(ha_json["modes"])
    term_cost = sum(len(mode["terms"]) for mode in ha_json["modes"])
    edge_cost = sum(edge_complexity(e) for e in ha_json["edges"])
    return np.log(K + 1) + 0.5 * term_cost + 0.3 * edge_cost
```

### 5.4 Self-Consistency Filtering

```python
def filter_by_consensus(candidates: List[HAJson], threshold=0.6) -> List[str]:
    """
    Generate multiple candidates with different temperatures.
    Keep terms that appear in >= threshold fraction of candidates.

    Example: If 'x**3' appears in 4/5 candidates, keep it.
    """
    term_counts = Counter()
    for c in candidates:
        for mode in c["modes"]:
            term_counts.update(mode["terms"])

    n = len(candidates)
    return [term for term, count in term_counts.items() if count / n >= threshold]
```

---

## 6. Handling Two Image Scenarios

### 6.1 Scenario A: Clear Mode Switching (e.g., Bouncing Ball)

**Visual Characteristics:**
- Sharp discontinuities in trajectory
- Velocity reversals at boundaries
- Clear geometric features (ground lines, impact points)

**Processing Strategy:**

```python
def handle_clear_switching(data, image_prior):
    """
    Priority: Accurate switch detection and guard/reset learning
    """
    config = {
        # Segmentation: more sensitive
        "changepoint_sensitivity": "high",
        "min_segment_length": 5,  # Allow short segments

        # Mode selection: favor K >= 2
        "k_prior_bias": +0.3,  # Boost multi-mode candidates
        "silhouette_threshold": 0.4,  # Accept moderate clustering

        # Guard learning: prioritize linear + direction
        "guard_forms": ["linear_directional", "linear", "quadratic"],
        "guard_direction_constraint": True,

        # Reset learning: prioritize physical (elastic collision)
        "reset_forms": ["componentwise_damped", "affine", "linear"],
        "reset_prior": "v+ = -c * v-",  # Physical prior

        # Scoring weights: emphasize event accuracy
        "weights": Weights(alpha=0.30, beta=0.35, gamma=0.25, lambda_=0.10),
    }
    return config
```

**Image-Enhanced Features:**
```python
if image_prior.has_floor_line:
    # Strong bias toward guard: y = 0 with direction ydot < 0
    guard_prior = {"expr": "x1 <= 0", "direction": "x2 < 0"}

if image_prior.rough_event_rate > 0.5:
    # High event density suggests multiple switches
    k_range = [2, 3, 4]  # Explore K >= 2
```

### 6.2 Scenario B: Subtle/No Mode Switching (e.g., Duffing)

**Visual Characteristics:**
- Smooth oscillations
- No obvious discontinuities
- Possibly nonlinear patterns (limit cycles, chaos)

**Processing Strategy:**

```python
def handle_subtle_switching(data, image_prior):
    """
    Priority: Accurate nonlinear term discovery
    """
    config = {
        # Segmentation: conservative
        "changepoint_sensitivity": "low",
        "min_segment_length": 50,  # Require longer segments

        # Mode selection: favor K = 1
        "k_prior_bias": -0.5,  # Penalize multi-mode
        "silhouette_threshold": 0.7,  # Require strong clustering

        # Equation learning: rich library
        "poly_degree": 3,
        "include_trig": True,
        "include_cross": True,

        # Guard/Reset: minimal
        "guard_forms": ["none", "linear"],  # Likely no guards
        "reset_forms": ["identity"],

        # Scoring weights: emphasize trajectory fit and simplicity
        "weights": Weights(alpha=0.50, beta=0.10, gamma=0.05, lambda_=0.35),
    }
    return config
```

**Image-Enhanced Features:**
```python
if image_prior.symmetry == "bilateral":
    # Suggests odd-power terms like x^3
    suggested_terms.append("x[?]**3")

if image_prior.trajectory_shape == "oscillatory":
    # May have damping or forcing
    suggested_terms.extend(["xdot[?]", "sin(w*t)"])
```

### 6.3 Adaptive Scenario Detection

```python
def detect_scenario(data, image_prior) -> str:
    """
    Automatically determine which scenario we're in.
    """
    # Compute derivative discontinuity measure
    Xdot = estimate_derivatives(data["t"], data["X"])
    jump_score = np.max(np.abs(np.diff(Xdot, axis=1)))

    # Compute event density from data
    features = compute_features(data["X"], Xdot)
    event_candidates = detect_changepoints(features, sensitivity="medium")
    event_density = len(event_candidates) / len(data["t"])

    # Combine with image prior if available
    if image_prior:
        event_density = 0.7 * event_density + 0.3 * image_prior.rough_event_rate

    # Decision
    if jump_score > 10 * np.std(Xdot) or event_density > 0.05:
        return "clear_switching"
    else:
        return "subtle_switching"
```

### 6.4 Conflict Resolution

When image prior conflicts with data evidence:

```python
def resolve_conflict(data_evidence, image_prior):
    """
    Data evidence takes precedence, but keep alternatives.
    """
    candidates = []

    # Always include data-driven candidate
    k_data = data_evidence.suggested_k
    candidates.append({"K": k_data, "source": "data", "priority": 1.0})

    # Include image-suggested candidate if different
    if image_prior and image_prior.suggested_k != k_data:
        k_image = image_prior.suggested_k
        candidates.append({"K": k_image, "source": "image", "priority": 0.7})

    # Include K+1 and K-1 as alternatives
    candidates.append({"K": k_data + 1, "source": "explore", "priority": 0.5})
    if k_data > 1:
        candidates.append({"K": k_data - 1, "source": "explore", "priority": 0.5})

    return candidates
```

---

## 7. JSON Output Specification

### 7.1 Output Format (Compatible with DAINARX)

```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "u1",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2[0], x2[1] = -0.5*x2[0] + x1[0] - 1.5*x1[0]**3 + u1"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "true",
        "reset": {}
      }
    ]
  },
  "init_state": [],
  "config": {
    "dt": 0.01,
    "total_time": 10.0,
    "order": 2,
    "need_reset": false,
    "kernel": "linear",
    "other_items": "x[?] ** 3",
    "self_loop": true
  },
  "meta": {
    "generated_by": "HA_LLM_Agent",
    "timestamp": "2024-01-15T10:30:00Z",
    "scores": {
      "rmse": 0.0123,
      "event_mae": 0.0,
      "switch_f1": 1.0,
      "mdl": 2.34,
      "total": 0.45
    },
    "search_info": {
      "iterations": 3,
      "candidates_explored": 12,
      "final_K": 1
    },
    "inferred_params": {
      "order": 2,
      "need_reset": false,
      "self_loop": true,
      "kernel": "linear",
      "other_items": "x[?] ** 3"
    }
  }
}
```

### 7.2 Equation Format Conversion

The agent outputs equations in a human-readable format that must be converted to DAINARX's discrete difference equation format:

```python
def convert_to_dainarx_format(continuous_eq: str, dt: float, order: int) -> str:
    """
    Convert: dx/dt = -0.5*x - 1.5*x^3 + u
    To: x[2] = x[1] + dt * (-0.5*x[1] - 1.5*x[1]**3 + u)
    Or for order=2: x[2] = a*x[1] + b*x[0] + c*x[1]**3 + d*u

    The conversion depends on discretization method and order.
    """
    pass
```

### 7.3 Validation

```python
def validate_output_json(ha_json: Dict) -> ValidationResult:
    """
    Check:
    1. Schema compliance (required fields present)
    2. Expression validity (can be parsed and evaluated)
    3. Mode/edge consistency (all referenced modes exist)
    4. Config completeness (all parameters specified)
    """
    errors = []
    warnings = []

    # Check required fields
    required = ["automaton", "config"]
    for field in required:
        if field not in ha_json:
            errors.append(f"Missing required field: {field}")

    # Validate expressions
    for mode in ha_json.get("automaton", {}).get("mode", []):
        try:
            parse_equation(mode["eq"])
        except:
            errors.append(f"Invalid equation in mode {mode['id']}")

    return ValidationResult(valid=len(errors)==0, errors=errors, warnings=warnings)
```

---

## 8. Integration with Existing Repository

### 8.1 File Structure

```
Dainarx-pq/
├── src/
│   ├── agent/                      # NEW: LLM Agent module
│   │   ├── __init__.py
│   │   ├── agent_ha.py             # Main agent entry point
│   │   ├── planner.py              # Planner agent logic
│   │   ├── critic.py               # Optional critic agent
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── data_tools.py       # load_npz, derivatives, features
│   │       ├── segment_tools.py    # changepoints, clustering
│   │       ├── equation_tools.py   # sindy_fit, sr_search
│   │       ├── guard_tools.py      # guard/reset learning
│   │       ├── assembly_tools.py   # json assembly, validation
│   │       └── image_tools.py      # image analysis (optional)
│   │
│   ├── DEConfig.py                 # Existing (unchanged)
│   ├── ChangePoints.py             # Existing (can be wrapped as tool)
│   ├── Clustering.py               # Existing (can be wrapped as tool)
│   ├── GuardLearning.py            # Existing (can be wrapped as tool)
│   └── ...
│
├── prompts/                        # NEW: Agent prompts
│   ├── planner.md                  # Planner system prompt
│   ├── critic.md                   # Critic system prompt
│   └── library_proposal.md         # Library suggestion prompt
│
├── main.py                         # Add --agent flag
├── main_agent.py                   # NEW: Agent-specific entry point
└── ...
```

### 8.2 Entry Point Extension

```python
# main_agent.py
import argparse
from src.agent.agent_ha import HALLMAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to npz data file(s)")
    parser.add_argument("--image", help="Optional path to trajectory plot")
    parser.add_argument("--out", default="result/agent_output", help="Output directory")
    parser.add_argument("--max-iter", type=int, default=5, help="Max refinement iterations")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam search width")
    args = parser.parse_args()

    agent = HALLMAgent(
        max_iterations=args.max_iter,
        beam_width=args.beam_width,
    )

    result = agent.run(
        npz_path=args.npz,
        image_path=args.image,
        output_dir=args.out,
    )

    print(f"Generated HA JSON: {result.json_path}")
    print(f"Scores: {result.scores}")

if __name__ == "__main__":
    main()
```

### 8.3 Existing Module Integration

The agent wraps existing DAINARX modules as tools:

```python
# src/agent/tools/segment_tools.py
from src.ChangePoints import find_change_point
from src.Clustering import clustering

class ChangepointTool(Tool):
    def __call__(self, features, config):
        # Wrapper around existing find_change_point
        return find_change_point(features, config)

class ClusteringTool(Tool):
    def __call__(self, slices, config):
        # Wrapper around existing clustering
        return clustering(slices, config)
```

---

## 9. Prompt Templates

### 9.1 Planner System Prompt

```markdown
# HA-LLM Planner Agent

You are an expert in hybrid systems identification. Your task is to infer
Hybrid Automata (HA) from time series data.

## Available Tools
- load_npz: Load trajectory data
- estimate_derivatives: Compute state derivatives
- detect_changepoints: Find mode switching points
- select_num_modes: Choose number of modes K
- propose_library: Suggest basis functions for equation fitting
- sindy_fit: Fit sparse dynamics equations
- learn_guard: Learn guard conditions
- learn_reset: Learn reset functions
- assemble_ha_json: Generate output JSON
- simulate_and_score: Evaluate the learned HA

## Workflow
1. Load data and compute derivatives
2. Detect potential mode switches
3. Select number of modes K (consider both K and K+1)
4. Propose library configuration
5. Fit equations per mode
6. Learn guards and resets (if K > 1)
7. Assemble JSON and evaluate
8. If score unsatisfactory, refine and iterate

## Decision Rules
- If trajectory has clear discontinuities: prioritize K >= 2
- If trajectory is smooth: prioritize K = 1, focus on nonlinear terms
- Always consider MDL to avoid overfitting
- When uncertain, explore multiple K values in parallel

## Output Format
Always output valid JSON compatible with DAINARX format.
Include inferred parameters in meta.inferred_params.
```

### 9.2 Library Proposal Prompt

```markdown
# Term Library Proposal

Based on the following data characteristics, propose a library configuration
for SINDy equation fitting.

## Data Summary
- Variables: {var_names}
- Time range: {t_min} to {t_max}
- Value ranges: {ranges}
- Derivative statistics: {deriv_stats}
- Detected modes: {K}
- Mode switching pattern: {switch_pattern}

## Image Hints (if available)
- Trajectory shape: {shape}
- Symmetry: {symmetry}
- Event density: {event_density}

## Task
Suggest which terms to include in the basis library:
1. Polynomial degree (1, 2, or 3)
2. Include trigonometric terms? (sin, cos)
3. Include cross-terms? (x*y)
4. Specific terms to add (e.g., x^3, sign(xdot))

Respond in JSON format:
{
  "poly_degree": <int>,
  "include_trig": <bool>,
  "include_cross": <bool>,
  "specific_terms": [<term1>, <term2>, ...]
}
```

---

## 10. Implementation Milestones

### Milestone A: Single-Mode Fitting (Duffing)
- [ ] Load npz data
- [ ] Derivative estimation with Savitzky-Golay
- [ ] Library proposal (fixed: poly_degree=3)
- [ ] SINDy fitting with STRidge
- [ ] JSON assembly
- [ ] Simulation scoring
- **Target**: RMSE < 0.05 on Duffing

### Milestone B: Multi-Mode Detection (Ball)
- [ ] Change point detection
- [ ] Clustering and mode assignment
- [ ] K selection with Silhouette/MDL
- [ ] Guard learning (linear SVM)
- [ ] Reset learning (componentwise affine)
- **Target**: Switch F1 > 0.9 on Ball

### Milestone C: Beam Search & Refinement
- [ ] Multiple candidate generation
- [ ] Parallel evaluation
- [ ] Error attribution
- [ ] Iterative refinement
- **Target**: Handle 80% of automata/ benchmarks

### Milestone D: Image Integration
- [ ] Image loading and preprocessing
- [ ] Event density estimation
- [ ] Shape/symmetry detection
- [ ] Prior integration into pipeline
- **Target**: Improve accuracy on ambiguous cases

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Mode not separable (e.g., Duffing) | Fall back to K=1, keep self-loop |
| High noise | Use TVReg for derivatives, RANSAC for fitting |
| Overfitting | MDL penalty, cross-validation, consensus filtering |
| LLM hallucination | DSL constraints, simulation validation |
| Computation cost | Limit beam width, cache intermediate results |
| Image misleading | Data evidence takes precedence |

---

## 12. Dependencies

### Core (existing)
```
numpy
scikit-learn
matplotlib
networkx
```

### Agent (new)
```
smolagent  # Hugging Face agent framework
# OR fallback:
# Simple Python orchestrator with JSON contracts
```

### Optional
```
pysindy  # For SINDy implementation
cvxpy    # For constrained optimization
opencv-python  # For image analysis
```

---

## 13. References

1. **SR-Scientist**: Program synthesis for scientific discovery with LLMs
2. **SINDy**: Sparse Identification of Nonlinear Dynamics (Brunton et al., 2016)
3. **DAINARX**: Derivative-Agnostic Inference of Nonlinear Hybrid Systems
4. **smolagent**: Hugging Face lightweight agent framework
5. **MDL/BIC**: Model selection via description length

---

*Document Version: 1.0*
*Last Updated: 2024-01-15*
