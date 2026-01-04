# HA-LLM-E: Evolutionary Hybrid Automaton Discovery Agent (v3.0)

**Date:** January 4, 2026
**Inspiration:** FunSearch (DeepMind), Eureka (NVIDIA), SR-Scientist
**Goal:** Fully autonomous, prior-free identification of hybrid systems using LLM-driven evolutionary search.

---

## 1. Philosophy: The "Evolutionary Research Scientist"

Unlike previous designs that treated the LLM as a simple "config filler," **HA-LLM-E** treats the identification process as an **evolutionary search** over the space of "Scientific Hypotheses."

The LLM acts as the **Mutation & Crossover Operator**: it looks at the current best models, reasons about their physical deficiencies (e.g., "Model A tracks the frequency but decays too fast; Model B fits the amplitude but misses the phase"), and generates *new* hypotheses (code/config) to fix them.

### Key Shift from v2.0
| Feature | v2.0 (Agentic Workflow) | v3.0 (Evolutionary Search) |
| :--- | :--- | :--- |
| **Search Method** | Linear Loop (Hypothesis -> Test -> Refine) | **Population-Based** (Maintain $N$ diverse hypotheses) |
| **LLM Role** | Decision Maker | **Evolutionary Operator** (Mutator/Generator) |
| **Memory** | Short-term context | **Research Log** (Long-term Reflection & RAG) |
| **Adaptability** | Hardcoded "Case A/B" logic | **Dynamic Strategy** (Discovered by LLM) |

---

## 2. Architecture

The system consists of three main components: the **Scientist (LLM)**, the **Laboratory (DAINARX Tools)**, and the **Archive (Memory)**.

### 2.1 The Scientist (LLM)
*   **Role:** Generates and refines hypotheses.
*   **Input:** `ResearchLog` (past experiments), `Population` (current best models), `DataProfile`.
*   **Output:** New `Config` objects or `FeatureExtraction` code snippets.
*   **Core Prompts:**
    *   `Gen_Initial_Population`: "Propose 3 diverse physical models (e.g., 1 linear damped, 1 chaotic, 1 switching)."
    *   `Mutate`: "Model X has high error at the peaks. Suggest a mathematical term (e.g., cubic stiffness) to fix this."
    *   `Reflect`: "Why did Model Y fail? It predicted a switch that didn't happen. What parameter controls sensitivity?"

### 2.2 The Laboratory (Tools)
Executes the "Experiments" defined by the Scientist.
*   **`run_experiment(config)`**: Runs the `DAINARX` pipeline (`main.py` logic).
*   **`simulate_and_score(HA, data)`**: Returns a multi-dimensional score:
    *   $S_{RMSE}$: Trajectory tracking error.
    *   $S_{Event}$: Event timing accuracy (DTW or Event Matching).
    *   $S_{MDL}$: Model Description Length (parsimony penalty).
    *   $S_{Phy}$: Physical consistency (e.g., energy conservation checks).

### 2.3 The Archive (Memory)
*   **Population:** Stores the top $K$ models found so far.
*   **Experiment Log:** A history of ALL attempts, including failures. Used for "Reflection" to avoid repeating mistakes.
    *   *Entry Example:* `{"id": 5, "config": {...}, "result": "Failed: Infinite Loop", "analysis": "Self-loop enabled without proper guard."}`

---

## 3. Workflow: The Evolutionary Loop

### Step 0: Data Profiling
The agent first analyzes the `.npz` data and plot.
*   **LLM Observation:** "The plot shows a periodic signal that suddenly changes amplitude. This suggests a switched system or a non-linear limit cycle. The derivative is discontinuous, implying a reset map."

### Step 1: Initialization
The LLM generates an initial population of $N=4$ diverse hypotheses:
1.  **H1 (Baseline):** Simple Linear, Order=2.
2.  **H2 (Non-linear):** Order=2, `other_items="x[?]**3"`.
3.  **H3 (Switched):** Order=1, `need_reset=True`.
4.  **H4 (Complex):** Order=2, `self_loop=True`, `kernel='rbf'`.

### Step 2: Evaluation
The Laboratory runs all 4 hypotheses.
*   *Result:* H1 (Score: 0.4), H2 (Score: 0.7), H3 (Score: 0.6), H4 (Score: 0.2 - Crashed).

### Step 3: Evolution (The Loop)
For $T$ generations:

1.  **Selection:** Pick the top survivors (e.g., H2 and H3).
2.  **Reflection:** The LLM analyzes H2 vs H3.
    *   "H2 captures the curve shape but drifts over time. H3 captures the sharp turns but has poor curve fitting."
3.  **Mutation/Crossover:** The LLM proposes new children:
    *   **H5 (Merge):** "Combine H2's cubic term with H3's reset logic." -> `{order: 2, other_items: "x[?]**3", need_reset: True}`.
    *   **H6 (Refine H2):** "Add a damping term to H2." -> `{order: 2, other_items: "x[?]**3; xdot[?]"}`.
4.  **Execution:** Run H5 and H6.
5.  **Update Archive:** If H5 (Score: 0.95) is better, it replaces the worst model in the population.

### Step 4: Final Selection
Return the hypothesis with the highest validation score on held-out data.

---

## 4. Prompt Engineering Strategy

We use **Chain-of-Thought (CoT)** to force the LLM to justify its physical reasoning.

**Template: Mutation Prompt**
```markdown
Current Best Model:
- Config: {order: 2, other_items: "x[?]**3"}
- Performance: RMSE=0.05, but misses the amplitude decay.

Data Analysis:
The trajectory amplitude decreases over time (damped oscillation).

Task:
Propose a mutation to the 'other_items' or 'order' to model this decay.
Explain your reasoning using physics (e.g., "Damping is usually proportional to velocity").

Output JSON:
{
  "reasoning": "The decay suggests a friction term. In a Duffing oscillator, this is linear damping.",
  "new_config": { ... "other_items": "x[?]**3; x[1]" ... }
}
```

---

## 5. Integration with smolagents

We will implement this as a custom `Agent` class in `smolagents` (or a similar loop if using raw API).

```python
class EvolutionaryAgent:
    def __init__(self, tools, memory, llm):
        self.population = []
        self.archive = []
    
    def run(self, data_path):
        # 1. Profile Data
        profile = self.tools.profile_data(data_path)
        
        # 2. Init Population
        self.population = self.llm.generate_initial(profile)
        
        # 3. Evolution Loop
        for generation in range(MAX_GEN):
            # Evaluate
            scores = self.tools.evaluate_batch(self.population)
            
            # Update Archive
            self.archive.extend(zip(self.population, scores))
            
            # Select & Mutate
            best_candidates = self.select_top_k(self.archive)
            new_hypotheses = self.llm.mutate(best_candidates, profile)
            
            self.population = new_hypotheses
            
        return self.get_best_model()
```

---

## 6. Advantages over v2.0
1.  **Robustness:** If the "Dual-Path" logic was wrong in v2.0, the agent would get stuck. Here, the diverse population ensures that if one path fails, another (e.g., the "Switched" hypothesis) might succeed.
2.  **Creativity:** The LLM can invent new feature combinations (`x*y`, `sin(x)`) that were not hardcoded in a "Case B" list.
3.  **Self-Correction:** The "Reflection" step explicitly asks the LLM to learn from failures, mimicking a human researcher.

