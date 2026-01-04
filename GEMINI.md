# Dainarx Project Guide

## Project Overview

Dainarx is a prototypical tool designed for the derivative-agnostic inference of nonlinear hybrid automata. It utilizes high-order NARX (Nonlinear AutoRegressive with eXogenous inputs) models to learn dynamics from input-output discrete-time traces of hybrid systems. The system supports complex, nonlinear dynamics without requiring explicit derivative information.

## Core Architecture

The project operates on a multi-stage inference pipeline:

1.  **Data Generation:** Simulates hybrid automata based on JSON specifications to create training traces (`CreatData.py`).
2.  **Change Point Detection:** Identifies switching points between different modes in the time series (`src/ChangePoints.py`).
3.  **Curve Slicing:** Segments the time series data based on detected change points (`src/CurveSlice.py`).
4.  **Clustering:** Groups similar dynamic segments into distinct modes (`src/Clustering.py`).
5.  **Guard Learning:** Learns the transition conditions (guards) between modes using SVMs (`src/GuardLearning.py`).
6.  **System Building:** Constructs the final Hybrid Automaton model from the learned components (`src/BuildSystem.py`).
7.  **Evaluation:** Compares the inferred model against the ground truth (`src/Evaluation.py`).

## Getting Started

### Prerequisites

-   Python 3.9
-   Required packages:
    ```bash
    pip install numpy scikit-learn networkx matplotlib
    ```

### Running the Project

**1. Run a Single Automaton Test**

To run the inference pipeline on a specific automaton (configured in `main.py`):

```bash
python main.py
```

*   **Configuration:** The target automaton is currently set to `./automata/non_linear/duffing.json` inside `main.py`. Modify the `eval_log = main(...)` line to test different automata.
*   **Output:** Generates plots (`plot_data.png`, `plot_mode.png`) and logs in the `result/` directory.

**2. Run All Tests**

To execute tests for all automata defined in the `automata/` directory:

```bash
python test_all.py
```

*   **Output:** Generates a comprehensive `evaluation_log.csv` in the root directory.

## Project Structure

*   **`main.py`**: The entry point for the application. Handles configuration, data loading, and orchestration of the inference pipeline.
*   **`src/`**: Contains the core logic modules:
    *   `HybridAutomata.py`: Defines the hybrid automaton structure and simulation logic.
    *   `DEConfig.py`: Handles feature extraction for NARX models.
    *   `Evaluation.py`: Tools for calculating metrics and comparing results.
*   **`automata/`**: JSON files defining ground truth hybrid automata, organized by type (e.g., `linear`, `non_linear`).
*   **`data/`**: Stores generated time-series data (NPZ format). Data is cached and hashed to prevent unnecessary regeneration.
*   **`result/`**: Output directory for evaluation logs and visual plots.

## Configuration (JSON)

Automata are defined in JSON files (e.g., `automata/non_linear/duffing.json`). Key configuration parameters include:

*   `dt`: Discrete time step (default: 0.01).
*   `order`: Order of the difference equation (default: 3).
*   `window_size`: Window size for change point detection.
*   `clustering_method`: Algorithm for mode clustering ("fit" or "dis").
*   `other_items`: Definition for custom nonlinear or cross terms in the difference equation.

## Development Conventions

*   **Data-Driven:** The system is heavily data-driven. Changes to JSON configurations or core parameters automatically trigger data regeneration.
*   **Evaluation:** The system uses a train/test split (last 6 trajectories for training by default). Metrics include mode clustering accuracy, change point detection accuracy, and trajectory prediction error.
*   **Visualization:** `matplotlib` is used extensively for debugging and result visualization. Check `result/` for generated plots after runs.
