# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dainarx is a prototypical tool for the derivative-agnostic inference of nonlinear hybrid automata with high-order NARX-modeled dynamics from input-output discrete-time traces of hybrid systems. It learns hybrid automata models from time series data without requiring derivatives.

## Core Architecture

The system follows a multi-stage pipeline:
1. **Data Generation** (`CreatData.py`) - Generates training data from automaton specifications
2. **Change Point Detection** (`src/ChangePoints.py`) - Identifies switching points in time series
3. **Curve Slicing** (`src/CurveSlice.py`) - Segments data based on change points
4. **Clustering** (`src/Clustering.py`) - Groups similar dynamics into modes
5. **Guard Learning** (`src/GuardLearning.py`) - Learns transition conditions between modes
6. **System Building** (`src/BuildSystem.py`) - Constructs the final hybrid automaton
7. **Evaluation** (`src/Evaluation.py`) - Validates results against ground truth

Key components:
- `HybridAutomata` class represents the learned automaton with modes and transitions
- `FeatureExtractor` (in `DEConfig.py`) handles NARX feature extraction
- `Slice` objects represent segments of dynamics in different modes
- `Node` class represents individual modes with their ODEs

## Dependencies

Required packages (install with pip):
```bash
pip install numpy scikit-learn networkx matplotlib
```

- `scikit-learn` for SVM-based guard learning
- `networkx` for computing evaluation metrics
- `matplotlib` for plotting results

## Common Commands

### Run single automaton test:
```bash
python main.py
```
The automaton path is specified in `main.py` (currently set to `"./automata/non_linear/duffing.json"`).

### Run all automata tests:
```bash
python test_all.py
```
This generates `evaluation_log.csv` with results for all automata in the `automata/` directory.

### Generate data only:
The system automatically regenerates data when the automaton specification or key parameters change. Data is cached using SHA256 hashes in the `data/` directory.

## Data Organization

- `automata/` - Contains JSON automaton specifications organized by type:
  - `linear/` - Linear dynamics automata
  - `non_linear/` - Nonlinear dynamics automata  
  - `ATVA/` - ATVA benchmark automata
  - `FaMoS/` - FaMoS benchmark automata
- `data/` - Generated time series data (NPZ files)
- `result/` - Output directory for evaluation logs and plots

## Configuration Parameters

Key parameters in automaton JSON files under `"config"`:
- `dt`: Discrete time step (default: 0.01)
- `total_time`: Total sampling time (default: 10.0)
- `order`: NARX difference equation order (default: 3)
- `window_size`: Sliding window size for change point detection (default: 10)
- `clustering_method`: "fit" or "dis" (default: "fit")
- `kernel`: SVM kernel function (default: "linear")
- `svm_c`: SVM regularization parameter (default: 1e6)
- `need_reset`: Whether to learn reset functions (default: false)
- `self_loop`: Whether to allow self-loops (default: false)

## Testing

The system uses a train/test split where the last 6 trajectories are used for training and the first trajectories for testing (configurable via `test_num` in main.py).

Evaluation metrics include:
- Mode clustering accuracy
- Change point detection accuracy  
- Trajectory prediction error
- Timing measurements for each pipeline stage

Results are saved to `result/eval_log.json` and plots to `result/plot_data.png` and `result/plot_mode.png`.