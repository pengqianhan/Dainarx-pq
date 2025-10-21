# Repository Guidelines

## Project Structure & Module Organization
Core source lives in `src/`, with mode learning logic in `HybridAutomata.py`, discrete equation builders in `DE.py` / `DE_System.py`, and guard utilities in `GuardLearning.py`. Entry points `main.py`, `HybridAutomata_simulate.py`, and `HybridAutomata_test.py` sit at the repo root for quick CLI use. Automaton definitions belong in `automata/`; generated trajectories and intermediate traces populate `data/`, `data_duffing/`, or `data_ball/`, while plots and CSV metrics land in `result/`. Keep research collateral under `LearningNotes/` and agent prompts under `prompts/`.

## Build, Test, and Development Commands
Use Python 3.9. Install runtime deps once: `pip install numpy scikit-learn networkx matplotlib`. Typical loops:
- `python main.py automata/example.json` learns from a single automaton spec and prints scores.
- `python test_all.py` sweeps every JSON in `automata/` and rewrites `evaluation_log.csv`.
- `python HybridAutomata_simulate.py --config automata/example.json --out result/example_run` runs the simulator; add `--plot` to emit figures.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents and snake_case for functions, variables, and filenames added under `src/`. Modules already using PascalCase filenames should stay consistent with their peers. Prefer explicit imports (`import numpy as np`, `from src import ...`) and module-level constants in ALL_CAPS. Keep functions pure where possible, document side effects in short docstrings, and type-hint new public APIs.

## Testing Guidelines
Before opening a PR, run `python test_all.py`; confirm `evaluation_log.csv` only contains your intended scenarios. For targeted debugging, create short scripts under `test/` or extend `HybridAutomata_test.py` with descriptive function names like `test_duffing_reset_logic`. Capture deterministic seeds in configs so regenerated traces remain comparable. If adding new automata configs, include at least one reproducible trace set in `data/` and describe coverage expectations in the PR notes.

## Commit & Pull Request Guidelines
Git history favors imperative, descriptive subjects (e.g., `Refactor get_init_state_ha logic`). Keep summaries under 72 chars, elaborating details in wrapped body paragraphs when needed. Link issues or experiments via `Refs:` in the body, and attach plots or CSV diffs when behavior changes. PRs should outline motivation, validation commands run, impacted automata, and any new data files to help reviewers reproduce results quickly.
