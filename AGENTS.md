# Repository Guidelines

## Project Structure & Module Organization
- Core source in `src/`: mode learning in `HybridAutomata.py`, discrete equation builders in `DE.py` / `DE_System.py`, guard utilities in `GuardLearning.py`.
- Entry points at repo root: `main.py`, `HybridAutomata_simulate.py`, `HybridAutomata_test.py`.
- Automata specs in `automata/` (JSON). Generated traces in `data/`, `data_duffing/`, or `data_ball/`. Plots/CSV metrics in `result/`.
- Research notes in `LearningNotes/`; agent prompts in `prompts/`.

## Build, Test, and Development Commands
- Use Python 3.9. Install deps once: `pip install numpy scikit-learn networkx matplotlib`.
- Learn from one spec: `python main.py automata/example.json` (prints scores).
- Sweep all specs: `python test_all.py` (rewrites `evaluation_log.csv`).
- Simulate: `python HybridAutomata_simulate.py --config automata/example.json --out result/example_run --plot`.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indents. Use `snake_case` for new functions, variables, and filenames under `src/`.
- Keep existing PascalCase module filenames consistent with their peers.
- Prefer explicit imports (`import numpy as np`, `from src import ...`). Use module-level constants in `ALL_CAPS`.
- Favor pure functions; document side effects in short docstrings. Type-hint new public APIs.

## Testing Guidelines
- Before PRs, run `python test_all.py` and ensure `evaluation_log.csv` only includes intended scenarios.
- For targeted debugging, add short scripts in `test/` or extend `HybridAutomata_test.py` (e.g., `test_duffing_reset_logic`).
- Capture deterministic seeds in configs so regenerated traces remain comparable.
- Adding automata? Include at least one reproducible trace set in `data/` and note expected coverage.

## Commit & Pull Request Guidelines
- Commits: imperative, descriptive subjects (≤72 chars), with details as needed. Example: `Refactor get_init_state_ha logic`.
- Link issues/experiments with `Refs:`. Attach plots or CSV diffs when behavior changes.
- PRs should state motivation, validation commands run, impacted automata, and any new data files so reviewers can reproduce quickly.

## Security & Configuration Tips
- Do not commit secrets or large binaries. Use relative paths; keep outputs under `result/`. Pin seeds in configs for reproducibility.
