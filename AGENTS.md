# Repository Guidelines

## Project Structure & Module Organization
Code lives in `src/`, grouped by function: learning logic (`HybridAutomata.py`, `DE_System.py`, `GuardLearning.py`), preprocessing (`CurveSlice.py`, `ChangePoints.py`), and orchestration helpers (`BuildSystem.py`, `utils.py`). Root scripts drive workflows: `main.py` runs a single automaton, `test_all.py` evaluates every scenario, and `CreatData.py` produces synthetic traces. Automaton definitions sit in `automata/`, while generated traces and logs land in `data*/`. Keep bulky experiment output inside clearly named subfolders and extend `.gitignore` when needed.

## Build, Test, and Development Commands
Set up Python 3.9 tooling before contributing:
`python -m venv .venv && source .venv/bin/activate`
`pip install numpy scikit-learn networkx matplotlib`
Execute `python main.py` to replay the default pipeline or edit the JSON path near the bottom of `main.py` to target a different automaton. Run `python test_all.py` for a full sweep that writes `evaluation_log.csv`. During iteration, call `python -c "from main import main; main('automata/non_linear/duffing.json')"` to avoid modifying the entry script.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and trailing-newline terminated files. Maintain existing PascalCase module names and class-per-file structure. Adopt descriptive identifiers (`mode_map`, `feature_list`), keep functions focused, and reuse helper utilities from `utils.py`. Complement algebra-heavy blocks with a single clarifying comment instead of restating each line.

## Testing Guidelines
There is no dedicated unit suite; rely on scenario coverage. Exercise new features against at least two automata (one with resets, one without) and confirm metrics in `evaluation_log.csv` stay within expected ranges. Place lightweight regression checks beside the code you touch (e.g., `src/test_guard_learning.py`) and use deterministic seeds for any randomized sampling so results can be reproduced.

## Commit & Pull Request Guidelines
Commits follow short, capitalized imperatives (`Add guard reset support`) and group related edits only. In pull requests, summarize intent, list verification commands, attach relevant logs or plots, and mention any new dependencies or flags. Link issues when available and highlight backwards-incompatible changes to automaton formats or data layout.

## Data & Configuration Tips
Automaton JSON files require consistent variable names (`x1[1]`, `u`) and orders; validate edits with a quick `python main.py` run before publishing. Keep personal datasets and large outputs untracked, and document new configuration knobs in the README or module docstrings for future maintainers.
