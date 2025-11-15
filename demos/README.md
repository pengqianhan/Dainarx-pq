# Dainarx LLM Auto-Configuration Demo

This directory contains demonstrations of using LLM + Genetic Algorithm to automatically configure Dainarx parameters.

## Overview

The automatic configuration system addresses the key limitation of Dainarx: the need for manual parameter tuning. It uses:

1. **Data Analyzer**: Extracts statistical, spectral, and dynamical features from time series
2. **LLM Analyzer**: Uses Google Gemini to analyze features and suggest parameter ranges
3. **Genetic Algorithm**: Searches the parameter space for optimal configuration
4. **Fitness Evaluator**: Evaluates configurations using multiple quality metrics

## Setup

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r llm_config_optimizer/requirements.txt

# Or install individually:
pip install numpy scipy google-generativeai python-dotenv matplotlib
```

### 2. Configure API Key

Get a Google Gemini API key from: https://makersuite.google.com/app/apikey

Edit `.env` file:
```bash
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=models/gemini-flash-lite-latest
```

## Running the Duffing Demo

### Quick Test (Lite Model, Small Population)

```bash
python demos/duffing_auto_config.py --test --pop-size 5 --generations 3
```

This runs a quick test with:
- Gemini Flash Lite model (faster, cheaper)
- Population size: 5
- Generations: 3
- Estimated time: 2-5 minutes

### Full Production Run

```bash
python demos/duffing_auto_config.py --full-run --pop-size 20 --generations 10
```

This runs a full optimization with:
- Gemini Flash model (more accurate)
- Population size: 20
- Generations: 10
- Estimated time: 15-30 minutes

### Skip LLM Analysis (Pure GA)

```bash
python demos/duffing_auto_config.py --no-llm --pop-size 10 --generations 5
```

This skips LLM analysis and uses genetic algorithm with random initialization.

## Output

The demo produces:

1. **Console Output**: Shows progress and results
2. **JSON Results**: `result/duffing_auto_config_result.json`
   - Best configuration found
   - Fitness score
   - LLM suggestions
   - GA optimization history
3. **Optimization Plot**: `result/duffing_ga_optimization.png`
   - Shows fitness evolution over generations

## Example Output

```
=== STEP 1: Data Feature Extraction ===
Dimension: 1
Dominant frequencies: [1.23 Hz]
Likely nonlinear: True
Suggested terms: ['x[?]**3']

=== STEP 2: LLM Configuration Analysis ===
System Type: nonlinear_oscillator
Confidence: 0.92

Parameter Recommendations:
  order: [2, 3]
  other_items: ["x[?]**3"]
  window_size: [10, 12]
  kernel: rbf
  need_reset: True

=== STEP 3: Genetic Algorithm Optimization ===
Generation 1/10:
  Best fitness: 0.6532
  Best config: order=2, window_size=10, other_items=x[?]**3, ...

Generation 10/10:
  Best fitness: 0.8947
  Best config: order=2, window_size=10, other_items=x[?]**3, ...

=== Best Configuration ===
  order: 2
  window_size: 10
  other_items: x[?]**3
  kernel: rbf
  svm_c: 1000000.0
  class_weight: 30.0
  self_loop: False
  need_reset: True

=== Comparison with Manual Configuration ===
  ✓ order: manual=2, auto=2
  ✓ other_items: manual=x[?]**3, auto=x[?]**3
  ✓ need_reset: manual=True, auto=True
```

## Understanding the Results

### Fitness Score

Fitness ranges from 0 to 1:
- **> 0.8**: Excellent configuration
- **0.6 - 0.8**: Good configuration
- **< 0.6**: Poor configuration

Fitness considers:
- Mode clustering accuracy (35%)
- Change point detection F1 (30%)
- Fitting error (20%)
- Model balance and complexity (15%)

### Configuration Parameters

**Critical Parameters:**
- `order`: NARX model order (2-5)
- `other_items`: Nonlinear terms
- `window_size`: Change point detection window
- `kernel`: SVM kernel function

**Tuning Parameters:**
- `svm_c`: SVM regularization
- `class_weight`: Class imbalance compensation
- `self_loop`: Allow same mode consecutively
- `need_reset`: Learn state jump functions

## Extending to Other Systems

To apply to a different system:

1. Create/modify automaton JSON in `automata/`
2. Run demo with the new path:
   ```python
   # In duffing_auto_config.py, modify:
   json_path = "./automata/your_system.json"
   ```
3. Adjust GA parameters if needed

## Troubleshooting

### API Key Error
```
ValueError: GEMINI_API_KEY not set
```
**Solution**: Add your API key to `.env` file

### Import Error
```
ImportError: google-generativeai not installed
```
**Solution**: `pip install google-generativeai`

### Low Fitness Scores
If optimization converges to low fitness (<0.5):
- Increase population size: `--pop-size 30`
- Increase generations: `--generations 15`
- Check if LLM suggestions are reasonable
- Verify data quality

### LLM Analysis Fails
The system will fall back to default ranges if LLM fails.
You can also use `--no-llm` flag to skip LLM entirely.

## Performance Tips

1. **Start small**: Test with `--pop-size 5 --generations 3` first
2. **Use caching**: Data is cached, regeneration is automatic
3. **Parallel evaluation**: Future work could parallelize fitness evaluation
4. **Early stopping**: GA stops early if no improvement for N generations

## Citation

If you use this auto-configuration system, please cite:

```
Dainarx LLM Auto-Configuration
https://github.com/your-repo/Dainarx-pq
```

## License

Same as parent Dainarx project.
