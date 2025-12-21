# Evaluation Results Comparison

**Automaton:** `./automata/non_linear/duffing.json`

## Performance Metrics

| Metric | order3 | x4 | baseline | Best |
|--------|--------|--------|--------|------|
| Clustering Error | 0 | 15 | 0 | **order3** |
| TC (Test Error) | 0.001000 | 1.013000 | 0.001000 | **order3** |
| Max Difference | 0.000795 | 2.049988 | 0.000280 | **baseline** |
| Mean Difference | 3.20e-05 | 0.363789 | 2.78e-05 | **baseline** |
| Train TC | 0.00e+00 | 0.00e+00 | 0.00e+00 | **order3** |

## Timing Comparison (seconds)

| Stage | order3 | x4 | baseline | Fastest |
|-------|--------|--------|--------|---------|
| change_points | 4.726 | 4.532 | 4.560 | **x4** |
| clustering | 2.771 | 7.494 | 2.403 | **baseline** |
| guard_learning | 0.052 | 0.109 | 0.050 | **baseline** |
| total | 7.956 | 12.401 | 7.369 | **baseline** |

## Summary

**Best scores count (higher is better):**

- **baseline**: 5 wins
- **order3**: 3 wins
- **x4**: 1 wins

## Detailed Analysis

### x4 vs order3

- **Clustering Error**: +15
- **Mean Diff**: +1135083.5%
- **Max Diff**: +257683.3%
- **Total Time**: +55.9%

### baseline vs order3

- **Mean Diff**: -13.3%
- **Max Diff**: -64.7%
- **Total Time**: -7.4%
