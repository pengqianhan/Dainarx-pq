"""
Result Analysis Script
Compare evaluation results from different experiment configurations.
"""

import json
import os


def load_eval_log(filepath: str) -> dict:
    """Load evaluation log from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_time_dict(time_list: list) -> dict:
    """Convert time list to dictionary."""
    return {item[0]: item[1] for item in time_list}


def format_value(value, precision=6):
    """Format numeric value for display."""
    if isinstance(value, float):
        if abs(value) < 0.0001:
            return f"{value:.2e}"
        return f"{value:.{precision}f}"
    return str(value)


def compare_results(result_dirs: dict, output_file: str = "result/comparison_report.md"):
    """
    Compare results from multiple experiment directories.

    Args:
        result_dirs: Dictionary mapping display name to directory path
        output_file: Output markdown file path
    """
    # Load all results
    results = {}
    for name, dir_path in result_dirs.items():
        filepath = os.path.join(dir_path, "eval_log.json")
        if os.path.exists(filepath):
            results[name] = load_eval_log(filepath)
        else:
            print(f"Warning: {filepath} not found")

    if not results:
        print("No results found!")
        return

    # Build markdown content
    md_lines = []

    # Header
    md_lines.append("# Evaluation Results Comparison\n")

    # Automaton name
    first_result = list(results.values())[0]
    md_lines.append(f"**Automaton:** `{first_result.get('name', 'Unknown')}`\n")

    # Prepare comparison table
    config_names = list(results.keys())

    # Key metrics to compare
    metrics = [
        ("clustering_error", "Clustering Error", "lower"),
        ("tc", "TC (Test Error)", "lower"),
        ("max_diff", "Max Difference", "lower"),
        ("mean_diff", "Mean Difference", "lower"),
        ("train_tc", "Train TC", "lower"),
    ]

    # Performance Metrics Table
    md_lines.append("## Performance Metrics\n")

    # Table header
    header = "| Metric |"
    separator = "|--------|"
    for name in config_names:
        header += f" {name} |"
        separator += "--------|"
    header += " Best |"
    separator += "------|"
    md_lines.append(header)
    md_lines.append(separator)

    # Table rows
    for metric_key, metric_name, best_dir in metrics:
        row = f"| {metric_name} |"
        values = []
        for name in config_names:
            val = results[name].get(metric_key, "N/A")
            values.append(val)
            row += f" {format_value(val)} |"

        # Find best value
        numeric_values = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
        if numeric_values:
            if best_dir == "lower":
                best_idx = min(numeric_values, key=lambda x: x[1])[0]
            else:
                best_idx = max(numeric_values, key=lambda x: x[1])[0]
            best_name = config_names[best_idx]
            row += f" **{best_name}** |"
        else:
            row += " - |"

        md_lines.append(row)

    md_lines.append("")

    # Timing Comparison Table
    md_lines.append("## Timing Comparison (seconds)\n")

    time_metrics = ["change_points", "clustering", "guard_learning", "total"]

    # Table header
    header = "| Stage |"
    separator = "|-------|"
    for name in config_names:
        header += f" {name} |"
        separator += "--------|"
    header += " Fastest |"
    separator += "---------|"
    md_lines.append(header)
    md_lines.append(separator)

    for time_key in time_metrics:
        row = f"| {time_key} |"
        values = []
        for name in config_names:
            time_dict = extract_time_dict(results[name].get("time", []))
            val = time_dict.get(time_key, "N/A")
            values.append(val)
            row += f" {format_value(val, 3)} |"

        # Find fastest
        numeric_values = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
        if numeric_values:
            best_idx = min(numeric_values, key=lambda x: x[1])[0]
            best_name = config_names[best_idx]
            row += f" **{best_name}** |"
        else:
            row += " - |"

        md_lines.append(row)

    md_lines.append("")

    # Summary
    md_lines.append("## Summary\n")

    # Calculate overall scores (lower is better for all our metrics)
    scores = {name: 0 for name in config_names}

    for metric_key, _, _ in metrics:
        values = [(name, results[name].get(metric_key, float('inf')))
                  for name in config_names]
        numeric_values = [(n, v) for n, v in values if isinstance(v, (int, float))]
        if numeric_values:
            best = min(numeric_values, key=lambda x: x[1])[0]
            scores[best] += 1

    for time_key in time_metrics:
        values = []
        for name in config_names:
            time_dict = extract_time_dict(results[name].get("time", []))
            val = time_dict.get(time_key, float('inf'))
            values.append((name, val))
        numeric_values = [(n, v) for n, v in values if isinstance(v, (int, float))]
        if numeric_values:
            best = min(numeric_values, key=lambda x: x[1])[0]
            scores[best] += 1

    md_lines.append("**Best scores count (higher is better):**\n")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        md_lines.append(f"- **{name}**: {score} wins")

    md_lines.append("")

    # Detailed analysis
    md_lines.append("## Detailed Analysis\n")

    # Check for significant differences
    baseline = config_names[0]  # Use first config as baseline

    for name in config_names[1:]:
        md_lines.append(f"### {name} vs {baseline}\n")

        comparisons = []

        # Clustering error
        base_ce = results[baseline].get("clustering_error", 0)
        comp_ce = results[name].get("clustering_error", 0)
        if base_ce != comp_ce:
            diff = comp_ce - base_ce
            comparisons.append(f"- **Clustering Error**: {'+' if diff > 0 else ''}{diff}")

        # Mean diff
        base_md = results[baseline].get("mean_diff", 0)
        comp_md = results[name].get("mean_diff", 0)
        if base_md != 0:
            pct_change = ((comp_md - base_md) / base_md) * 100
            comparisons.append(f"- **Mean Diff**: {'+' if pct_change > 0 else ''}{pct_change:.1f}%")

        # Max diff
        base_maxd = results[baseline].get("max_diff", 0)
        comp_maxd = results[name].get("max_diff", 0)
        if base_maxd != 0:
            pct_change = ((comp_maxd - base_maxd) / base_maxd) * 100
            comparisons.append(f"- **Max Diff**: {'+' if pct_change > 0 else ''}{pct_change:.1f}%")

        # Total time
        base_time = extract_time_dict(results[baseline].get("time", [])).get("total", 0)
        comp_time = extract_time_dict(results[name].get("time", [])).get("total", 0)
        if base_time != 0:
            pct_change = ((comp_time - base_time) / base_time) * 100
            comparisons.append(f"- **Total Time**: {'+' if pct_change > 0 else ''}{pct_change:.1f}%")

        if comparisons:
            md_lines.extend(comparisons)
        else:
            md_lines.append("No significant differences.")

        md_lines.append("")

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Report saved to: {output_file}")


def main():
    """Main function to run the analysis."""
    # Define result directories to compare
    result_dirs = {
        "order3": "result_oder3",
        "x4": "result_x4",
        "baseline": "result0",
    }

    # Run comparison
    compare_results(result_dirs)


if __name__ == "__main__":
    main()
