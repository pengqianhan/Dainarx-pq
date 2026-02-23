#!/bin/bash
# run_all.sh - Process all JSON files under automata/ and summarize results into a markdown file.
#
# Usage:
#   bash run_all.sh              # run all benchmarks
#   bash run_all.sh linear       # run only the "linear" category
#   bash run_all.sh non_linear   # run only the "non_linear" category

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$PROJECT_DIR/result/batch_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$RESULT_DIR/summary.md"
CATEGORY_FILTER="${1:-}"   # optional: only run a specific category

mkdir -p "$RESULT_DIR"

# Collect all JSON files
mapfile -t JSON_FILES < <(find "$PROJECT_DIR/automata" -name "*.json" -type f | sort)

TOTAL=${#JSON_FILES[@]}
echo "Found $TOTAL benchmark JSON files."
if [[ -n "$CATEGORY_FILTER" ]]; then
    echo "Filtering by category: $CATEGORY_FILTER"
fi

SUCCESS=0
FAIL=0

for idx in "${!JSON_FILES[@]}"; do
    json_file="${JSON_FILES[$idx]}"
    rel_path="${json_file#$PROJECT_DIR/}"
    category=$(echo "$rel_path" | cut -d'/' -f2)
    benchmark=$(basename "$json_file" .json)

    # Apply category filter
    if [[ -n "$CATEGORY_FILTER" && "$category" != "$CATEGORY_FILTER" ]]; then
        continue
    fi

    echo ""
    echo "[$((idx+1))/$TOTAL] Processing: $rel_path"
    echo "----------------------------------------------"

    # Per-benchmark result directory
    bench_dir="$RESULT_DIR/${category}"
    mkdir -p "$bench_dir"
    result_json="$bench_dir/${benchmark}.json"

    # Run the benchmark via a small Python wrapper
    if python3 - "$PROJECT_DIR" "$rel_path" "$result_json" << 'PYEOF'
import sys, json, os

project_dir = sys.argv[1]
rel_path    = sys.argv[2]
out_path    = sys.argv[3]

os.chdir(project_dir)
sys.path.insert(0, project_dir)

# Suppress matplotlib GUI
import matplotlib
matplotlib.use("Agg")

from main import main

try:
    eval_log = main(rel_path, need_plot=False)
    eval_log["status"] = "ok"
except Exception as e:
    eval_log = {"status": "error", "error": str(e)}

with open(out_path, "w") as f:
    json.dump(eval_log, f, indent=2)

print(json.dumps(eval_log, indent=2))
PYEOF
    then
        echo "  -> OK, result saved to $result_json"
        SUCCESS=$((SUCCESS + 1))
    else
        # If Python itself crashed, still write an error marker
        echo "{\"status\": \"crash\"}" > "$result_json"
        echo "  -> FAILED"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=============================================="
echo "Benchmark runs finished: $SUCCESS succeeded, $FAIL failed"
echo "Generating summary markdown ..."
echo "=============================================="

# Generate the summary markdown from all per-benchmark JSON files
python3 - "$RESULT_DIR" "$SUMMARY_FILE" << 'PYEOF'
import sys, json, os, glob

result_dir  = sys.argv[1]
summary_file = sys.argv[2]

# Collect results grouped by category
results = {}  # category -> list of dicts
for jf in sorted(glob.glob(os.path.join(result_dir, "**", "*.json"), recursive=True)):
    category = os.path.basename(os.path.dirname(jf))
    benchmark = os.path.splitext(os.path.basename(jf))[0]
    with open(jf) as f:
        data = json.load(f)
    data["_category"] = category
    data["_benchmark"] = benchmark
    results.setdefault(category, []).append(data)

with open(summary_file, "w") as md:
    md.write("# Batch Evaluation Summary\n\n")

    # Overall table
    md.write("## Results\n\n")
    md.write("| Category | Benchmark | Mode Error | Train TC | Test TC | Max Diff | Mean Diff | Total Time (s) | Status |\n")
    md.write("|----------|-----------|------------|----------|---------|----------|-----------|-----------------|--------|\n")

    for cat in sorted(results.keys()):
        for r in results[cat]:
            name = r["_benchmark"]
            status = r.get("status", "unknown")
            if status == "ok":
                total_time = sum(t[1] for t in r.get("time", []))
                md.write(f"| {cat} | {name} "
                         f"| {r.get('clustering_error', 'N/A')} "
                         f"| {r.get('train_tc', 0):.4f} "
                         f"| {r.get('tc', 0):.4f} "
                         f"| {r.get('max_diff', 0):.4f} "
                         f"| {r.get('mean_diff', 0):.4f} "
                         f"| {total_time:.2f} "
                         f"| ok |\n")
            else:
                err = r.get("error", status)
                md.write(f"| {cat} | {name} | - | - | - | - | - | - | {err} |\n")

    # Per-category summary
    md.write("\n## Category Summary\n\n")
    md.write("| Category | Benchmarks | Succeeded | Avg Mode Error | Avg Train TC | Avg Test TC | Avg Max Diff | Avg Mean Diff | Avg Time (s) |\n")
    md.write("|----------|------------|-----------|----------------|--------------|-------------|--------------|---------------|---------------|\n")

    for cat in sorted(results.keys()):
        ok_results = [r for r in results[cat] if r.get("status") == "ok"]
        total = len(results[cat])
        n = len(ok_results)
        if n == 0:
            md.write(f"| {cat} | {total} | 0 | - | - | - | - | - | - |\n")
            continue
        avg_ce   = sum(r.get("clustering_error", 0) for r in ok_results) / n
        avg_ttc  = sum(r.get("train_tc", 0) for r in ok_results) / n
        avg_tc   = sum(r.get("tc", 0) for r in ok_results) / n
        avg_maxd = sum(r.get("max_diff", 0) for r in ok_results) / n
        avg_md   = sum(r.get("mean_diff", 0) for r in ok_results) / n
        avg_time = sum(sum(t[1] for t in r.get("time", [])) for r in ok_results) / n
        md.write(f"| {cat} | {total} | {n} "
                 f"| {avg_ce:.2f} | {avg_ttc:.4f} | {avg_tc:.4f} "
                 f"| {avg_maxd:.4f} | {avg_md:.4f} | {avg_time:.2f} |\n")

    # Timing breakdown
    md.write("\n## Timing Breakdown\n\n")
    md.write("| Category | Benchmark | Change Points (s) | Clustering (s) | Guard Learning (s) | Total (s) |\n")
    md.write("|----------|-----------|--------------------|-----------------|--------------------|----------|\n")

    for cat in sorted(results.keys()):
        for r in results[cat]:
            if r.get("status") != "ok":
                continue
            name = r["_benchmark"]
            time_dict = {t[0]: t[1] for t in r.get("time", [])}
            cp_t = time_dict.get("change_points", 0)
            cl_t = time_dict.get("clustering", 0)
            gl_t = time_dict.get("guard_learning", 0)
            total_t = time_dict.get("total", sum(time_dict.values()))
            md.write(f"| {cat} | {name} | {cp_t:.2f} | {cl_t:.2f} | {gl_t:.2f} | {total_t:.2f} |\n")

print(f"Summary written to: {summary_file}")
PYEOF

echo ""
echo "Done! Results in: $RESULT_DIR"
echo "Summary:          $SUMMARY_FILE"
