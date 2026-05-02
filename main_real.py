import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.BuildSystem import build_system, get_init_state
from src.Clustering import clustering
from src.CurveSlice import Slice, slice_curve
from src.DEConfig import FeatureExtractor
from src.Evaluation import Evaluation
from src.GuardLearning import guard_learning
from src.HybridAutomata import HybridAutomata
from src.utils import get_mode_list, get_ture_chp, max_bipartite_matching


REAL_DATASETS = (
    "ControlledCells_g",
    "F1tenthCar_g",
    "Handwriting_g",
)

DEFAULT_CONFIG = {
    "dt": 0.01,
    "total_time": None,
    "order": 3,
    "window_size": 10,
    "clustering_method": "fit",
    "minus": False,
    "need_bias": True,
    "other_items": "",
    "kernel": "linear",
    "svm_c": 1e6,
    "class_weight": 1.0,
    "need_reset": False,
    "self_loop": False,
}


def file_sort_key(path: Path):
    match = re.search(r"(\d+)", path.name)
    number = int(match.group(1)) if match else -1
    return number, path.name


def load_npz_files(data_dir: Path):
    files = sorted(data_dir.glob("*.npz"), key=file_sort_key)
    records = []

    for file_path in files:
        with np.load(file_path, allow_pickle=False) as npz_file:
            missing = {"state", "input", "mode"} - set(npz_file.files)
            if missing:
                raise KeyError(f"{file_path} is missing keys: {sorted(missing)}")

            state = np.asarray(npz_file["state"])
            input_data = np.asarray(npz_file["input"])
            mode = np.asarray(npz_file["mode"])

            if state.ndim != 2:
                raise ValueError(f"{file_path}: state must be 2D, got shape {state.shape}")
            if input_data.ndim != 2:
                raise ValueError(f"{file_path}: input must be 2D, got shape {input_data.shape}")
            if mode.ndim != 1:
                raise ValueError(f"{file_path}: mode must be 1D, got shape {mode.shape}")
            if state.shape[1] != mode.shape[0] or input_data.shape[1] != mode.shape[0]:
                raise ValueError(
                    f"{file_path}: time dimension mismatch among state/input/mode: "
                    f"{state.shape}, {input_data.shape}, {mode.shape}"
                )

            if "change_points" in npz_file.files and len(npz_file["change_points"]) > 0:
                change_points = np.asarray(npz_file["change_points"])
            else:
                change_points = np.asarray(get_ture_chp(mode))

            records.append(
                {
                    "path": file_path,
                    "state": state,
                    "input": input_data,
                    "mode": mode,
                    "change_points": change_points,
                }
            )

    return records


def split_records(records, train_count):
    if len(records) <= train_count:
        raise ValueError(
            f"Need more than {train_count} .npz files to create train/test split, got {len(records)}"
        )
    return records[:train_count], records[train_count:]


def unique_mode_count(records):
    if not records:
        return 0
    return len(set(np.concatenate([record["mode"] for record in records]).tolist()))


def run_with_change_points(data_list, input_data, change_points_list, config, evaluation: Evaluation):
    get_feature = FeatureExtractor(
        data_list[0].shape[0],
        input_data[0].shape[0],
        order=config["order"],
        dt=config["dt"],
        minus=config["minus"],
        need_bias=config["need_bias"],
        other_items=config["other_items"],
    )
    Slice.clear()
    slice_data = []
    chp_list = []
    for data, input_val, change_points in zip(data_list, input_data, change_points_list):
        change_points = list(change_points)
        chp_list.append(change_points)
        print("GT ChP:\t", change_points)
        slice_curve(slice_data, data, input_val, change_points, get_feature)

    evaluation.submit(chp=chp_list)
    evaluation.recording_time("change_points")
    Slice.Method = config["clustering_method"]
    Slice.fit_threshold(slice_data)
    clustering(slice_data, config["self_loop"])
    evaluation.recording_time("clustering")
    adj = guard_learning(slice_data, get_feature, config)
    evaluation.recording_time("guard_learning")
    sys = build_system(slice_data, adj, get_feature)
    evaluation.stop("total")
    evaluation.submit(slice_data=slice_data)
    if not sys.mode_list:
        raise ValueError("No valid modes were learned from the training data.")
    return sys, slice_data


def plot_first_test_trace(
    dataset_name,
    state_data,
    fit_data,
    gt_mode,
    fit_mode,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for var_idx in range(state_data.shape[0]):
        plt.figure()
        plt.plot(np.arange(state_data.shape[1]), state_data[var_idx], color="c", label="ground truth")
        plt.plot(np.arange(fit_data.shape[1]), fit_data[var_idx], color="r", label="fit")
        plt.legend()
        plt.savefig(output_dir / f"{dataset_name}_state_{var_idx}.png")
        plt.close()

    plt.figure()
    plt.plot(np.arange(len(gt_mode)), gt_mode, color="c", label="ground truth")
    plt.plot(np.arange(len(fit_mode)), fit_mode, color="r", label="fit")
    plt.legend()
    plt.savefig(output_dir / f"{dataset_name}_mode.png")
    plt.close()


def evaluate_real_dataset(
    data_dir: Path,
    train_count=3,
    config=None,
    need_plot=False,
    plot_dir: Optional[Path] = None,
):
    config = DEFAULT_CONFIG.copy() if config is None else {**DEFAULT_CONFIG, **config}
    records = load_npz_files(data_dir)
    train_records, test_records = split_records(records, train_count)

    evaluation = Evaluation(str(data_dir))
    evaluation.submit(gt_mode_num=unique_mode_count(records))
    HybridAutomata.LoopWarning = not config["self_loop"]

    train_data = [record["state"] for record in train_records]
    train_input = [record["input"] for record in train_records]
    train_mode = [record["mode"] for record in train_records]
    train_chp = [record["change_points"] for record in train_records]

    test_data = [record["state"] for record in test_records]
    test_input = [record["input"] for record in test_records]
    test_mode = [record["mode"] for record in test_records]

    print(f"\n== {data_dir.name} ==")
    print(f"train files: {[record['path'].name for record in train_records]}")
    print(f"test files: {[record['path'].name for record in test_records]}")
    print("Be running!")

    evaluation.submit(gt_chp=train_chp)
    evaluation.submit(train_mode_list=train_mode)
    evaluation.start()
    sys, slice_data = run_with_change_points(train_data, train_input, train_chp, config, evaluation)
    print(f"mode number: {len(sys.mode_list)}")
    print("Start simulation")

    all_fit_mode, all_gt_mode = get_mode_list(slice_data, train_mode)
    mode_map, mode_map_inv = max_bipartite_matching(all_fit_mode, all_gt_mode)

    init_state_test = get_init_state(test_data, mode_map, test_mode, config["order"])
    fit_data_list = []
    mode_data_list = []

    for idx, (data, mode, input_data, init_state) in enumerate(
        zip(test_data, test_mode, test_input, init_state_test)
    ):
        fit_data = [data[:, i] for i in range(config["order"])]
        mode_data = list(mode[: config["order"]])
        sys.reset(init_state, input_data[:, : config["order"]])

        for i in range(config["order"], data.shape[1]):
            state, fit_mode, _ = sys.next(input_data[:, i])
            fit_data.append(state)
            mode_data.append(mode_map_inv.get(fit_mode, -fit_mode))

        fit_data = np.transpose(np.asarray(fit_data))
        evaluation.submit(mode_num=len(sys.mode_list))
        fit_data_list.append(fit_data)
        mode_data_list.append(mode_data)

        if need_plot and idx == 0:
            if plot_dir is None:
                plot_dir = Path("result") / "real_plots"
            plot_first_test_trace(data_dir.name, data, fit_data, mode, mode_data, plot_dir)

    evaluation.submit(
        fit_mode=mode_data_list,
        fit_data=fit_data_list,
        gt_mode=test_mode,
        gt_data=test_data,
        dt=config["dt"],
    )
    result = evaluation.calc()
    result["dataset"] = data_dir.name
    result["train_files"] = [str(record["path"]) for record in train_records]
    result["test_files"] = [str(record["path"]) for record in test_records]
    result["status"] = "ok"
    return result


def write_markdown_summary(results, output_path: Path):
    lines = [
        "# DAINARX Real Dataset Results",
        "",
        "| Dataset | max_diff | mean_diff |",
        "| --- | ---: | ---: |",
    ]

    for result in results:
        if "max_diff" not in result or "mean_diff" not in result:
            continue
        lines.append(
            f"| {result['dataset']} | {result['max_diff']} | {result['mean_diff']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the main pipeline on real .npz datasets with the first files used for training."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_real"))
    parser.add_argument("--train-count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("result") / "real_eval_log.json")
    parser.add_argument("--markdown-output", type=Path, default=Path("result") / "dainarx_real.md")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    results = []

    for dataset in REAL_DATASETS:
        data_dir = args.data_root / dataset
        try:
            result = evaluate_real_dataset(
                data_dir,
                train_count=args.train_count,
                config=config,
                need_plot=args.plot,
                plot_dir=args.output.parent / "real_plots",
            )
        except Exception as exc:
            result = {
                "dataset": dataset,
                "name": str(data_dir),
                "status": "error",
                "error": str(exc),
            }
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(results, file, indent=2)
    write_markdown_summary(results, args.markdown_output)

    print("\nEvaluation log:")
    for result in results:
        print(json.dumps(result, indent=2))
    print(f"\nSaved to: {args.output}")
    print(f"Saved markdown to: {args.markdown_output}")


if __name__ == "__main__":
    result_path = "result"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    main()
