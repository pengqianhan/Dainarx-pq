import argparse
import json
import os
import re
from typing import Any, Dict, Optional

import numpy as np

from CreatData import plot_fun
from src.HybridAutomata import HybridAutomata

def extract_number_from_filename(filename):
    """从文件名中提取数字部分，用于获取初始索引"""
    basename = os.path.basename(filename)
    # 使用正则表达式查找文件名中的数字
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0  # 默认值，如果没有找到数字

def get_init_state_ha(data, mode_map, mode, bias):

    init_state = {'mode': mode_map[mode[bias - 1]]}
    for i in range(data.shape[0]):
        # print("data[i, (bias - 1)::-1]: ", data[i, (bias - 1)::-1]) # reverse slice, from bias-1 to start
        init_state['x' + str(i)] = data[i, (bias - 1)::-1]
    return init_state

class HybridAutomataSimulation:
    def __init__(self, args: argparse.Namespace) -> None:
        self.json_path = args.json
        self.init_index = extract_number_from_filename(args.npz_file)
        self.npz_file = args.npz_file
        self._result: Optional[Dict[str, Any]] = None

    def simulate(self) -> Dict[str, Any]:
        if self._result is not None:
            return self._result

        with open(self.json_path, "r") as f:
            automaton_data = json.load(f)


        config = automaton_data.get("config", {})
        dt = config.get("dt", 0.001)
        total_time = config.get("total_time", 10.0)

        HybridAutomata.LoopWarning = not config.get("self_loop", False)

        sys = HybridAutomata.from_json(automaton_data["automaton"])
        npz_data = np.load(self.npz_file)
        state_data = npz_data["state"]
        mode_data = npz_data["mode"]
        input_data = npz_data["input"]
        # print("state_data.shape: ", state_data.shape)
        # print("mode_data.shape: ", mode_data.shape)
        # print("input_data.shape: ", input_data.shape)
        # change_points = npz_data["change_points"]
        mode_map = {np.int64(1): np.int64(1), np.int64(2): np.int64(2), np.int64(3): np.int64(3)}
        init_state = get_init_state_ha(state_data, mode_map, mode_data, config['order'])
        
        print("init_state: ", init_state)
        print("input_data[:, :config['order']].shape: ", input_data[:, :config['order']].shape)
        print("input_data.shape: ", input_data.shape)
        sys.reset(init_state)

        mode_pred = [init_state['mode'] for _ in range(config['order'])]
        print("mode_pred: ", mode_pred)
        change_points = [0]
        fit_data = [state_data[:, i] for i in range(config['order'])]## 取前0:order-1个时间步作为初始值

        for i in range(config['order'], state_data.shape[1]):## 从order个时间步开始模拟
            state, mode, switched = sys.next(input_data[:, i]) ## 基于当前输入预测下一状态
            fit_data.append(state)
            mode_pred.append(mode)
        # fit_data = np.transpose(np.array(fit_data))
        fit_data = np.array(fit_data)
        mode_pred = np.array(mode_pred)
        # mode_pred = np.concatenate([[init_state['mode']].repeat(len(init_state['x0'])), mode_pred])
        print("fit_data.shape: ", fit_data.shape)
        print("mode_pred.shape: ", mode_pred.shape)
        print("state_data.shape: ", state_data.shape)
        assert fit_data.shape ==state_data.shape , "fit_data.shape: {fit_data.shape} != state_data.shape: {state_data.shape}"

        self._result = {
            "state": fit_data,
            "mode": mode_pred,
            "input": input_data,
            "change_points": np.array(change_points, dtype=np.int32),
            "dt": dt,
            "total_time": total_time,
            "json_path": self.json_path,
            "init_index": self.init_index,
        }
        return self._result

    def save(self, out_dir: str) -> str:
        result = self.simulate()
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(result["json_path"]))[0]
        filename = f"simulation_{base_name}_init{result['init_index']}.npz"
        path = os.path.join(out_dir, filename)
        np.savez(
            path,
            state=result["state"],
            mode=result["mode"],
            input=result["input"],
            change_points=result["change_points"],
            dt=result["dt"],
            total_time=result["total_time"],
        )
        return path

    def plot(self, out_dir: str, show: bool) -> str:
        result = self.simulate()
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(result["json_path"]))[0]
        figure_name = f"simulation_{base_name}_init{result['init_index']}.png"
        figure_path = os.path.join(out_dir, figure_name)
        plot_fun(
            result["state"],
            result["input"],
            result["dt"],
            system_name=base_name,
            sample_index=result["init_index"] + 1,
            save_path=figure_path,
            show=show,
        )
        return figure_path


def main():
    parser = argparse.ArgumentParser(description="Simulate a hybrid automaton using the first initial value.")
    parser.add_argument(
        "--json",
        default="automata/non_linear/duffing_simulate.json",
        help="Path to the hybrid automaton description (default: automata/non_linear/duffing.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="result",
        help="Directory where plots and simulation data will be stored (default: result).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating a plot for the simulated trajectory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the simulated trajectory to an NPZ file.",
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        default="data_duffing/test_data0.npz",
        help="the npz file path (default: data_duffing\test_data0.npz).",
    )
    args = parser.parse_args()

    simulation = HybridAutomataSimulation(args)
    sim_result = simulation.simulate()
    unique_modes = np.unique(sim_result["mode"])
    print(
        f"Simulated {sim_result['total_time']} seconds with dt={sim_result['dt']} "
        f"using init_state[{simulation.init_index}] from {args.json}."
    )
    print(f"Visited modes: {unique_modes.tolist()}")
    print(f"Detected change points (steps): {sim_result['change_points'].tolist()}")

    if args.save:
        data_path = simulation.save(args.out_dir)
        print(f"Saved simulation arrays to {data_path}")

    if not args.no_plot:
        figure_path = simulation.plot(args.out_dir, show=args.show)
        print(f"Saved plot to {figure_path}")


if __name__ == "__main__":
    main()
