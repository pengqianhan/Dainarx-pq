import argparse
import json
import os
from typing import Any, Dict

import numpy as np

from CreatData import plot_fun
from src.HybridAutomata import HybridAutomata


def simulate_hybrid_automata(json_path: str, init_index: int = 0) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        automaton_data = json.load(f)

    init_states = automaton_data.get("init_state", [])
    if not init_states:
        raise ValueError("init_state is empty in the provided automaton file.")
    if init_index < 0 or init_index >= len(init_states):
        raise IndexError(f"init_index {init_index} is out of range (available: 0-{len(init_states) - 1}).")

    config = automaton_data.get("config", {})
    dt = config.get("dt", 0.01)
    total_time = config.get("total_time", 10.0)
    steps = int(round(total_time / dt))
    if steps <= 0:
        raise ValueError("The simulation requires total_time / dt to produce at least one step.")

    HybridAutomata.LoopWarning = not config.get("self_loop", False)

    system = HybridAutomata.from_json(automaton_data["automaton"])
    init_state = init_states[init_index]
    system.reset(init_state)

    state_trace = []
    mode_trace = []
    input_trace = []
    change_points = [0]

    for step in range(steps):
        state, mode, switched = system.next(dt)
        state_trace.append(state)
        mode_trace.append(mode)
        input_trace.append(system.getInput())
        if switched:
            change_points.append(step + 1)

    change_points.append(steps)

    state_array = np.array(state_trace, dtype=np.float64).T
    mode_array = np.array(mode_trace, dtype=np.int32)
    if input_trace and len(input_trace[0]) > 0:
        input_array = np.array(input_trace, dtype=np.float64).T
    else:
        input_array = np.empty((0, steps), dtype=np.float64)

    return {
        "state": state_array,
        "mode": mode_array,
        "input": input_array,
        "change_points": np.array(change_points, dtype=np.int32),
        "dt": dt,
        "total_time": total_time,
        "json_path": json_path,
        "init_index": init_index,
    }


def save_simulation(result: Dict[str, Any], out_dir: str) -> str:
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


def plot_simulation(result: Dict[str, Any], out_dir: str, show: bool) -> str:
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
        default="automata/non_linear/duffing.json",
        help="Path to the hybrid automaton description (default: automata/non_linear/duffing.json).",
    )
    parser.add_argument(
        "--init-index",
        type=int,
        default=0,
        help="Index of the initial state to use for the simulation (default: 0).",
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
    args = parser.parse_args()

    sim_result = simulate_hybrid_automata(args.json, init_index=args.init_index)
    unique_modes = np.unique(sim_result["mode"])
    print(
        f"Simulated {sim_result['total_time']} seconds with dt={sim_result['dt']} "
        f"using init_state[{args.init_index}] from {args.json}."
    )
    print(f"Visited modes: {unique_modes.tolist()}")
    print(f"Detected change points (steps): {sim_result['change_points'].tolist()}")

    if args.save:
        data_path = save_simulation(sim_result, args.out_dir)
        print(f"Saved simulation arrays to {data_path}")

    if not args.no_plot:
        figure_path = plot_simulation(sim_result, args.out_dir, show=args.show)
        print(f"Saved plot to {figure_path}")


if __name__ == "__main__":
    main()
