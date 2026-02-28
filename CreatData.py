import numpy as np
import os
from math import *
import src.DE as DE
import matplotlib.pyplot as plt
from src.HybridAutomata import HybridAutomata
import json


def creat_data(json_path: str, data_path: str, dT: float, times: float, noise_level: float = 0.05):
    r"""
    :param json_path: File path of automata.
    :param data_path: Data storage path.
    :param dT: Discrete time.
    :param times: Total sampling time.
    :param noise_level: Measurement noise level (e.g., 0.05 means 5% of each dimension's std).
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        files = os.listdir(data_path)
        for file in files:
            os.remove(os.path.join(data_path, file))

    with open(json_path, 'r') as f:
        data = json.load(f)
        sys = HybridAutomata.from_json(data['automaton'])
        state_id = 0
        cnt = 0
        for init_state in data['init_state']:
            cnt += 1
            state_data = []
            mode_data = []
            input_data = []
            change_points = [0]
            sys.reset(init_state)
            now = 0.
            idx = 0
            while now < times:
                now += dT
                idx += 1
                state, mode, switched = sys.next(dT)
                state_data.append(state)
                mode_data.append(mode)
                input_data.append(sys.getInput())
                if switched:
                    change_points.append(idx)
            change_points.append(idx)
            state_data = np.transpose(np.array(state_data))
            input_data = np.transpose(np.array(input_data))
            mode_data = np.array(mode_data)
            # Add measurement noise scaled by each dimension's standard deviation
            if noise_level > 0:
                for dim in range(state_data.shape[0]):
                    dim_std = np.std(state_data[dim])
                    if dim_std > 0:
                        noise = np.random.normal(0, noise_level * dim_std, size=state_data.shape[1])
                    else:
                        noise = np.random.normal(0, noise_level, size=state_data.shape[1])
                    state_data[dim] += noise
            np.savez(os.path.join(data_path, "test_data" + str(state_id)),
                     state=state_data, mode=mode_data, input=input_data, change_points=change_points)
            state_id += 1


if __name__ == "__main__":
    creat_data('automata/1.json', 'data', 0.01, 10)
