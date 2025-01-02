import json

import numpy as np
import matplotlib.pyplot as plt

from CreatData import creat_data

from src.CurveSlice import Slice
from src.ChangePoints import FindChangePoint, FeatureExtractor
from src.Clustering import clustering
from src.GuardLearning import guard_learning
from src.BuildSystem import build_system, get_init_state
import os
import re


def get_ture_chp(data):
    last = None
    change_points = [0]
    idx = 0
    for now in data:
        if last is not None and last != now:
            change_points.append(idx)
        idx += 1
        last = now
    change_points.append(len(data))
    return change_points


def cut_segment(cut_data, data, change_points, get_feature):
    last = 0
    for point in change_points:
        if point == 0:
            continue
        cut_data.append(Slice(data[:, last:point], get_feature, last == 0))
        last = point
    return cut_data


def run(data_list, get_feature, config):
    # chp detection
    slice_data = []

    detect_fun = FeatureExtractor(config['dim'], config['need_bias'])
    detect_fun = get_feature

    for data in data_list:
        change_points, err_data = FindChangePoint(data, detect_fun)
        print(change_points)
        cut_segment(slice_data, data, change_points, detect_fun)
    Slice.fit_threshold(slice_data)
    clustering(slice_data)
    adj = guard_learning(slice_data)
    sys = build_system(slice_data, adj, get_feature, config['need_bias'], config['other_items'])
    model_now = adj[(1, 2)]
    # print(model_now.coef_, model_now.intercept_)
    return sys


def get_config(json_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    config = {}
    if json_path.isspace():
        config = {'dt': 0.01, 'total_time': 10, 'dim': 3,
                  'need_bias': False, 'other_items': '', 'kernel': 'linear'}
    else:
        with open(json_path) as f:
            json_file = json.load(f).get('fit_config', {})
        config['dt'] = json_file.get('dt', 0.01)
        config['total_time'] = json_file.get('total_time', 10)
        config['dim'] = json_file.get('dim', 3)
        config['need_bias'] = json_file.get('need_bias', False)
        config['other_items'] = json_file.get('other_items', '')
        config['kernel'] = json_file.get('kernel', 'linear')
    return config


def main(json_path: str, need_creat=False):
    config = get_config(json_path)
    print('config: ')
    for key, value in config.items():
        print(f'\t{key}: {value}')

    if need_creat:
        print("Data being generated!")
        creat_data(json_path, 'data', config['dt'], config['total_time'])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = []
    mode_list = []
    for root, dirs, files in os.walk(os.path.join(current_dir, "data")):
        print("Loading data!")
        for file in sorted(files):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['arr_0'], npz_file['arr_1']
            print("GT: ", get_ture_chp(mode_data_temp))
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)

    print("Be running!")
    get_feature = FeatureExtractor(config['dim'], config['need_bias'], config['other_items'])
    sys = run(data, get_feature, config)

    print("Start simulation")
    fit_idx = 1
    data = data[fit_idx]
    mode_list = mode_list[fit_idx]

    init_state = get_init_state(data, mode_list, config['dim'])
    fit_data = [data[:, i] for i in range(config['dim'])]
    mode_data = []
    sys.reset(init_state)
    for i in range(data.shape[1] - config['dim']):
        state, mode = sys.next(config['dt'])
        fit_data.append(state)
        mode_data.append(mode)
    fit_data = np.array(fit_data)

    for var_idx in range(data.shape[0]):
        plt.plot(np.arange(len(data[var_idx])), data[var_idx], color='c')
        plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], color='r')
        plt.show()


if __name__ == "__main__":
    main("./automata/1.json", need_creat=True)
