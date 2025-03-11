import copy

import numpy as np
from src.CurveSlice import Slice
from src.HybridAutomata import HybridAutomata
from src.DE import DE
from src.DE_System import DESystem


class ModelFun:
    def __init__(self, model):
        self.model = copy.copy(model)

    def __call__(self, *args):
        return self.model.predict([[*args]])[0] > 0.5


def build_system(data: list[Slice], res_adj: dict, get_feature):
    data_of_mode = {}
    for cur in data:
        if data_of_mode.get(cur.mode) is None:
            data_of_mode[cur.mode] = [[], []]
        data_of_mode[cur.mode][0].append(cur.data)
        data_of_mode[cur.mode][1].append(cur.input_data)
    mode_list = {}
    for (mode, cur_list) in data_of_mode.items():
        feature_list = get_feature(cur_list[0], cur_list[1], is_list=True)[0]
        mode_list[mode] = DESystem(feature_list, [], [], get_feature)
    adj = {}
    for (u, v), (model, reset_fun) in res_adj.items():
        if adj.get(u) is None:
            adj[u] = []
        adj[u].append((v, ModelFun(model), reset_fun))
    return HybridAutomata(mode_list, adj)


def get_init_state(data, mode_list, bias):
    init_state = {'mode': mode_list[bias - 1]}
    for i in range(data.shape[0]):
        init_state['x' + str(i)] = data[i, (bias - 1)::-1]
    return init_state
