import copy
import re

import numpy as np
from .DE import ODE
import json
import matplotlib.pyplot as plt


class Node:
    def __init__(self, var_list, eq_list):
        self.var_list = var_list.copy()
        self.ode_list = []
        for var in var_list:
            fl = False
            for eq in eq_list:
                if var not in eq:
                    continue
                fl = True
                self.ode_list.append(ODE(eq, var_name=var, method='rk'))
                break
            if not fl:
                raise Exception("There is no matching equation for variable {}.".format(var))

    def next(self, dT: float):
        res = []
        for ode in self.ode_list:
            res.append(ode.next(dT))
        return res

    def load(self, other_node):
        for i in range(len(self.var_list)):
            self.ode_list[i].load(other_node.ode_list[i])

    def reset(self, init_state):
        for i in range(len(self.var_list)):
            self.ode_list[i].clear(init_state.get(self.var_list[i]))


class HybridAutomata:
    def __init__(self, info):
        var_list = re.split(r"\s*,\s*", info['var'])
        mode_list = {}

        adj = {}
        for mode in info['mode']:
            mode_id = mode['id']
            mode_list[mode_id] = Node(var_list, re.split(r"\s*,\s*", mode['eq']))
            adj[mode_id] = []
        for edge in info['edge']:
            u_v = re.findall(r'\d+', edge['direction'])
            fun = eval('lambda ' + info['var'] + ':' + edge['condition'])
            adj[int(u_v[0])].append((int(u_v[1]), fun))

        self.mode_state = info['mode'][0]['id']
        self.var_list = var_list
        self.mode_list = mode_list
        self.adj = adj

    def next(self, dT: float):
        res = self.mode_list[self.mode_state].next(dT)
        for to, fun in self.adj[self.mode_state]:
            if fun(*res):
                self.mode_list[to].load(self.mode_list[self.mode_state])
                self.mode_state = to
                break
        return res, self.mode_state

    def reset(self, init_state):
        self.mode_state = init_state.get('mode', self.mode_state)
        self.mode_list[self.mode_state].reset(init_state)


if __name__ == "__main__":
    with open('../automata/test.json', 'r') as f:
        data = json.load(f)
        sys = HybridAutomata(data['automation'])
        res = []
        for i in range(1000):
            res.append(sys.next(0.01)[0])
        res = np.array(res)
        x2_data = res[:, 1]
        plt.plot(np.arange(0, len(x2_data)), x2_data)
        plt.show()
