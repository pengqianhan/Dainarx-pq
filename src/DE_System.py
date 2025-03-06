import numpy as np
from collections import deque
from math import *
from src.DEConfig import FeatureExtractor
import matplotlib.pyplot as plt
import re


class DESystem:
    def __init__(self, eq, init_state, config: FeatureExtractor):
        self.var_num = len(eq)
        self.config = config
        self.init_state = init_state.copy()
        self.state = init_state.copy()
        self.eq = eq.copy()
        self.fit_state_size()
        self.var_list = [('x' + str(idx)) for idx in range(self.var_num)]

    def fit_state_size(self):
        state = [deque(0 for _ in range(self.config.dim)) for _ in range(self.var_num)]
        for i in range(min(len(self.state), self.var_num)):
            for j in range(min(len(self.state[i]), self.config.dim)):
                state[i][j] = self.state[i][j]
        self.state = state

    def next(self):
        res = []
        for i in range(self.var_num):
            res.append(np.dot(self.config.get_items(self.state, i), self.eq[i]))
        for i in range(self.var_num):
            self.state[i].pop()
            self.state[i].appendleft(res[i])
        return res

    def reset(self, init_state):
        res = []
        for var in self.var_list:
            res.append(init_state.get(var, []))
        self.clear(res)

    def clear(self, init_state):
        self.state = init_state.copy()
        self.fit_state_size()

    def load(self, other):
        self.state = other.state.copy()
        self.fit_state_size()

