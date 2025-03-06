import numpy as np
from collections import deque
from math import *
from src.DEConfig import FeatureExtractor
import matplotlib.pyplot as plt
import re


class ODESystem:
    @staticmethod
    def analyticalExpression(expr_list: str, var_list: list[str], input_list: list[str] = None):
        if input_list is None or input_list == []:
            input_expr = ""
        else:
            input_expr = ',' + ','.join(input_list)
        pattern = r'(' + '|'.join(map(re.escape, var_list)) + r')\[(\d+)\]'
        var_pattern = r'(\S+)\[(\d+)\]'

        def repl(match):
            string = match.group(1)
            number = match.group(2)
            idx = var_list.index(string)
            return f"x[{idx}][{number}]"

        def get_info(string):
            match = re.search(var_pattern, string)
            return var_list.index(match.group(1)), int(match.group(2))

        res = [(lambda x: 0) for _ in var_list]
        dim_list = [0 for _ in var_list]
        expr_list = expr_list.split(',')
        for expr in expr_list:
            var_expr, eq_expr = expr.split('=')
            eq_expr = re.sub(pattern, repl, eq_expr)
            idx, dim = get_info(var_expr)
            res[idx] = eval(f"lambda x{input_expr}: " + eq_expr)
            dim_list[idx] = dim
        return dim_list, res

    def analyticalInput(self, input_expr=None):
        res = []
        if input_expr is None:
            input_expr = {}
        for var in self.input_list:
            expr = input_expr.get(var)
            if expr is None:
                res.append(lambda t: 0)
            else:
                res.append(eval("lambda t: " + expr))
        return res

    def getInput(self, t):
        res = [fun(t) for fun in self.input_fun_list]
        return res

    def __init__(self, expr_list, var_list, input_list=None, init_state=None, method='rk'):
        if init_state is None:
            init_state = []
        if input_list is None:
            input_list = []
        dim_list, eq_list = ODESystem.analyticalExpression(expr_list, var_list, input_list)
        self.var_list = var_list
        self.input_list = input_list
        self.dim_list = dim_list
        self.eq_list = eq_list
        self.var_num = len(eq_list)
        self.state = init_state.copy()
        self.init_state = init_state.copy()
        self.max_dim = max(self.dim_list)
        self.method = method
        self.fit_state_size()
        self.input_fun_list = self.analyticalInput()
        self.now_time = 0.

    def fit_state_size(self):
        state = np.zeros((self.var_num, self.max_dim)).astype(np.float64)
        for idx_var in range(min(self.var_num, len(self.state))):
            for idx_dim in range(min(self.dim_list[idx_var], len(self.state[idx_var]))):
                state[idx_var][idx_dim] = self.state[idx_var][idx_dim]
        self.state = state

    def rk_fun(self, y, t):
        res = np.roll(y, -1, axis=1)
        for idx in range(self.var_num):
            res[idx][self.dim_list[idx] - 1] = self.eq_list[idx](y, *self.getInput(t))
        return res

    def next(self, dT: float):
        if self.method == "rk":
            k1 = self.rk_fun(self.state, self.now_time)
            k2 = self.rk_fun(self.state + 0.5 * dT * k1, self.now_time + 0.5 * dT)
            k3 = self.rk_fun(self.state + 0.5 * dT * k2, self.now_time + 0.5 * dT)
            k4 = self.rk_fun(self.state + dT * k3, self.now_time + dT)
            self.state = self.state + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            for idx_var in range(self.var_num):
                last_d = self.eq_list[idx_var](self.state)
                for i in range(self.dim_list[idx_var] - 1, -1, -1):
                    self.state[idx_var][i] = self.state[idx_var][i] + dT * last_d
                    last_d = self.state[idx_var][i]
        self.now_time += dT
        return self.state[:, 0]

    def reset(self, init_state, input_expr=None):
        res = []
        for var in self.var_list:
            res.append(init_state.get(var, []))
        self.clear(res)
        self.input_fun_list = self.analyticalInput(input_expr)

    def clear(self, init_state=None):
        if init_state is None:
            self.state = self.init_state.copy()
        else:
            self.state = init_state.copy()
        self.fit_state_size()
        self.now_time = 0.

    def load(self, other_sys):
        self.state = other_sys.state.copy()
        self.input_fun_list = other_sys.input_fun_list.copy()
        self.now_time = other_sys.now_time
        self.fit_state_size()


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


if __name__ == "__main__":
    ode = ODESystem("x1[1] = u1", ["x1"], ["u1"])
    ode.reset({"x1": [1]}, {"u1": "1"})
    res = []
    for i in range(100):
        val = ode.next(0.01)
        res.append(val[0])
    plt.plot(np.arange(len(res)), res)
    plt.show()
