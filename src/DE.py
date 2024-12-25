import copy
import time

import numpy as np
from collections import deque
from math import *
import matplotlib.pyplot as plt
import re


def analyticalExpression(expr, val_name):
    general = re.sub(val_name, 'x', expr)
    prefix = re.sub("=(.)*", "", general)
    suffix = re.sub("(.)*=", "", general)
    dim = int(re.findall(r'\d+', prefix)[0])
    return dim, suffix


class ODE:

    def __init__(self, eq: str, init_state=None, var_name='x'):
        r"""
        :param eq:
            Expression of ordinary differential equation.
            The k derivative of x is expressed as x[k].
            The left side of the equal sign is the highest derivative of the equation,
            and the right side is the expression.
            Implicit functions are not supported.
        :param init_state:
            The initial state of the variable,
            init_state[k] represents the initial state of the k derivative of the variable.
            The default value is 0 for all dimensions.
            If the number of dimensions is less than +1, 0 is filled backward.
        :param var_name:
            The name of the variable in the expression, default is 'x'.
        """
        if init_state is None:
            init_state = []

        try:
            dim, expr = analyticalExpression(eq, var_name)
            self.dim = dim
            self.eq = eval("lambda x: " + expr)
            self.state = init_state.copy()
            self.fit_state_size()
            # check eq
            self.eq(self.state)
        except Exception as ex:
            raise Exception("invalid ode \"{}\"".format(eq), ex)

        self.init_state = self.state.copy()

    def fit_state_size(self):
        while len(self.state) > self.dim + 1:
            self.state.pop()
        while len(self.state) < self.dim + 1:
            self.state.append(0)

    def next(self, dT: float):
        self.state[-1] = self.eq(self.state)
        for i in range(self.dim - 1, -1, -1):
            self.state[i] = self.state[i] + dT * self.state[i + 1]
        return self.state[0]

    def clear(self, init_state=None):
        if init_state is None:
            self.state = self.init_state.copy()
        else:
            self.state = init_state.copy()
        self.fit_state_size()

    def load(self, other_ode):
        self.state = other_ode.state.copy()
        self.fit_state_size()


class DE:
    def __init__(self, eq, init_state, bias=0):
        self.init_state = init_state
        self.state = deque(init_state)
        self.eq = eq
        self.dim = len(eq)
        self.bias = bias

    def next(self):
        val = 0
        for i in range(self.dim):
            val += self.state[i] * self.eq[i]
        self.state.appendleft(val)
        self.state.pop()
        return val

    def clear(self):
        self.state = self.init_state


if __name__ == "__main__":
    ode = ODE("pos[2] = -3 * pos[1] - 25 * pos[0] + 25", var_name='pos')
    # ode = DE([1, 1], [1, 1])
    X = []
    Y = []
    start = time.time()
    for pos in range(500):
        X.append(pos)
        Y.append(ode.next(0.01))
    print(time.time() - start)
    plt.plot(X, Y)
    plt.show()
