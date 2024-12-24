import numpy as np
from collections import deque
from math import *
import matplotlib.pyplot as plt


class ODE:
    def __init__(self, dim: int, eq: str, init_state=None):
        if init_state is None:
            init_state = []
        self.init_state = init_state
        self.state = init_state
        self.dim = dim
        self.eq = eq

    def next(self, dT: float):
        x = self.state
        x[-1] = eval(self.eq)
        for i in range(self.dim - 1, -1, -1):
            x[i] = x[i] + dT * x[i + 1]
        self.state = x
        return x[0]

    def clear(self, init_state=None):
        if init_state is None:
            self.state = self.init_state
        else:
            self.state = init_state


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
    ode = ODE(2, "-3 * x[1] - 25 * x[0] + 25", [0, 0, 0])
    # ode = DE([1, 1], [1, 1])
    X = []
    Y = []
    for pos in range(500):
        X.append(pos)
        Y.append(ode.next(0.01))
    plt.plot(X, Y)
    plt.show()
