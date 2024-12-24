import numpy as np
import os
from math import *
import src.DE as DE
import matplotlib.pyplot as plt


def creat_data(data_num: int, data_len: int, ode_list: list[DE.ODE], dT: float, init_state_list=None, path="./data/"):

    first_data = None

    for data_idx in range(data_num):
        for idx in range(len(ode_list)):
            if init_state_list is None:
                ode_list[idx].clear()
            else:
                ode_list[idx].clear(init_state_list[data_idx][idx])

        res = []
        need_num = data_len

        while need_num > 0:
            state = []
            for idx in range(len(ode_list)):
                state.append(ode_list[idx].next(dT))
            res.append(state)
            need_num -= 1
        res = np.transpose(np.array(res))

        if path is not None:
            np.save(os.path.join(path, "test_data" + str(data_idx)), res)

        if first_data is None:
            first_data = res

    return first_data


if __name__ == "__main__":
    first_res = creat_data(1, 500, [DE.ODE(2, "-3 * x[1] - 25 * x[0] + 25", [0, 0, 0])], 0.01)
    print(first_res[0])
    plt.plot(np.arange(0, len(first_res[0])), first_res[0])
    plt.show()
