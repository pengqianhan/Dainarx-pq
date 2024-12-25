import numpy as np
import matplotlib.pyplot as plt
from src.ChangePoings import FindChangePoint
import os
import re


def getGT(data):
    last = None
    change_points = []
    idx = 0
    for now in data:
        if last is not None and last != now:
            change_points.append(idx)
        idx += 1
        last = now
    return change_points


def run(data, modes):
    gt = getGT(modes)
    print("GT: ", gt)
    change_points, err_data = FindChangePoint(data)
    # change_points 是一个一维数组，合并所有轨迹的change points
    # err_data 是一个n*m长的数组，n是变量个数，m是采样点个数，表示每个位置计算的误差
    print(change_points)

    # 以下代码只用来画图
    plot_idx = 1
    err = err_data[plot_idx] / np.max(err_data[plot_idx]) * np.max(data[plot_idx])
    plt.plot(np.arange(0, len(data[plot_idx])), data[plot_idx])
    plt.plot(np.arange(0, len(err)), err)
    plt.show()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(os.path.join(current_dir, "data")):
        for file in sorted(files):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            print("yes")
            npz_file = np.load(os.path.join(root, file))
            state_data, mode_data = npz_file['arr_0'], npz_file['arr_1']
            # state_data 是一个n*m的数组，n是变量个数，m是采样点个数，表示每个变量的采样点
            # mode_data 是一个m长的数组，表示每个时刻的状态
            run(state_data, mode_data)


if __name__ == "__main__":
    main()
