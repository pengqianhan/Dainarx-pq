import numpy as np
import matplotlib.pyplot as plt
from src.ChangePoings import FindChangePoint, getFeature
from src.DE import DE, ODE
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
    change_points, err_data = FindChangePoint(data, dim=4)
    # change_points 是一个一维数组，合并所有轨迹的change points
    # err_data 是一个n*m长的数组，n是变量个数，m是采样点个数，表示每个位置计算的误差
    print(change_points)

    # 以下代码只用来画图
    plot_idx = data.shape[0] - 1
    # err = err_data[plot_idx] / np.max(err_data[plot_idx]) * np.max(data[plot_idx])
    # plt.plot(np.arange(0, len(data[plot_idx])), data[plot_idx])
    # plt.plot(np.arange(0, len(err)), err)
    # plt.show()

    # 以下代码用来拟合图像
    cur = data[plot_idx]
    pos = 1
    sys = None
    ch_idx = 0
    fit_cur = []
    if len(change_points) > 0:
        next_ch = change_points[0]
    else:
        next_ch = len(cur)
    while pos < len(cur):
        if sys is None:
            fit_cur.append(cur[0])
            feature = getFeature(cur[pos:next_ch], 3, lambda arr: arr[2] * arr[2], lambda arr: arr[1] * arr[1],
                                 lambda arr: arr[0] * arr[0])
            print("feature: ", feature)
            # sys = DE(getFeature(cur[0:next_ch], 3), [cur[0]])
            sys = DE(feature, [cur[0]], 0, lambda arr: arr[0] * arr[0], lambda arr: arr[1] * arr[1],
                     lambda arr: arr[2] * arr[2])
        else:
            fit_cur.append(sys.next())
        pos += 1
        if len(cur) > pos >= next_ch:
            print("change")
            ch_idx += 1
            if ch_idx < len(change_points):
                next_ch = change_points[ch_idx]
            else:
                next_ch = len(cur)
            feature = getFeature(cur[pos:next_ch], 3, lambda arr: arr[2] * arr[2], lambda arr: arr[1] * arr[1],
                                 lambda arr: arr[0] * arr[0])
            print(feature)
            new_sys = DE(feature, [], 0, lambda arr: arr[0] * arr[0], lambda arr: arr[1] * arr[1],
                         lambda arr: arr[2] * arr[2])
            new_sys.load(sys)
            sys = new_sys
    plt.plot(np.arange(len(cur)), np.array(cur))
    plt.plot(np.arange(len(fit_cur)), np.array(fit_cur))
    plt.show()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(os.path.join(current_dir, "data")):
        for file in sorted(files):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data, mode_data = npz_file['arr_0'], npz_file['arr_1']
            # state_data 是一个n*m的数组，n是变量个数，m是采样点个数，表示每个变量的采样点
            # mode_data 是一个m长的数组，表示每个时刻的状态
            run(state_data, mode_data)


if __name__ == "__main__":
    main()
