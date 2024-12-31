import numpy as np
from src.ChangePoings import FindChangePoint
from src.Clustering import clustering_learning
from src.GuardLearning import guard_learning
from src.ODELearning import ODELearning
import os
import re


def get_ture_chp(data):
    last = None
    change_points = [1]
    idx = 0
    for now in data:
        if last is not None and last != now:
            change_points.append(idx + 1)
        idx += 1
        last = now
    change_points.append(len(data))
    return change_points


def run(trace, num_var, num_ud):
    # chp detection
    x_lists = []
    ud_lists = []

    for i in range(len(trace)):
        chpoints, err_data = FindChangePoint(trace[i]['x'], num_var)
        trace[i]['chpoints'] = chpoints
        x_lists.append(trace[i]['x'])
        ud_lists.append(trace[i]['ud'])
        print("GT: ", get_ture_chp(trace[i]['mode']))
        print(chpoints)

        # trace[i]['chpoints'] = trace[i]['true_chp']
        # trace[i]['chpoints_per_var'][0] = trace[i]['true_chp']
        # trace[i]['chpoints_per_var'][1] = trace[i]['true_chp']

    x = np.hstack(x_lists)
    ud = 0
    if num_ud:
        ud = np.hstack(ud_lists)

    # clustering
    trace = clustering_learning(trace, x, ud, num_var, num_ud)

    # guard learning
    models = guard_learning(trace, 'linear')

    # ODE Learning
    # ode = ODELearning(trace, x, ud)

    # svm test
    model_now = models['1'][0]
    x_test = np.array([[7.45983, 12.43306]])
    print(model_now.predict(x_test))
    print(model_now.coef_, model_now.intercept_)
    # print(model_now.intercept_ / model_now.coef_[0])

    # 以下代码只用来画图
    # plot_idx = 1
    # err = err_data[plot_idx] / np.max(err_data[plot_idx]) * np.max(trace[i]['x'][plot_idx])
    # plt.plot(np.arange(0, len(trace[i]['x'][plot_idx])), trace[i]['x'][plot_idx])
    # plt.plot(np.arange(0, len(err)), err)
    # plt.show()
    print('done')


def main(num_var, num_ud):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(os.path.join(current_dir, "data")):

        trace = []

        for file in sorted(files):
            if re.search(r"(.)*\.npz", file) is None:
                continue

            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['arr_0'], npz_file['arr_1']
            trace_temp = {}

            trace_temp['x'] = state_data_temp
            trace_temp['mode'] = mode_data_temp
            trace_temp['true_chp'] = get_ture_chp(mode_data_temp)
            if num_ud == 0:
                trace_temp['ud'] = None
            else:
                trace_temp['ud'] = npz_file['arr_2']

            trace.append(trace_temp)

        run(trace, num_var, num_ud)


if __name__ == "__main__":
    num_x = 2
    num_u = 0

    main(num_x, num_u)
