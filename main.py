import numpy as np

from src.CurveSlice import Slice

from src.ChangePoints import FindChangePoint, getFeature, FeatureExtractor
from src.Clustering import clustering
from src.GuardLearning import guard_learning
from src.ODELearning import ODELearning
import os
import re
get_feature = FeatureExtractor(3)


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


def cut_segment(cut_data, data, change_points):
    last = 0
    for point in change_points:
        if point == 0:
            continue
        cut_data.append(Slice(data[:, last:point], get_feature, last == 0))
        last = point
    return cut_data


def run(data_list):
    # chp detection
    slice_data = []

    for data in data_list:
        change_points, err_data = FindChangePoint(data)
        print(change_points)
        cut_segment(slice_data, data, change_points)
    clustering(slice_data)
    adj = guard_learning(slice_data)
    to, model_now = adj[1][0]
    print(model_now.coef_, model_now.intercept_)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(os.path.join(current_dir, "data")):
        data = []

        for file in sorted(files):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['arr_0'], npz_file['arr_1']
            data.append(state_data_temp)

        run(data)


if __name__ == "__main__":
    main()
