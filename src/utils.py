from src.CurveSlice import Slice
import matplotlib.pyplot as plt


def get_ture_chp(data):
    last = None
    change_points = [0]
    idx = 0
    for now in data:
        if last is not None and last != now:
            change_points.append(idx)
        idx += 1
        last = now
    change_points.append(len(data))
    return change_points
