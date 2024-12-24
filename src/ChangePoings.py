import numpy as np


def getFeature(data: np.array, dim: int) -> np.array:
    A = []
    b = []
    for i in range(len(data) - dim):
        this_line = []
        for j in range(dim):
            this_line.append(data[i + j])
        A.append(this_line)
        b.append(data[i + dim])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def FindChangePoint(data: np.array, dim: int = 3, w: int = 10):
    tail_len = 0
    pos = 0
    last = None
    change_points = []
    error_data = []

    while pos + w < len(data):
        res = getFeature(data[pos:(pos + w)], dim)
        if last is not None:
            err = np.mean(res - last)
            error_data.append(err)
            if abs(err) > 0.1 and tail_len == 0:
                change_points.append(pos)
                tail_len = w
            tail_len = max(tail_len - 1, 0)
        last = res
        pos += 1

    return change_points, error_data
