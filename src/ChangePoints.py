import numpy as np
from src.DE import DE


def mergeChangePoints(data, th: float):
    data = np.unique(np.sort(data))
    res = []
    last = None
    for pos in data:
        if last is None or pos - last > th:
            res.append(pos)
        last = pos
    return res


def find_change_point(data: np.array, get_feature, w: int = 10, th: float = 0.1, merge_th=None):
    r"""
    :param data: (N, M) Sample points for N variables.
    :param get_feature: Feature extraction function.
    :param w: Slide window size, default is 10.
    :param th: Error detection threshold. The default value is 0.1.
    :param merge_th: Change point merge threshold. The default value is w.
    :return: change_points, err_data: The change points, and the error in each position of N variables.
    """
    change_points = []
    error_datas = []
    tail_len = 0
    pos = 0
    last = None
    if merge_th is None:
        merge_th = w

    while pos + w < data.shape[1]:
        feature = get_feature(data[:, pos:(pos + w)])
        if last is not None:
            err = []
            for i in range(len(feature)):
                err.append(np.mean(np.abs(feature[i] - last[i])))
            err = np.array(err)
            if np.sum(err > th) > 0 and tail_len == 0:
                change_points.append(pos + w - 2)
                tail_len = w
            tail_len = max(tail_len - 1, 0)
        last = feature
        pos += 1

    res = mergeChangePoints(change_points, merge_th)
    res.append(data.shape[1])
    res.insert(0, 0)

    return res
