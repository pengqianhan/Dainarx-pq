import numpy as np
import itertools

def getFeature(data: np.array, dim: int, need_bias=False) -> np.array:
    A = []
    b = []
    for i in range(len(data) - dim):
        this_line = []
        for j in range(dim):
            this_line.append(data[i + j])
        this_line = list(reversed(this_line))
        if need_bias:
            this_line.append(1.)
        A.append(this_line)
        b.append(data[i + dim])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def mergeChangePoints(data, th):
    data = np.array(data).ravel()
    data = np.unique(np.sort(data))
    res = []
    last = None
    for pos in data:
        if last is None or pos - last > th:
            res.append(pos)
        last = pos
    return res


def FindChangePoint(data_list: np.array, num_var, dim: int = 3, w: int = 10, th: float = 0.1, merge_th=10):
    r"""
    :param data_list: (N, M) Sample points for N variables.
    :param dim: Fit the dimension of the difference equation, which defaults to 3.
    :param w: Slide window size, default is 10.
    :param th: Error detection threshold. The default value is 0.1.
    :param merge_th: Change point merge threshold. The default value is 10.
    :return: change_points, err_data: The change points, and the error in each position of N variables.
    """
    change_points_list = []
    error_datas = []
    for idx in range(num_var):
        data = data_list[idx]
        tail_len = 0
        pos = 0
        last = None
        change_points = [1]
        error_data = []

        while pos + w < len(data):
            res = getFeature(data[pos:(pos + w)], dim, True)
            if last is not None:
                err = np.mean(np.abs(res - last))
                error_data.append(err)
                if abs(err) > th and tail_len == 0:
                    change_points.append(pos + w - 1)
                    tail_len = w
                tail_len = max(tail_len - 1, 0)
            last = res
            pos += 1
        change_points.append(1001)
        change_points_list.append(change_points)

        error_datas.append(error_data)

    merged_list = list(dict.fromkeys(itertools.chain(*change_points_list)))

    return change_points_list, merged_list, np.array(error_datas)
