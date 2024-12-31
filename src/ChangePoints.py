import numpy as np
from src.DE import DE


class FeatureExtractor:
    def __init__(self, dim: int, need_bias=False, other_items=""):
        self.dim = dim
        self.need_bias = need_bias
        self.fun_list = DE.analyticalExpression(other_items)

    def __call__(self, data):
        matrix_a = []
        b = []
        if type(data) == list or len(data.shape) > 1:
            for cur in data:
                self.append_data(matrix_a, b, cur)
        else:
            self.append_data(matrix_a, b, data)
        return np.linalg.lstsq(matrix_a, b, rcond=None)[0]

    def append_data(self, matrix_a, b, data: np.array):
        for i in range(len(data) - self.dim):
            this_line = []
            for j in range(self.dim):
                this_line.append(data[i + j])
            this_line = list(reversed(this_line))
            for fun in self.fun_list:
                if i == 0:
                    this_line.append(fun(data[(i + self.dim)::-1]))
                else:
                    this_line.append(fun(data[(i + self.dim):(i - 1):-1]))
            if self.need_bias:
                this_line.append(1.)
            matrix_a.append(this_line)
            b.append(data[i + self.dim])


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


def mergeChangePoints(data, th: float):
    data = np.unique(np.sort(data))
    res = []
    last = None
    for pos in data:
        if last is None or pos - last > th:
            res.append(pos)
        last = pos
    return res


def FindChangePoint(data_list: np.array, get_feature, w: int = 10, th: float = 0.1, merge_th=10):
    r"""
    :param get_feature:
    :param data_list: (N, M) Sample points for N variables.
    :param dim: Fit the dimension of the difference equation, which defaults to 3.
    :param w: Slide window size, default is 10.
    :param th: Error detection threshold. The default value is 0.1.
    :param merge_th: Change point merge threshold. The default value is 10.
    :return: change_points, err_data: The change points, and the error in each position of N variables.
    """
    change_points = []
    error_datas = []
    for idx in range(data_list.shape[0]):
        data = data_list[idx]
        tail_len = 0
        pos = 0
        last = None
        error_data = []

        while pos + w < len(data):
            res = get_feature(data[pos:(pos + w)])
            if last is not None:
                err = np.mean(np.abs(res - last))
                error_data.append(err)
                if abs(err) > th and tail_len == 0:
                    change_points.append(pos + w - 2)
                    tail_len = w
                tail_len = max(tail_len - 1, 0)
            last = res
            pos += 1
        error_datas.append(error_data)

    res = mergeChangePoints(change_points, merge_th)
    res.append(data_list.shape[1])
    res.insert(0, 0)

    return res, np.array(error_datas)
