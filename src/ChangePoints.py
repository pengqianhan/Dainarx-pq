import numpy as np
import re


def analyticalExpression(expr_list: str):
    if expr_list == '':
        return []
    expr_list = expr_list.split(',')
    fun_list = [eval("lambda x: " + expr) for expr in expr_list]
    return fun_list


class FeatureExtractor:
    def __init__(self, dim: int, need_bias=False, other_items=""):
        self.dim = dim
        self.need_bias = need_bias
        self.fun_list = analyticalExpression(other_items)

    def __call__(self, data: np.array):
        A = []
        b = []
        for i in range(len(data) - self.dim):
            this_line = []
            for j in range(self.dim):
                this_line.append(data[i + j])
            this_line = list(reversed(this_line))
            if self.need_bias:
                this_line.append(1.)
            for fun in self.fun_list:
                this_line.append(fun(data[(i + self.dim)::-1]))
            A.append(this_line)
            b.append(data[i + self.dim])
        return np.linalg.lstsq(A, b, rcond=None)[0]




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


def FindChangePoint(data_list: np.array, dim: int = 3, w: int = 10, th: float = 0.1, merge_th=10):
    r"""
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
            res = getFeature(data[pos:(pos + w)], dim, need_bias=True)
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
