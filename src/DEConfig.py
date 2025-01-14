import numpy as np
from src.DE import DE


class FeatureExtractor:
    def __init__(self, dim: int, need_bias: bool, other_items: str):
        self.dim = dim
        self.need_bias = need_bias
        self.fun_list = DE.analyticalExpression(other_items)

    def get_items(self, data, idx):
        res = list(data[idx][(-self.dim):])
        delay_array = [0] + res.copy()
        for fun in self.fun_list:
            res.append(fun(delay_array))
        if self.need_bias:
            res.append(1.)
        return res

    def __call__(self, data, is_list=False, need_err=False):
        res = []
        err = []
        var_num = len(data) if not is_list else len(data[0])
        matrix_list = [[] for _ in range(var_num)]
        b_list = [[] for _ in range(var_num)]
        if is_list:
            for block in data:
                self.append_data(matrix_list, b_list, block)
        else:
            self.append_data(matrix_list, b_list, data)
        for a, b in zip(matrix_list, b_list):
            x = np.linalg.lstsq(a, b, rcond=None)[0]
            res.append(x)
            if need_err:
                err.append(np.sum(np.abs((a @ x) - b)))
        res = np.array(res)
        if need_err:
            return res, err
        return res

    def append_data(self, matrix_list, b_list, data: np.array):
        data = np.array(data)
        for i in range(len(data[0]) - self.dim):
            if i == 0:
                this_line = data[:, (self.dim - 1)::-1]
            else:
                this_line = data[:, (self.dim + i - 1):(i - 1):-1]
            for idx in range(len(this_line)):
                matrix_list[idx].append(self.get_items(this_line, idx))
                b_list[idx].append(data[idx][i + self.dim])


if __name__ == '__main__':
    get_feature = FeatureExtractor(1, True, "")
    feature, err = get_feature([[[1, 1, 1, 1, 1], [1, 3, 5, 7, 9]], [[2, 2, 2, 2, 2], [1, 3, 5, 7, 9]]], is_list=True,
                               need_err=True)
    print(feature, err)
