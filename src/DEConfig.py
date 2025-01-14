import re

import numpy as np
from src.DE import DE


class FeatureExtractor:
    @staticmethod
    def unfoldItem(expr_list, idx, var_num):
        res = []
        expr_list = expr_list.copy()
        for expr in expr_list:
            expr = re.sub(r'x\[', 'x[' + str(idx) + '][', expr)
            expr = re.sub(r'x(\d)', r'x[\1]', expr)
            for i in range(var_num):
                if i == idx:
                    continue
                res.append(re.sub(r'x_\[', 'x[' + str(i) + '][', expr))
        return res

    @staticmethod
    def unfoldDigit(expr_list, dim):
        res = []
        for expr in expr_list:
            for i in range(dim):
                res.append(re.sub(r'\[\?', r"[" + str(i), expr))
        return res

    @staticmethod
    def extractValidExpression(expr_list, idx):
        res = []
        for expr in expr_list:
            if expr.isspace() or expr == '':
                continue
            expr = re.sub(r'\[(\d+)', lambda x: '[' + str(int(x.group(1)) - 1), expr)
            if ':' not in expr:
                res.append(expr)
            else:
                filed, s = expr.split(':')
                if 'x' + str(idx) in filed:
                    res.append(s)
        return res

    @staticmethod
    def analyticalExpression(expr_list: str, var_num, dim):
        res = [[] for _ in range(var_num)]
        expr_list = expr_list.split(';')
        for idx in range(var_num):
            expr = FeatureExtractor.extractValidExpression(expr_list, idx)
            expr = FeatureExtractor.unfoldDigit(expr, dim)
            expr = FeatureExtractor.unfoldItem(expr, idx, var_num)
            for s in expr:
                res[idx].append(eval('lambda x: ' + s))
        return res

    def __init__(self, var_num: int, dim: int, need_bias: bool, other_items: str):
        self.var_num = var_num
        self.dim = dim
        self.need_bias = need_bias
        self.fun_list = FeatureExtractor.analyticalExpression(other_items, var_num, dim)

    def get_items(self, data, idx):
        res = []
        for i in range(len(data[idx]) - self.dim, len(data[idx])):
            res.append(data[idx][i])
        for fun in self.fun_list[idx]:
            res.append(fun(data))
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
    get_feature = FeatureExtractor(2, 1, True, "")
    feature, err = get_feature([[[1, 1, 1, 1, 1], [1, 3, 5, 7, 9]], [[2, 2, 2, 2, 2], [1, 3, 5, 7, 9]]], is_list=True,
                               need_err=True)
    print(feature, err)
