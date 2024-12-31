import numpy as np


class Slice:
    RelativeErrorThreshold = 1e-3
    AbsoluteErrorThreshold = 1e-6

    def __init__(self, data, get_feature, isFront):
        self.data = data
        self.feature = []
        for v in self.data:
            self.feature.append(np.array(get_feature(v)).astype(np.float32))
        self.mode = None
        self.isFront = isFront

    def setMode(self, mode):
        self.mode = mode

    def __and__(self, other):
        for v1, v2 in zip(self.feature, other.feature):
            dis = np.linalg.norm(v1 - v2, ord=1)
            d1 = np.linalg.norm(v1, ord=1)
            d2 = np.linalg.norm(v2, ord=1)
            d_min = min(d1, d2)
            relative_dis = dis / max(d_min, Slice.AbsoluteErrorThreshold)
            if relative_dis > Slice.RelativeErrorThreshold and \
                    dis > Slice.AbsoluteErrorThreshold:
                return False
        return True
