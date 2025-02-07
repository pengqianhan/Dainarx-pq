import numpy as np
import bisect


def max_min_abs_diff(a, b):
    sorted_b = sorted(b)
    max_diff = 0

    # 处理a中的每个元素，找到在sorted_b中的最小差
    for x in a:
        pos = bisect.bisect_left(sorted_b, x)
        if pos == 0:
            diff = abs(sorted_b[0] - x)
        elif pos == len(sorted_b):
            diff = abs(sorted_b[-1] - x)
        else:
            left = sorted_b[pos - 1]
            right = sorted_b[pos]
            diff = min(abs(x - left), abs(x - right))
        max_diff = max(max_diff, diff)

    return max_diff


def eva_trace(mode, trace, gt_mode, gt_trace, Ts):
    tc = max(max_min_abs_diff(mode, gt_mode), max_min_abs_diff(gt_mode, mode))
    mean_diff = np.mean(np.abs(trace - gt_trace))
    return tc * Ts, mean_diff



