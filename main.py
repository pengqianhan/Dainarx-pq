import numpy as np
import matplotlib.pyplot as plt
from src.ChangePoings import FindChangePoint
import os
import re


def run(data):
    for cur in data:
        change_points, err_data = FindChangePoint(cur)
        plt.plot(np.arange(0, len(err_data)), err_data)
        plt.show()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(os.path.join(current_dir, "data")):
        for file in sorted(files):
            if re.search(r"(.)*\.npy", file) is None:
                continue
            data = np.load(os.path.join(root, file))
            run(data)


if __name__ == "__main__":
    main()