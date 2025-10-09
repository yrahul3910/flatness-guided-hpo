import os
import sys

import numpy as np


def parse(base_dir: str):
    files = [f"{base_dir}/{x}" for x in os.listdir(base_dir)]

    results = []
    for file in files:
        lines = open(file).readlines()
        lines = [line for line in lines if line.startswith("Score")][:300]
        lines = np.array([eval(x.split(":")[1])[-1] for x in lines])

        lines = lines.reshape((int(len(lines) // 10), 10))
        results.append(np.max(np.mean(lines, axis=-1), axis=0))

    return results


if __name__ == "__main__":
    results = parse(f"./results/{sys.argv[1]}/{sys.argv[2]}/")
