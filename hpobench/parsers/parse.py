import os
import sys

from numpy import array, float32
import numpy as np


def parse(base_dir: str):
    files = [f'{base_dir}/{x}' for x in os.listdir(base_dir)]
    
    results = []
    for file in files:
        print(f'Parsing {file}')
        lines = open(file, 'r').readlines()
        lines = [line for line in lines if line.startswith('Score')][:300]
        lines = np.array([eval(x.split(':')[1])[-1] for x in lines])

        lines = lines.reshape((int(len(lines) // 10), 10))
        results.append(np.max(np.mean(lines, axis=-1), axis=0))
    
    return results


if __name__ == '__main__':
    results = parse(f'./results/{sys.argv[1]}/{sys.argv[2]}/')
    print('AUC scores:', results)
    print('Median AUC:', np.median(results, axis=0))
