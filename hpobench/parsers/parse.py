import subprocess
import os
import threading

class list(list):
    def map(self, f):
        return list(map(f, self))

import sys

from numpy import array, float32
import numpy as np


def parse(base_dir: str):
    __proc = subprocess.Popen(f'ls -1 {base_dir}/*.out', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    __proc.wait()
    EXIT_CODE = __proc.returncode
    __comm = __proc.communicate()
    _, STDERR = __comm[0].decode('utf-8').rstrip(), __comm[1].decode('utf-8').rstrip()
    _ = _.split('\n')
    try:
        _ = list(_)
    except ValueError:
        raise
    files = _
    
    results = []
    for file in files:
        print(f'Parsing {file}')
        __proc = subprocess.Popen(f'grep "^Score:" {file}', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        __proc.wait()
        EXIT_CODE = __proc.returncode
        __comm = __proc.communicate()
        _, STDERR = __comm[0].decode('utf-8').rstrip(), __comm[1].decode('utf-8').rstrip()
        _ = _.split('\n')
        try:
            _ = list(_)
        except ValueError:
            raise
        lines = _
        
        
        
        
        lines = np.array([eval(x.split(':')[1])[-1] for x in lines])

        lines = lines.reshape((int(len(lines) // 10), 10))
        results.append(np.max(np.mean(lines, axis=-1), axis=0))
    
    return results


if __name__ == '__main__':
    results = parse(f'./results/{sys.argv[1]}/{sys.argv[2]}/')
    print('AUC scores:', results)
    print('Median AUC:', np.median(results, axis=0))
