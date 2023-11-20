import subprocess
import os
import threading

class list(list):
    def map(self, f):
        return list(map(f, self))

import sys
import numpy as np


def parse_turbo(base_dir: str):
    scores = []
    __proc = subprocess.Popen(f'ls {base_dir}/turbo/*.out', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    __proc.wait()
    EXIT_CODE = __proc.returncode
    __comm = __proc.communicate()
    _, STDERR = __comm[0].decode('utf-8').rstrip(), __comm[1].decode('utf-8').rstrip()
    _ = _.split('\n')
    try:
        _ = list(_)
    except ValueError:
        raise
    for file in _:
        print("Parsing", file)
        __proc = subprocess.Popen(f'tail -n1 {file}', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        __proc.wait()
        EXIT_CODE = __proc.returncode
        __comm = __proc.communicate()
        _, STDERR = __comm[0].decode('utf-8').rstrip(), __comm[1].decode('utf-8').rstrip()
        line = _
        line = eval(line.split(':')[1])[0]
        scores.append(line)

    return scores


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} DIR')
        sys.exit(1)

    dir = sys.argv[1]
    print('Scores:', scores)
    print('Median:', np.median(scores))
