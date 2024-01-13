import subprocess
import os
import threading

class list(list):
    def map(self, f):
        return list(map(f, self))

import sys
import numpy as np


def general_parse(dir):
    scores = []
    __proc = subprocess.Popen(f'ls {dir}/*.out', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        if EXIT_CODE != 0:
            print("ls failed:", STDERR)
            continue

        print("Parsing", file)
        __proc = subprocess.Popen(f'grep Accuracy {file} | cut -f 2 -d ":"', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        __proc.wait()
        EXIT_CODE = __proc.returncode
        __comm = __proc.communicate()
        _, STDERR = __comm[0].decode('utf-8').rstrip(), __comm[1].decode('utf-8').rstrip()
        _ = [float(x) for x in _.split('\n')]
        lines = _

        if 'bohb' in dir:
            lines = [-x for x in lines]

        scores.append(max(lines))

    return scores


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} DIR')
        sys.exit(1)

    dir = sys.argv[1]
    scores = general_parse(dir)
    print("Scores:", scores)
    print()
    print("Median:", np.median(scores))
