import subprocess
import os

class list(list):
    def map(self, f):
        return list(map(f, self))

import numpy as np

scores = []
_ = subprocess.Popen(f'ls *.out', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
_ = _.split('\n')
try:
    _ = list(_)
except ValueError:
    raise
for file in _:
    print("Parsing", file)
    _ = subprocess.Popen(f'grep "\[main\] Accuracy" {file}', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').rstrip()
    _ = _.split('\n')
    try:
        _ = list(_)
    except ValueError:
        raise
    lines = _.map(lambda x: float(x.split(":")[1]))
    scores.append(max(lines))

print("Scores:", scores)
print()
print("Median:", np.median(scores))
