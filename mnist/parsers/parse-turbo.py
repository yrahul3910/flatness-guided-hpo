import subprocess
import os
import threading

class list(list):
    def map(self, f):
        return list(map(f, self))

import numpy as np


scores = []
__proc = subprocess.Popen(f'ls *.out', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

print('Scores:', scores)
print('Median:', np.median(scores))
