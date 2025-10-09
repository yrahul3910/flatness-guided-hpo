import os
import subprocess


class list(list):
    def map(self, f):
        return list(map(f, self))

import sys


def general_parse(dir):
    scores = []
    __proc = subprocess.Popen(f"ls {dir}/*.out", shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    __proc.wait()
    EXIT_CODE = __proc.returncode
    __comm = __proc.communicate()
    _, _STDERR = __comm[0].decode("utf-8").rstrip(), __comm[1].decode("utf-8").rstrip()
    _ = _.split("\n")
    try:
        _ = list(_)
    except ValueError:
        raise
    for file in _:
        if EXIT_CODE != 0:
            continue

        __proc = subprocess.Popen(f"tail -n2 {file} | head -n1", shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        __proc.wait()
        EXIT_CODE = __proc.returncode
        __comm = __proc.communicate()
        _, _STDERR = __comm[0].decode("utf-8").rstrip(), __comm[1].decode("utf-8").rstrip()
        lines = eval(_)
        lines = [x for x in lines if x != 100.]

        scores.append(max(lines))

    return scores


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    dir = sys.argv[1]
    scores = general_parse(dir)
