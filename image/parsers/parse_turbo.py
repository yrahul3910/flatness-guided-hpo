import os
import subprocess


class list(list):
    def map(self, f):
        return list(map(f, self))

import sys


def parse_turbo(base_dir: str):
    scores = []
    __proc = subprocess.Popen(f"ls {base_dir}/turbo/*.out", shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

        __proc = subprocess.Popen(f"tail -n1 {file}", shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        __proc.wait()
        EXIT_CODE = __proc.returncode
        __comm = __proc.communicate()
        _, _STDERR = __comm[0].decode("utf-8").rstrip(), __comm[1].decode("utf-8").rstrip()
        line = _
        line = eval(line.split(":")[1])[0]
        scores.append(line)

    return scores


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    dir = sys.argv[1]
