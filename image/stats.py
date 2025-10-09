
class list(list):
    def map(self, f):
        return list(map(f, self))

import sys

from parsers.parse import general_parse
from parsers.parse_opentuner import parse_opentuner
from parsers.parse_random_svhn import general_parse as parse_random_svhn
from raise_utils.interpret import KruskalWallis

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    dataset = sys.argv[1]
    parsers = {
        "bohb": general_parse,
        "hyperopt": general_parse,
        "opentuner": parse_opentuner,
        "random": general_parse if dataset == "mnist" else parse_random_svhn,
        "stcvx_min": general_parse

    }

    scores = {}
    for alg, parser in parsers.items():
        dir = f"./results/{dataset}/{alg}/"

        scores[alg] = parser(dir)

    kw = KruskalWallis(scores)
    kw.pprint()

