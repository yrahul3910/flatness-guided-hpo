import subprocess
import os
import threading

class list(list):
    def map(self, f):
        return list(map(f, self))

from raise_utils.interpret import KruskalWallis

from parsers.parse import general_parse
from parsers.parse_opentuner import parse_opentuner
from parsers.parse_turbo import parse_turbo


if __name__ == '__main__':
    parsers = {
        'bohb': general_parse,
        'hyperopt': general_parse,
        'opentuner': parse_opentuner,
        'random': general_parse,
        'stcvx': general_parse,
        'turbo': parse_turbo
    }

    scores = {}
    for alg, parser in parsers.items():
        print(f'Parsing {alg}...')
        dir = './results/'
        
        if parser == general_parse:
            dir += f'{alg}/'

        print(f'Using dir = {dir}')
        scores[alg] = parser(dir)
    
    print(scores)
    kw = KruskalWallis(scores)
    kw.pprint()

