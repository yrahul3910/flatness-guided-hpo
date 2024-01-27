# File description

* `*_hpo.py`: Implementation of HPO using the algorithm in the filename.
* `common.py`: Shared functions. Probably should move to src/util.
* `stats.pysh`: Code to run statistics.
* `stcvx.py`: STCVX algorithm (strong convexity-based HPO)
* `parsers/`: Result parsers for each output file
* `results/`: The raw results. Warning: the files are compressed, and expand to about 4GB.
* `src/`: Utility functions

Please download the SVHN dataset, in the cropped digits format, from http://ufldl.stanford.edu/housenumbers/, and place the files here.
