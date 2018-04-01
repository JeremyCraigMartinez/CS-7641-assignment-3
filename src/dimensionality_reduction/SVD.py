from os.path import dirname, realpath
from collections import defaultdict
from itertools import product
import sys
from functools import partial

import pandas as pd
from sklearn.decomposition import TruncatedSVD

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.clustering import pairwiseDistCorr, reconstructionError
from helpers.dim_reduction import run_dim_alg, get_data
from helpers.constants import SVD_DIMS_R, SVD_DIMS_C

OUT = '{}/../../OUTPUT/SVD'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

def main():
    for i, val in enumerate(SVD_DIMS_R): # [0.6, 0.7, 0.8, 0.9]:
        decomp1 = TruncatedSVD(n_components=val)
        run_dim_alg(r_X, r_y, 'reviews', decomp1, val, OUT)
    for i, val in enumerate(SVD_DIMS_C): # [0.6, 0.7, 0.8, 0.9]:
        decomp2 = TruncatedSVD(n_components=val)
        run_dim_alg(c_X, c_y, 'cancer', decomp2, val, OUT)

if __name__ == '__main__':
    main()
