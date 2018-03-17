from os.path import dirname, realpath
import sys
from functools import partial

import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

cmap = cm.get_cmap('Spectral')

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.dim_reduction import run_dim_alg, get_data

#sys.path.insert(0, '/Users/jeremy.martinez/georgia-tech-code/ass3/jontay/src')
OUT = '{}/../../OUTPUT/PCA'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

def main():
    for i in [0.6, 0.7, 0.8, 0.9, 0.95]:
        #%% Data for 2
        decomp1 = partial(PCA, n_components=i, random_state=10)
        decomp2 = partial(PCA, n_components=i, random_state=10)
        run_dim_alg(r_X, r_y, 'pca', 'reviews', decomp1, i, OUT)
        run_dim_alg(c_X, c_y, 'pca', 'cancer', decomp2, i, OUT)

if __name__ == '__main__':
    main()
