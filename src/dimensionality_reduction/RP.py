from os.path import dirname, realpath
from collections import defaultdict
from itertools import product
import sys
from functools import partial

import pandas as pd
from sklearn.random_projection import SparseRandomProjection

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.clustering import pairwiseDistCorr, reconstructionError
from helpers.dim_reduction import run_dim_alg, get_data

OUT = '{}/../../OUTPUT/RP'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

dims = r_dims = c_dims = [i for i in range(5, 62, 8)]

tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(r_X), r_X)
tmp = pd.DataFrame(tmp).T
tmp.to_csv('{}/review scree1.csv'.format(OUT))

tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(c_X), c_X)
tmp = pd.DataFrame(tmp).T
tmp.to_csv('{}/cancer scree1.csv'.format(OUT))


tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(r_X)
    tmp[dim][i] = reconstructionError(rp, r_X)
tmp = pd.DataFrame(tmp).T
tmp.to_csv('{}/review scree2.csv'.format(OUT))


tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(c_X)
    tmp[dim][i] = reconstructionError(rp, c_X)
tmp = pd.DataFrame(tmp).T
tmp.to_csv('{}/cancer scree2.csv'.format(OUT))

def main():
    init_decomp = partial(SparseRandomProjection, random_state=5)
    decomp1 = partial(SparseRandomProjection, n_components=5, random_state=5)
    decomp2 = partial(SparseRandomProjection, n_components=30, random_state=5)
    run_dim_alg(r_X, r_y, 'rp', dims, (init_decomp, decomp1), OUT)
    run_dim_alg(c_X, c_y, 'rp', dims, (init_decomp, decomp2), OUT)

if __name__ == '__main__':
    #main()
