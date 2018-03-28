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

r_dims = c_dims = [i for i in range(16, 80, 7)] # final has 46 dims, no reduction just transformation. Interesting base case

def rpFluctuation(dims, ds):
    tmp = defaultdict(dict)
    for i, dim in product(range(10), dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(r_X), r_X)
    tmp = pd.DataFrame(tmp).T
    tmp.to_csv('{}/{}_comparison.csv'.format(OUT, ds))

'''
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
'''
def main():
    for i, val in enumerate(r_dims): # [0.6, 0.7, 0.8, 0.9]:
        decomp1 = SparseRandomProjection(n_components=val)
        run_dim_alg(r_X, r_y, 'rp', 'reviews', decomp1, val, OUT)
    for i, val in enumerate(c_dims): # [0.6, 0.7, 0.8, 0.9]:
        decomp2 = SparseRandomProjection(n_components=val)
        run_dim_alg(c_X, c_y, 'rp', 'cancer', decomp2, val, OUT)

if __name__ == '__main__':
    if 'fluc' in sys.argv:
        rpFluctuation(r_dims, 'reviews')
        rpFluctuation(c_dims, 'cancer')
    else:
        main()
