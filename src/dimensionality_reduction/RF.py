from os.path import dirname, realpath
import sys
from functools import partial

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.clustering import ImportanceSelect, clusters, dims
from helpers.dim_reduction import run_dim_alg, get_data

#sys.path.insert(0, '/Users/jeremy.martinez/georgia-tech-code/ass3/jontay/src')
OUT = '{}/../../OUTPUT/RF'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
fs_r = rfc.fit(r_X, r_y).feature_importances_
fs_c = rfc.fit(c_X, c_y).feature_importances_

tmp = pd.Series(np.sort(fs_r)[::-1])
tmp.to_csv('{}/review scree.csv'.format(OUT))

tmp = pd.Series(np.sort(fs_c)[::-1])
tmp.to_csv('{}/cancer scree.csv'.format(OUT))

init_decomp = partial(ImportanceSelect, rfc)
decomp1 = partial(ImportanceSelect, rfc, 5)
decomp2 = partial(ImportanceSelect, rfc, 30)
run_dim_alg(r_X, r_y, 'rf', dims, (init_decomp, decomp1), OUT, True)
run_dim_alg(c_X, c_y, 'rf', dims, (init_decomp, decomp2), OUT, True)
