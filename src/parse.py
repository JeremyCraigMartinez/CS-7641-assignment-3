from os.path import dirname, realpath
import pandas as pd

from helpers.health_data_preprocessing import get_train_test_set as h_get_data
from helpers.reviews_preprocessing import get_train_test_set as r_get_data

dir_path = dirname(realpath(__file__))

OUT = '{}/../OUTPUT/BASE'.format(dir_path)

h_X, h_y = h_get_data()
r_X, r_y = r_get_data()

def build_hdf(_X, _y, name):
    X = pd.DataFrame(_X)
    y = pd.DataFrame(_y)
    y.columns = ['Class']

    data = pd.concat([X, y], 1)
    data = data.dropna(axis=1, how='all')
    data.to_hdf('{}/datasets.hdf'.format(OUT), name, complib='blosc', complevel=9)

if __name__ == '__main__':
    build_hdf(h_X, h_y, 'cancer')
    build_hdf(r_X, r_y, 'reviews')
