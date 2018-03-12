from os.path import dirname, realpath
import pandas as pd

from helpers.health_data_preprocessing import get_train_test_set as h_get_data
from helpers.reviews_preprocessing import get_train_test_set as r_get_data

dir_path = dirname(realpath(__file__))

OUT = '{}/../OUTPUT/BASE'.format(dir_path)

h_train_X, h_test_X, h_train_y, h_test_y = h_get_data()
r_train_X, r_test_X, r_train_y, r_test_y = r_get_data()

def build_hdf(_X, _y, name):
    X = pd.DataFrame(_X)
    y = pd.DataFrame(_y)
    y.columns = ['Class']

    data = pd.concat([X, y], 1)
    data = data.dropna(axis=1, how='all')
    data.to_hdf('{}/{}_datasets.hdf'.format(OUT, name), name, complib='blosc', complevel=9)

if __name__ == '__main__':
    build_hdf(h_train_X, h_train_y, 'cancer')
    build_hdf(h_test_X, h_test_y, 'cancer_test')

    build_hdf(r_train_X, r_train_y, 'reviews')
    build_hdf(r_test_X, r_test_y, 'reviews_test')
