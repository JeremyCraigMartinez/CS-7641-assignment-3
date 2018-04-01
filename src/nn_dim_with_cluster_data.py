from os.path import dirname, realpath
import sys
from time import clock
from itertools import product

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.dim_reduction import get_data
from helpers.scoring import metrics
from helpers.constants import ICA_DIMS, SVD_DIMS_R as SVD_DIMS, RP_DIMS, PCA_DIMS
from helpers.figures import bar_plot
from helpers.clustering import r_clusters

PCA = '{}/../OUTPUT/PCA/{}'.format(dir_path, 'datasets-w-cluster')
ICA = '{}/../OUTPUT/ICA/{}'.format(dir_path, 'datasets-w-cluster')
RP = '{}/../OUTPUT/RP/{}'.format(dir_path, 'datasets-w-cluster')
SVD = '{}/../OUTPUT/SVD/{}'.format(dir_path, 'datasets-w-cluster')
BASE = '{}/../OUTPUT/BASE'.format(dir_path)
OUT = '{}/../OUTPUT/NN'.format(dir_path)

data_BASE = get_data(BASE, prefix='', ds='r')
data_ICA = []#[get_data(ICA, ds='r', suffix='%s-%s-datasets.km.hdf') for d, c in product(ICA_DIMS, r_clusters)]
data_RP = []#[get_data(RP, ds='r', suffix='%s-%s-datasets.km.hdf') for d, c in product(RP_DIMS, r_clusters)]
data_SVD = []#[get_data(SVD, ds='r', suffix='%s-%s-datasets.km.hdf') for d, c in product(SVD_DIMS, r_clusters)]

data_PCA_KM = [(get_data(PCA, ds='r', suffix='%s-%s-datasets.km.hdf' % (c, d)), '%s-%s-km' % (c, d)) for d, c in product(PCA_DIMS, r_clusters)]
data_PCA_GMM = [(get_data(PCA, ds='r', suffix='%s-%s-datasets.gmm.hdf' % (c, d)), '%s-%s-gmm' % (c, d)) for d, c in product(PCA_DIMS, r_clusters)]

data_ICA_KM = [(get_data(ICA, ds='r', suffix='%s-%s-datasets.km.hdf' % (c, d)), '%s-%s-km' % (c, d)) for d, c in product(ICA_DIMS, r_clusters)]
data_ICA_GMM = [(get_data(ICA, ds='r', suffix='%s-%s-datasets.gmm.hdf' % (c, d)), '%s-%s-gmm' % (c, d)) for d, c in product(ICA_DIMS, r_clusters)]

data_SVD_KM = [(get_data(SVD, ds='r', suffix='%s-%s-datasets.km.hdf' % (c, d)), '%s-%s-km' % (c, d)) for d, c in product(SVD_DIMS, r_clusters)]
data_SVD_GMM = [(get_data(SVD, ds='r', suffix='%s-%s-datasets.gmm.hdf' % (c, d)), '%s-%s-gmm' % (c, d)) for d, c in product(SVD_DIMS, r_clusters)]

data_RP_KM = [(get_data(RP, ds='r', suffix='%s-%s-datasets.km.hdf' % (c, d)), '%s-%s-km' % (c, d)) for d, c in product(RP_DIMS, r_clusters)]
data_RP_GMM = [(get_data(RP, ds='r', suffix='%s-%s-datasets.gmm.hdf' % (c, d)), '%s-%s-gmm' % (c, d)) for d, c in product(RP_DIMS, r_clusters)]

np.random.seed(0)

def get_schwifty(X, y, dataset, dims, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    start = clock()
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    elapsed = clock()-start
    print('%ss for %s at %s dimensions' % (elapsed, dataset, dims))
    return metrics(y_test, pred)

def get_max(alg, data, classifier, results=None):
    max_acc = 0
    max_args = None

    for _, each in enumerate(data):
        res = get_schwifty(*each[0], alg, each[1], classifier)
        if res > max_acc:
            max_acc = res
            max_args = '%s\n%s' % (alg, each[1])
    results[0].append(max_acc)
    results[1].append(max_args)

def run_it(alg, classifier):
    results = [[get_schwifty(*data_BASE, 'BASE', [100], classifier)], ['BASE']]
    get_max('PCA', data_PCA_KM, classifier, results=results)
    get_max('PCA', data_PCA_GMM, classifier, results=results)
    get_max('ICA', data_ICA_KM, classifier, results=results)
    get_max('ICA', data_ICA_GMM, classifier, results=results)
    get_max('SVD', data_SVD_KM, classifier, results=results)
    get_max('SVD', data_SVD_GMM, classifier, results=results)
    get_max('RP', data_RP_KM, classifier, results=results)
    get_max('RP', data_RP_GMM, classifier, results=results)
    print(results)
    bar_plot('%s - with Cluster as Attribute' % alg, results[0], results[1])

if __name__ == '__main__':
    print('SHOW ME WHAT YOU GOT. I WANT TO SEE WHAT YOU GOT.')
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    run_it('NN', mlp)
