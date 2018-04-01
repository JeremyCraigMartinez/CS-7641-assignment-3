from os.path import dirname, realpath
import sys
from time import clock

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.dim_reduction import get_data
from helpers.scoring import metrics
from helpers.constants import ICA_DIMS, SVD_DIMS_R as SVD_DIMS, RP_DIMS, PCA_DIMS
from helpers.figures import bar_plot

PCA = '{}/../OUTPUT/PCA'.format(dir_path)
ICA = '{}/../OUTPUT/ICA'.format(dir_path)
RP = '{}/../OUTPUT/RP'.format(dir_path)
SVD = '{}/../OUTPUT/SVD'.format(dir_path)
BASE = '{}/../OUTPUT/BASE'.format(dir_path)
OUT = '{}/../OUTPUT/NN'.format(dir_path)

data_BASE = get_data(BASE, '', 'r')
data_ICA = [get_data(ICA, '%d-' % i, 'r') for i in ICA_DIMS]
data_RP = [get_data(RP, '%d-' % i, 'r') for i in RP_DIMS]
data_SVD = [get_data(SVD, '%d-' % i, 'r') for i in SVD_DIMS]
data_PCA = [get_data(PCA, '%s-' % i, 'r') for i in PCA_DIMS]

np.random.seed(0)

def get_schwifty(X, y, dataset, dims, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    start = clock()
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    elapsed = clock()-start
    print('%ss for %s at %s dimensions' % (elapsed, dataset, dims))
    return metrics(y_test, pred)

def get_max(alg, dims, data, classifier, results=None):
    max_acc = 0
    max_dim = None
    for i, each in enumerate(data):
        res = get_schwifty(*each, alg, dims[i], classifier)
        if res > max_acc:
            max_acc = res
            max_dim = dims[i]
    results[0].append(max_acc)
    results[1].append('%s-%s' % (alg, max_dim))

def run_it(alg, classifier):
    results = [[get_schwifty(*data_BASE, 'BASE', [100], classifier)], ['BASE']]
    get_max('PCA', PCA_DIMS, data_PCA, classifier, results=results)
    get_max('ICA', ICA_DIMS, data_ICA, classifier, results=results)
    get_max('SVD', SVD_DIMS, data_SVD, classifier, results=results)
    get_max('RP', RP_DIMS, data_RP, classifier, results=results)
    bar_plot(alg, results[0], results[1])

if __name__ == '__main__':
    print('SHOW ME WHAT YOU GOT. I WANT TO SEE WHAT YOU GOT.')
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    run_it('NN', mlp)

    rfc = RandomForestClassifier(criterion='entropy')
    run_it('Random Forest', rfc)
    run_it('Random Forest', rfc)
    run_it('Random Forest', rfc)
    run_it('Random Forest', rfc)
    run_it('Random Forest', rfc)
