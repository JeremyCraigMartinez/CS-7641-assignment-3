#!/usr/bin/env python
# python-3.6

import sys
import matplotlib.pyplot as plt
import numpy as np

from helpers.read_csvs import read_csv, read_csv_sideways, interpolate_gaps
from helpers.figures import Figures
from helpers.constants import RP_DIMS, SVD_DIMS_R, SVD_DIMS_C

OUT = sys.argv[1] if len(sys.argv) >= 2 else 'BASE'

_plot_colors = ('r', 'g', 'b', 'c', 'darkorange', 'black', 'purple', 'y', 'm', 'ro')

from helpers.clustering import c_clusters

def RP_it_comparsion(title, y_axis, x_axis, _OUT=""):
    def plot(d, _file):
        _f = Figures("%s - %s" % (title, d), y_axis, x_axis)
        _f.start()
        lines = read_csv_sideways('./OUTPUT/{}/{}'.format(_OUT, _file), rows=10)
        its = lines[0]
        lines = lines[1:]
        for i, _ in enumerate(lines):
            l = lines[i]
            _f.plot_curve(l[0], its, interpolate_gaps(l[1]), plot_colors=_plot_colors)
        _f.finish()
    plot('Cancer', 'cancer_comparison.csv')
    plot('Reviews', 'reviews_comparison.csv')

def single_col_compare(_file, title, y_axis, x_axis, it, _OUT=OUT, is_r=False):
    _f = Figures("%s" % (title), y_axis, x_axis)
    _f.start()
    _x = c_clusters
    for i in it:
        if is_r and i == 44:
            _, _, x = read_csv('./OUTPUT/{}/{}/{}'.format(_OUT, i, _file), 3)
        else:
            _, x = read_csv('./OUTPUT/{}/{}/{}'.format(_OUT, i, _file), 2)
        _f.plot_curve(str(i), _x, interpolate_gaps(x), plot_colors=_plot_colors)
    _f.finish()

def compare_between_params(_file, title, y_axis, x_axis, it, _OUT=OUT):
    def plot(d):
        _f = Figures("%s - %s" % (title, d), y_axis, x_axis)
        _f.start()
        _x = c_clusters
        for i in it:
            _, c, r = read_csv('./OUTPUT/{}/{}/{}'.format(_OUT, i, _file), 3)
            _f.plot_curve(str(i), _x, interpolate_gaps(c if d == 'Cancer' else r), plot_colors=_plot_colors)
        _f.finish()
    plot('Cancer')
    plot('Reviews')

def acc_between_params(_file, title, y_axis, x_axis, it, _OUT=OUT):
    def plot(alg):
        _f = Figures("%s %s" % (alg, title), y_axis, x_axis)
        _f.start()
        _x = c_clusters
        for i in it:
            _, g, k = read_csv_sideways('./OUTPUT/{}/{}/{}'.format(_OUT, i, _file))
            _f.plot_curve(str(i), _x, interpolate_gaps(g[1] if alg == 'EM' else k[1]), plot_colors=_plot_colors)
        _f.finish()
    plot('EM')
    plot('KMeans')

def acc_between_params_same(_file, title, y_axis, x_axis, it, _OUT=OUT):
    _f = Figures(title, y_axis, x_axis)
    _f.start()
    def plot(alg):
        _x = c_clusters
        for i in it:
            _, g, k = read_csv_sideways('./OUTPUT/{}/{}/{}'.format(_OUT, i, _file))
            _f.plot_curve(alg + str(i), _x, interpolate_gaps(g[1] if alg == 'EM' else k[1]), plot_colors=_plot_colors)
    plot('EM')
    plot('KMeans')
    _f.finish()

def SVD():
    # SVD SSE
    single_col_compare('SSE.csv', "Cancer - SVD - Sum of Squared Error", "K Clusters", "Squared Error", SVD_DIMS_C, _OUT='SVD')
    single_col_compare('SSE.csv', "Reviews - SVD - Sum of Squared Error", "K Clusters", "Squared Error", SVD_DIMS_R, _OUT='SVD', is_r=True)

    # SVD log likelihood
    # loglitklihood... typo when running clustering
    single_col_compare('loglitklihood.csv', "Cancer - SVD - Log Likelihood", "K Clusters", "Log Likelihood", SVD_DIMS_C, _OUT='SVD')
    single_col_compare('loglitklihood.csv', "Reviews - SVD - Log Likelihood", "K Clusters", "Log Likelihood", SVD_DIMS_R, _OUT='SVD', is_r=True)

def RP():
    # RP SSE
    compare_between_params('SSE.csv', "RP - Sum of Squared Error", "K Clusters", "Squared Error", RP_DIMS, _OUT='RP')
    compare_between_params('logliklihood.csv', "RP - Log Likelihood", "K Clusters", "Log Likelihood", RP_DIMS, _OUT='RP')

    # RP Comparsion
    RP_it_comparsion("Comparsion in Iterations", "Iteration", "PairwiseDistCorr", _OUT='RP')

    acc_between_params('reviews_acc.csv', "Accuracy - RP Reduced - Reviews", "K Clusters", "Accuracy", RP_DIMS, _OUT='RP')
    acc_between_params('cancer_acc.csv', "Accuracy - RP Reduced - Cancer", "K Clusters", "Accuracy", RP_DIMS, _OUT='RP')

def ICA():
    # ICA Mutual Information - special
    acc_between_params_same('reviews_adjMI.csv', "Mutual Information - ICA Reduced - Reviews", "K Clusters/Components", "Adjusted Mutual Information", [13, 21], _OUT='ICA')
    acc_between_params_same('cancer_adjMI.csv', "Mutual Information - ICA Reduced - Cancer", "K Clusters/Components", "Adjusted Mutual Information", [13, 61], _OUT='ICA')

     # ICA Mutual Information
    acc_between_params('reviews_adjMI.csv', "Mutual Information - ICA Reduced - Reviews", "N Components", "Adjusted Mutual Information", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')
    acc_between_params('cancer_adjMI.csv', "Mutual Information - ICA Reduced - Cancer", "N Components", "Adjusted Mutual Information", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')

    # ICA - Kurtosis
    x, rk = read_csv('./OUTPUT/{}/reviews kurtosis.csv'.format(OUT), 2)
    x, ck = read_csv('./OUTPUT/{}/cancer kurtosis.csv'.format(OUT), 2)
    f = Figures("Kurtosis", "N Components", "Average Component Kurtosis")
    f.start()
    f.plot_curve("cancer", x, ck, plot_colors=_plot_colors)
    f.plot_curve("reviews", x, rk, plot_colors=_plot_colors)
    f.finish()

    # ICA Accuracy
    acc_between_params('reviews_acc.csv', "Accuracy - ICA Reduced - Reviews", "K Clusters", "Accuracy", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')
    acc_between_params('cancer_acc.csv', "Accuracy - ICA Reduced - Cancer", "K Clusters", "Accuracy", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')

    # SSE
    compare_between_params('SSE.csv', "ICA - Sum of Squared Error", "K Clusters", "Squared Error", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')
    compare_between_params('logliklihood.csv', "ICA - Log Likelihood", "K Clusters", "Log Likelihood", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')

def PCA(): # pylint: disable=R0915
    acc_between_params('reviews_acc.csv', "PCA Reduced - KMeans Accuracy - Reviews", "K Clusters", "Accuracy", [0.6, 0.7, 0.8, 0.9], _OUT='PCA')
    acc_between_params('cancer_acc.csv', "PCA Reduced - KMeans Accuracy - Cancer", "K Clusters", "Accuracy", [0.6, 0.7, 0.8, 0.9], _OUT='PCA')

    # PCA Mutual Information
    acc_between_params('reviews_adjMI.csv', "Mutual Information - PCA Reduced - Reviews", "K Clusters", "Accuracy", [0.6, 0.7, 0.8, 0.9], _OUT='PCA')
    acc_between_params('cancer_adjMI.csv', "Mutual Information - PCA Reduced - Cancer", "K Clusters", "Accuracy", [0.6, 0.7, 0.8, 0.9], _OUT='PCA')

    # PCA SSE
    compare_between_params('SSE.csv', "PCA - Sum of Squared Error", "K Clusters", "Squared Error", [0.6, 0.7, 0.8, 0.9], _OUT='PCA')
    compare_between_params('logliklihood.csv', "PCA - Log Likelihood", "K Clusters", "Log Likelihood", [0.6, 0.7, 0.8, 0.9], _OUT='PCA')

    compare_between_params('SSE.csv', "ICA - Sum of Squared Error", "K Clusters", "Squared Error", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')
    compare_between_params('logliklihood.csv', "ICA - Log Likelihood", "K Clusters", "Log Likelihood", [5, 13, 21, 29, 37, 45, 53, 61], _OUT='ICA')

    # EM Accuracy - Reviews - PCA
    x, r_gmm_6, _ = read_csv_sideways('./OUTPUT/{}/0.6/reviews_acc.csv'.format(OUT))
    x, r_gmm_7, _ = read_csv_sideways('./OUTPUT/{}/0.7/reviews_acc.csv'.format(OUT))
    x, r_gmm_8, _ = read_csv_sideways('./OUTPUT/{}/0.8/reviews_acc.csv'.format(OUT))
    x, r_gmm_9, _ = read_csv_sideways('./OUTPUT/{}/0.9/reviews_acc.csv'.format(OUT))
    f = Figures("PCA Reduced - EM Accuracy - Reviews", "K Clusters", "Accuracy")
    f.start()

    f.plot_curve("0.6", x, interpolate_gaps(r_gmm_6[1]), plot_colors=_plot_colors)
    f.plot_curve("0.7", x, interpolate_gaps(r_gmm_7[1]), plot_colors=_plot_colors)
    f.plot_curve("0.8", x, interpolate_gaps(r_gmm_8[1]), plot_colors=_plot_colors)
    f.plot_curve("0.9", x, interpolate_gaps(r_gmm_9[1]), plot_colors=_plot_colors)
    f.finish()

    # EM Accuracy - Reviews - PCA
    x, c_gmm_6, _ = read_csv_sideways('./OUTPUT/{}/0.6/cancer_acc.csv'.format(OUT))
    x, c_gmm_7, _ = read_csv_sideways('./OUTPUT/{}/0.7/cancer_acc.csv'.format(OUT))
    x, c_gmm_8, _ = read_csv_sideways('./OUTPUT/{}/0.8/cancer_acc.csv'.format(OUT))
    x, c_gmm_9, _ = read_csv_sideways('./OUTPUT/{}/0.9/cancer_acc.csv'.format(OUT))
    f = Figures("PCA Reduced - EM Accuracy - Cancer", "K Clusters", "Accuracy")
    f.start()

    x = [i for i in range(1, 78, 6)]
    f.plot_curve("0.6", x, interpolate_gaps(c_gmm_6[1][:-5]), plot_colors=_plot_colors)
    f.plot_curve("0.7", x, interpolate_gaps(c_gmm_7[1][:-5]), plot_colors=_plot_colors)
    f.plot_curve("0.8", x, interpolate_gaps(c_gmm_8[1][:-5]), plot_colors=_plot_colors)
    f.plot_curve("0.9", x, interpolate_gaps(c_gmm_9[1][:-5]), plot_colors=_plot_colors)
    f.finish()

    # KMeans Accuracy - Reviews - PCA
    x, _, r_km_6 = read_csv_sideways('./OUTPUT/{}/0.6/reviews_acc.csv'.format(OUT))
    x, _, r_km_7 = read_csv_sideways('./OUTPUT/{}/0.7/reviews_acc.csv'.format(OUT))
    x, _, r_km_8 = read_csv_sideways('./OUTPUT/{}/0.8/reviews_acc.csv'.format(OUT))
    x, _, r_km_9 = read_csv_sideways('./OUTPUT/{}/0.9/reviews_acc.csv'.format(OUT))
    f = Figures("PCA Reduced - KMeans Accuracy - Reviews", "K Clusters", "Accuracy")
    f.start()

    f.plot_curve("0.6", x, interpolate_gaps(r_km_6[1]), plot_colors=_plot_colors)
    f.plot_curve("0.7", x, interpolate_gaps(r_km_7[1]), plot_colors=_plot_colors)
    f.plot_curve("0.8", x, interpolate_gaps(r_km_8[1]), plot_colors=_plot_colors)
    f.plot_curve("0.9", x, interpolate_gaps(r_km_9[1]), plot_colors=_plot_colors)
    f.finish()

    # KMeans Accuracy - Reviews - PCA
    x, _, c_km_6 = read_csv_sideways('./OUTPUT/{}/0.6/cancer_acc.csv'.format(OUT))
    x, _, c_km_7 = read_csv_sideways('./OUTPUT/{}/0.7/cancer_acc.csv'.format(OUT))
    x, _, c_km_8 = read_csv_sideways('./OUTPUT/{}/0.8/cancer_acc.csv'.format(OUT))
    x, _, c_km_9 = read_csv_sideways('./OUTPUT/{}/0.9/cancer_acc.csv'.format(OUT))
    f = Figures("PCA Reduced - KMeans Accuracy - Cancer", "K Clusters", "Accuracy")
    f.start()

    x = [i for i in range(1, 78, 6)]
    f.plot_curve("0.6", x, interpolate_gaps(c_km_6[1][:-5]), plot_colors=_plot_colors)
    f.plot_curve("0.7", x, interpolate_gaps(c_km_7[1][:-5]), plot_colors=_plot_colors)
    f.plot_curve("0.8", x, interpolate_gaps(c_km_8[1][:-5]), plot_colors=_plot_colors)
    f.plot_curve("0.9", x, interpolate_gaps(c_km_9[1][:-5]), plot_colors=_plot_colors)
    f.finish()

def DEFAULT(): # pylint: disable=R0915
    # GMM Accuracy - Cancer
    x, c_gmm, c_km = read_csv_sideways('./OUTPUT/{}/cancer_acc.csv'.format(OUT))
    f = Figures("EM Accuracy - Cancer", "K Clusters", "Accuracy")
    f.start()
    f.plot_curve("cancer", x, c_gmm[1])
    f.finish()

    # GMM Accuracy - Reviews
    x, r_gmm, r_km = read_csv_sideways('./OUTPUT/{}/reviews_acc.csv'.format(OUT))
    f = Figures("EM Accuracy - Reviews", "K Clusters", "Accuracy")
    f.start()
    f.plot_curve("reviews", x, r_gmm[1])
    f.finish()

    # Log likelihood
    x, c_y, r_y = read_csv('./OUTPUT/{}/logliklihood.csv'.format(OUT), 3)
    f = Figures("EM Log Likelihood", "K Clusters", "Log Likelihood")
    f.start()
    f.plot_curve("cancer", x, c_y)
    f.plot_curve("reviews", x, r_y)
    f.finish()

    # KMeans Accuracy - Reviews
    x, gmm, r_km = read_csv_sideways('./OUTPUT/{}/reviews_acc.csv'.format(OUT))
    f = Figures("KMeans Accuracy - Reviews", "K Clusters", "Accuracy")
    f.start()
    f.plot_curve("reviews", x, r_km[1])
    f.finish()

    # KMeans Accuracy - Cancer
    x, gmm, c_km = read_csv_sideways('./OUTPUT/{}/cancer_acc.csv'.format(OUT))
    f = Figures("KMeans Accuracy - Cancer", "K Clusters", "Accuracy")
    f.start()
    f.plot_curve("cancer", x, c_km[1])
    f.finish()

    # SSE
    x, c_y, r_y = read_csv('./OUTPUT/{}/SSE.csv'.format(OUT), 3)
    f = Figures("Sum of Squared Error", "K Clusters", "Squared Error")
    f.start()
    f.plot_curve("cancer", x, c_y)
    f.plot_curve("reviews", x, r_y)
    f.finish()


    x, gmm, km = read_csv_sideways('./OUTPUT/{}/reviews_adjMI.csv'.format(OUT))
    f = Figures("Reviews Mutual Information", "K Clusters", "Squared Error")
    f.start()
    f.plot_curve(km[0], x, np.isfinite(km[1]))
    f.plot_curve(gmm[0], x, np.isfinite(gmm[1]))
    f.finish()

    _, x, y, label = read_csv('./OUTPUT/{}/reviews.csv'.format(OUT), 4)
    f = Figures("Reviews Mutual Information", "K Clusters", "Squared Error")

    x1 = [x[i] for i, l in enumerate(x) if label[i] == 1]
    y1 = [y[i] for i, l in enumerate(y) if label[i] == 1]
    x0 = [x[i] for i, l in enumerate(x) if label[i] == 0]
    y0 = [y[i] for i, l in enumerate(y) if label[i] == 0]
    f.start()
    f.plot_curve('positive', x1, y1)
    f.plot_curve('negative', x0, y0)
    f.finish()

    _, x, y, label = read_csv('./OUTPUT/{}/cancer.csv'.format(OUT), 4)
    f = Figures("Cancer Mutual Information", "K Clusters", "Squared Error")

    x1 = [x[i] for i, l in enumerate(x) if label[i] == 1]
    y1 = [y[i] for i, l in enumerate(y) if label[i] == 1]
    x0 = [x[i] for i, l in enumerate(x) if label[i] == 0]
    y0 = [y[i] for i, l in enumerate(y) if label[i] == 0]
    f.start()
    f.plot_curve('positive', x1, y1)
    f.plot_curve('negative', x0, y0)
    f.finish()

if __name__ == '__main__':
    if 'SVD' in sys.argv:
        SVD()
    if 'RP' in sys.argv:
        RP()
    if 'ICA' in sys.argv:
        ICA()
    if 'PCA' in sys.argv:
        PCA()
    if 'DEFAULT' in sys.argv:
        DEFAULT()

    plt.show()
