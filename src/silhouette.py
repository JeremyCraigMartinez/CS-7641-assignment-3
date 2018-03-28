# code found at http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

from __future__ import print_function
from os.path import dirname, realpath
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from helpers.dim_reduction import get_data

print(__doc__)

dir_path = dirname(realpath(__file__))
output_dir = sys.argv[1] if len(sys.argv) >= 2 else 'BASE'
OUTPUT = '{}/../OUTPUT'.format(dir_path)
OUT = '{}/{}'.format(OUTPUT, output_dir)
#BASE = '{}/../OUTPUT/BASE'.format(dir_path)
BASE = OUT

r_components = [8, 13, 21, 34, 55, 89, 104, 119, 134, 159]
c_components = [8, 10, 14, 18, 25, 35, 45, 55, 65, 75]
range_n_clusters = c_components

def main(X, ds="", xtra_title=""):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            #ax1.text(-.35, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("%s Silhouette plot (%d clusters)%s" % (ds, n_clusters, xtra_title))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()

def ICA():
    global range_n_clusters
    range_n_clusters = [i for i in range(5, 62, 8)]
    _r, _c = get_data('{}/ICA'.format(OUTPUT), '37-')
    c_X, _ = _c
    main(c_X)
    _r, _c = get_data('{}/ICA'.format(OUTPUT), '45-')
    r_X, _ = _r
    main(r_X)

def RP():
    global range_n_clusters
    range_n_clusters = [16, 30, 45]
    def runitc(p):
        _r, _c = get_data('{}/RP'.format(OUTPUT), '%s-' % p)
        c_X, _ = _c
        print('Cancer at %s dimensions' % p)
        main(c_X, 'Cancer', ' (%s dims)' % p)
    [runitc(i) for i in ['16', '23', '30', '37', '44']]
    def runitr(p):
        _r, _c = get_data('{}/RP'.format(OUTPUT), '%s-' % p)
        r_X, _ = _r
        print('Reviews at %s dimensions' % p)
        main(r_X, 'Reviews', ' (%s dims)' % p)
    [runitr(i) for i in ['44', '51', '65', '72', '79']]

def PCA():
    def runit(p):
        _r, _c = get_data('{}/PCA'.format(OUTPUT), '%s-' % p)
        r_X, _ = _r
        c_X, _ = _c
        #main(r_X)
        main(c_X)
    [runit(i) for i in ['0.6']]#, '0.7', '0.8', '0.9']]

if __name__ == '__main__':
    if sys.argv[1] == 'ICA':
        ICA()
    if sys.argv[1] == 'RP':
        RP()
    elif sys.argv[1] == 'PCA':
        PCA()
