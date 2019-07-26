__author__ = 'dedwards'

import numpy as np
from itertools import product, groupby

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.io import show, output_file
from bokeh.palettes import Spectral11, BrBG11


def rows_distance(X, v):
    return np.sqrt((np.apply_along_axis(lambda x: x - v, 1, X)**2).sum(axis=1))


def db_index(X, c):
    # Put the data in cluster order
    order = list(zip(*sorted(enumerate(c), cmp=lambda a, b: cmp(a[1], b[1])))[0])
    X = X[order, :]
    c = c[order]

    # Calculate the centroids and average distances in each cluster
    centroids = []
    ave_dists = []
    for _, rows in groupby(enumerate(list(X)), lambda z: c[z[0]]):
        Xs = np.array([row for _, row in rows])
        cent = Xs.mean(axis=0)
        ave_dists.append(rows_distance(Xs, cent).mean())
        centroids.append(cent)
    centroids = np.array(centroids)

    # Calculate the distances between each cluster
    dmat = np.array([rows_distance(centroids, v) for v in list(centroids)])

    # Construct the indicator
    nc = len(ave_dists)
    amat = np.matrix([a + b for a, b in product(ave_dists, ave_dists)]).reshape((nc, nc))
    mat = amat / dmat
    np.fill_diagonal(mat, 0.)

    return -mat.max(axis=1).mean()  # Negate so that maximum is best


def choose_clustering(X, clust_class, options, metric='dbi', plot=False):

    exp_options = list(product(*[[(k, vv) for vv in v] for k, v in options.iteritems()]))

    scores = []
    best_clusters = None
    best_opts = None
    best_score = -np.inf
    for opts in exp_options:
        clust_obj = clust_class(**dict(opts))
        clusters = clust_obj.fit_predict(X)
        try:
            sh = silhouette_score(X, clusters)
            dbi = db_index(X, clusters)
        except ValueError:
            sh = 0.
            dbi = 0.

        scores.append((dbi, sh))
        score = dbi if metric == 'dbi' else sh
        if score > best_score:
            best_score = score
            best_opts = opts
            best_clusters = clusters.copy()

    if plot and len(best_opts) == 1:
        opt_n = best_opts[0][0]
        opt_v = [opts[0][1] for opts in exp_options]
        best_opt_v = best_opts[0][1]
        scores = zip(*scores)
        s = ColumnDataSource({opt_n: opt_v, 'dbi': scores[0], 'silhouette': scores[1]})
        p = figure(width=500, height=300)
        for ii, metric in enumerate(['silhouette', 'dbi']):
            colour = Spectral11[ii]
            p.line(x=opt_n, y=metric, color=colour, legend=metric, source=s)
            p.scatter(x=opt_n, y=metric, color=colour, legend=metric, source=s)
        min_score = np.array(scores).min()
        max_score = np.array(scores).max()
        p.line([best_opt_v, best_opt_v], [min_score, max_score], color='red')
    else:
        p = None

    return best_score, best_clusters, best_opts, p


if __name__ == '__main__':
    from numpy.random import multivariate_normal
    X = np.concatenate([multivariate_normal([5, 5], [[1., 0.4], [0.4, 1.]], 100),
                        multivariate_normal([-3, 7], [[2., -0.1], [-0.1, 0.5]], 50),
                        multivariate_normal([-6, -4], [[3., 0.6], [0.6, 5.]], 100),
                        multivariate_normal([6, -1], [[0.1, 0.2], [0.2, 3.1]], 200)])

    def Cluster(**kwargs):
        return AgglomerativeClustering(linkage='ward', **kwargs)
    score, clusters, opts, p0 = choose_clustering(X, Cluster, {'n_clusters': range(2, 15)},
                                                  plot=True, metric='dbi')

    output_file('/tmp/cluster.html')
    s = ColumnDataSource({'x': X[:, 0], 'y': X[:, 1], 'c': [(Spectral11 + BrBG11)[ii] for ii in clusters]})
    p = figure(width=500, height=500)
    p.scatter(x='x', y='y', color='c', source=s)
    show(column(p, p0))

    print dict(opts)