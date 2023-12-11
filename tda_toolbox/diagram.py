"""
Generate the diagram of a given neuron
"""

import os.path
import re
import subprocess
from struct import pack, unpack, calcsize

import dionysus as di
import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
#import phstuff.barcode as bc
import tmd
from scipy import stats
from scipy.linalg import norm, eigh
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn.decomposition import PCA


def RipsFiltration(neuron_file, k=2, r=.5):
    """
    Compute the rips filtration of a neuron with dionysus.
    Requires dionysus library
    :param neuron_file:
    :param k: max skeleton dimension
    :param r: compute the Ribs Filtration with max value = r * max distance between points, r \in [0, 1]
    :return:
    """
    nrn = tmd.io.load_neuron(neuron_file)

    points = np.concatenate([[np.array([x, y, z]) for x, y, z in zip(t.x, t.y, t.z)] for t in nrn.neurites])
    D = pdist(points)

    return di.fill_rips(squareform(D), k, r * D.max())


def RipserFiltration(neuron_file, k=2, threshold=None, output=None, forced=False):
    """
    Compute the rips filtration of a neuron with ripser.
    Requires ripser executable at the root of the project.
    :param neuron_file:
    :param k: max skeleton dimension
    :param r: compute the Ribs Filtration with max value = r * max distance between points, r \in [0, 1]
    :return:
    """

    if output is None:
        output = '/tmp/filtration.txt'
        forced = True

    if not os.path.exists(output) or forced:
        nrn = tmd.io.load_neuron(neuron_file)
        point_cloud = '/tmp/neuron.xyz'

        with open(point_cloud, 'w') as f:
            for t in nrn.neurites:
                f.writelines(["{}, {}, {}\n".format(x, y, z) for x, y, z in zip(t.x, t.y, t.z)])

        with open(point_cloud, 'r') as stdin:
            with open('/tmp/filtration.txt', 'w') as stdout:
                args = ["./ripser", "--format", "point-cloud", "--dim", str(k)]
                if threshold is not None:
                    args.extend(["--threshold", threshold])
                subprocess.check_call(args, stdin=stdin, stdout=stdout)
        if output != '/tmp/filtration.txt':
            subprocess.check_call(["cp", '/tmp/filtration.txt', output])

    dgms, dgm = [], None

    with open(output, 'r') as out:
        pattern = re.compile('^persistence intervals in dim ([0-9]+):$')
        bar_pattern = re.compile('^ \[([0-9.]+),([0-9.]*| )\)$')
        for l in out.readlines():
            if pattern.match(l):
                if dgm is not None:
                    dgms.append(dgm)
                dgm = []
            else:
                match = bar_pattern.match(l)
                if match:
                    dgm.append((float(match.group(1)), float(match.group(2)) if match.group(2) != " " else np.inf))
        if dgm is not None:
            dgms.append(dgm)

    return dgms


def dipha(neuron_file, k=2, output=None, forced=False):
    """
    Compute the alpha filtration of a neuron and return its persistence diagram.
    Requires dipha executable at the root of the project.
    :param neuron_file:
    :param k: max skeleton dimension
    :return:
    """

    if output is None:
        output = '/tmp/filtration.txt'
        forced = True

    if not os.path.exists(output) or forced:
        nrn = tmd.io.load_neuron(neuron_file)
        distance_matrix = '/tmp/distance_matrix.txt'

        points = []
        for t in nrn.neurites:
            points.extend(zip(t.x, t.y, t.z))

        D = squareform(pdist(points))

        with open(distance_matrix, 'wb') as f:
            f.write(pack('<qqq', 8067171840, 7, len(D)))
            f.writelines([pack('<' + 'd' * len(row), *row) for row in D])

        args = ["./dipha", "--upper_dim", str(k), distance_matrix, "/tmp/filtration.txt"]
        subprocess.check_call(args)

        if output != '/tmp/filtration.txt':
            subprocess.check_call(["cp", '/tmp/filtration.txt', output])

    dgms = {}

    with open(output, 'rb') as f:
        dipha_identifier, = unpack('<q', f.read(calcsize('q')))
        assert dipha_identifier == 8067171840, "Wrong dipha file"
        diagram_identifier, = unpack('<q', f.read(calcsize('q')))
        assert diagram_identifier == 2, "input is not a persistence_diagram file"
        num_pairs, = unpack('<q', f.read(calcsize('q')))

        for i in range(num_pairs):
            dim, birth, death = unpack('<qdd', f.read(calcsize('<q') + 2 * calcsize('<d')))

            if dim < 0:
                dim = -dim - 1
            dgms.setdefault(dim, []).append((birth, death))

    return dgms


def save_filtration(file, f):
    """
    Save a filtration to a file
    :param file:
    :param f:
    :return:
    """
    return np.save(file, [([v for v in s], s.data) for s in f])


def load_filtration(file):
    """
    Load a filtration
    :param file:
    :return:
    """
    f = di.Filtration()
    simplices = np.load(file)
    for vertices, data in simplices:
        f.append(di.Simplex(vertices, data))
    return f


def save_diagram(file, dgm):
    """
    Save a diagram in numpy format
    :param file:
    :param dgm:
    :return:
    """
    return np.save(file, [(bar.birth, bar.death) for bar in dgm])


def load_diagram(file, dgm):
    """
    Load a diagram saved as a list of tuples in numpy format
    :param file:
    :param dgm:
    :return:
    """
    barcode = np.load(file)
    return di.Diagram(barcode)


def TMD(neuron_handle):
    """
    Compute the TMD of a neuron i.e. its dimension 0 persistence
    using path distance as a filtration
    :param neuron_handle:
    :return:
    """
    nrn = nm.load_neuron(neuron_handle)
    diag = []
    roots = []
    f = {None: 0}

    nodes = [neurite.root_node for neurite in nm.iter_neurites(nrn)]
    while len(nodes) > 0:
        n = nodes.pop()
        f[n] = f[n.parent] + n.length
        nodes.extend(n.children)

    for neurite in nm.iter_neurites(nrn):
        R = neurite.root_node  # the root of the neuron tree
        A = {}  # A set of active nodes, initialized to the set of tree leaves
        for l in R.ileaf():
            A[l] = f[l]

        while not R in A:
            for l in A:
                p = l.parent
                # break if a child of p is not active, might need to improve this to avoid quadratic complexity
                stop = False
                m = 0
                c0 = None
                for c in p.children:
                    if not c in A:
                        stop = True
                        break
                    elif A[c] > m:
                        m = A[c]
                        c0 = c
                if not stop:
                    A[p] = m
                    for c in p.children:
                        if not c == c0:
                            diag.append((min(A[c], f[p]), max(A[c], f[p])))
                        A.pop(c)
                    break

        roots.append((min(A[R], f[R]), max(A[R], f[R])))

    # merge root of dendrites
    i = np.argmax(map(lambda a, b: b, roots))
    a, b = roots.pop(i)
    diag.extend(roots)
    diag.append((b,))

    return diag


# def tmd_to_barcode(tmd):
#     """
#     transform a set of tuples to a set of bc.Interval
#     :param tmd:
#     :return:
#     """
#     barcode = []
#     for bar in tmd:
#         if len(bar) == 1:
#             barcode.append(bc.Interval(bar[0]))
#         else:
#             barcode.append(bc.Interval(bar[0], bar[1]))
#     return barcode


def d(bar1, bar2=None):
    if bar2 is None:
        return (bar1[1] - bar1[0]) / 2 if len(bar1) == 2 else 2 ** 32
    else:
        return 2 ** 32 if len(bar1) == 1 or len(bar2) == 1 else max(abs(bar1[1] - bar2[1]),
                                                                    abs(bar1[0] - bar2[0]))


def signature(barcode, n):
    """
    Compute the signature of a neuron
    https://geometrica.saclay.inria.fr/team/Steve.Oudot/papers/coo-stbp3ds-15/coo-stbp3ds-15.pdf
    :param barcode:
    :param n:
    :return: a vector of dimension n(n+1)/2
    """
    # sort the bar length in decreasing order
    lengths = sorted(list(map(d, barcode)), reverse=True)
    sign = lengths[0:n]  # take only the n first bars
    sign.extend(np.zeros(n - len(sign)))  # complete with zeros

    # compute the modified distance matrix, where d[i,j] = min(d(x,y), d(x), d(y))
    distances = np.zeros([len(barcode), len(barcode)])
    for i in np.arange(0, len(barcode), 1):
        for j in np.arange(0, len(barcode), 1):
            distances[i, j] = min(d(barcode[i], barcode[j]), min(d(barcode[i]), d(barcode[j])))
    distances = sorted(distances.reshape(len(barcode) * len(barcode)), reverse=True)

    # fill sign with the n(n-1)/2 largest entries of d
    sign.extend(distances[0:n * (n - 1) // 2])
    # complete with zeros to get an n*(n+1)/2 vector
    sign.extend(np.zeros(n * (n + 1) // 2 - len(sign)))

    return sign


def min_max(barcode):
    """
    return the min birth and the max death of a barcode
    :param barcode:
    :return:
    """
    m, M = np.inf, -np.inf
    for bar in barcode:
        if bar[0] < m:
            m = bar[0]
        if len(bar) > 1 and bar[1] > M:
            M = bar[1]
    return m, M


def f(bar, t):
    if len(bar) == 1:
        return max(0, bar[0] - t)
    else:
        return min(max(0, bar[0] - t), max(0, t - bar[1]))


def landscape(barcode, k, m):
    """
    compute the toplogical landscape of a barcode
    :param barcode:
    :param k:
    :param m:
    :return:
    """
    b, d = min_max(barcode)
    sign = []
    for t in np.linspace(b, d, m):
        beta_t = sorted(map(lambda bar: f(bar, t), barcode), reverse=True)[0:k]
        beta_t.extend(np.zeros(k - len(beta_t)))
        sign.extend(beta_t)
    return np.array(sign)


def finitize(barcode):
    """
    Replace all infinite death in the barcode to the maximal finite death
    :param barcode:
    :return:
    """
    M = -np.inf
    inf_ix = []

    for i, (x, y) in enumerate(barcode):
        if y < np.inf:
            if y > M:
                M = y
        else:
            inf_ix.append(i)

    for i in inf_ix:
        barcode[i][1] = M


def gaussian_image(ph, xlims=None, ylims=None, bins=100j):
    """
    Transform a barcode into a gaussian image
    """
    if xlims is None:
        xlims = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    if ylims is None:
        ylims = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]

    X, Y = np.mgrid[xlims[0]:xlims[1]:bins, ylims[0]:ylims[1]:bins]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    return Z


def pca(signatures, n=3):
    """
    Apply pca to a set of signatures
    :param signatures:
    :param n:
    :return:
    """
    pca = PCA(n_components=n)
    return pca.fit_transform(signatures)


def dot(x, dir):
    """
    Finite dot product between a possibly infinite tuple x and dir
    :param x:
    :param dir:
    :return:
    """
    return 2 ** 31 if len(x) == 1 else x[0] * dir[0] + x[1] * dir[1]


def orth_proj(x, dir):
    """
    Projection of x onto dir
    :param x:
    :param dir:
    :return:
    """
    return dot(x, dir) * dir


def gram_matrix(D):
    """
    Compute a gram matrix X^T * X out of a distance matrix D
    :param D:
    :return:
    """
    X_i0 = np.matmul(D, np.matrix([[1] + [0] * (len(D) - 1), ] * len(D)).T)
    X_0j = np.matmul(np.matrix([[1] + [0] * (len(D) - 1), ] * len(D)), D)
    return .5 * (X_i0 + X_0j - D)


def cosine_kernel(D, smoothing=10 ** -3):
    """
    Compute a cosine kernel
    :param D:
    :param smoothing:
    :return:
    """
    X_ii = np.diag(1 / np.sqrt(smoothing + D[0, :]))
    G = gram_matrix(D)
    return X_ii * G * X_ii


def polynomial_kernel(D, gamma, p):
    """
    Compute a polynomial kernel k(x, y) = (1 + \gamma * x.dot(y) )^p
    :param D:
    :param gamma:
    :param p:
    :return:
    """
    G = gram_matrix(D)
    return np.power(1 + gamma * G, p)


def sigmoid_kernel(D, gamma):
    """
    Compute a sigmoid kernel k(x, y) = tanh(1 + gamma * x.dot(y) )
    :param D:
    :param gamma:
    :return:
    """
    G = gram_matrix(D)
    return np.tanh(1 + gamma * G)


def linear_kernel(D):
    """
    Compute a linear kernel k(x, y) = x.dot(y)
    :param D:
    :return:
    """
    return gram_matrix(D)


def gaussian_kernel(D, sigma):
    """
    Compute a gaussian kernel with dispersion sigma from the given distance matrix D
    :param D:
    :param sigma:
    :return:
    """
    return np.exp(-D / (2 * sigma ** 2))


def wasserstein_distance(barcode1, barcode2):
    dg1, dg2 = barcode_to_diagram(barcode1), barcode_to_diagram(barcode2)
    return di.wasserstein_distance(dg1, dg2)


def bottleneck_distance(barcode1, barcode2, delta=0.01):
    dg1, dg2 = barcode_to_diagram(barcode1), barcode_to_diagram(barcode2)
    return di.bottleneck_distance(dg1, dg2, delta)


def cluster_score(X, y, true_labels=None, _metric="silhouette_score", **kwds):
    """
    A score to compare clusterings.
    If you know the true labels, use fowlkes_mallows_score. Otherwise use silhouette_score.
    """
    # Unsupervised metrics
    if _metric == 'calinski_harabaz_score':
        return metrics.calinski_harabaz_score(X, y)
    elif _metric == 'silhouette_score':
        return metrics.silhouette_score(X, y, **kwds)

    # Supervised metrics
    elif _metric == 'adjusted_rand_score':
        return metrics.adjusted_rand_score(true_labels, y)
    elif _metric == 'fowlkes_mallows_score':
        return metrics.fowlkes_mallows_score(true_labels, y)

    else:
        raise ValueError('Unimplemented metric')


def sliced_wasserstein_distance(barcode1, barcode2, M, ord=1):
    """
    Approximate Sliced Wasserstein distance between two barcodes
    :param barcode1:
    :param barcode2:
    :param M: the approximation factor, bigger M means more accurate result
    :param ord: p-Wassertein distance to use
    :return:
    """
    diag = np.array([np.sqrt(2), np.sqrt(2)])
    b1 = list(barcode1)
    b2 = list(barcode2)
    for bar in barcode1:
        b2.append(orth_proj(bar, diag))
    for bar in barcode2:
        b1.append(orth_proj(bar, diag))
    b1 = np.array(b1, copy=False)
    b2 = np.array(b2, copy=False)
    s = np.pi / M
    theta = -np.pi / 2
    sw = 0
    for i in range(M):
        dir = np.array([np.cos(theta), np.sin(theta)])
        v1 = np.sort(np.dot(b1, dir))
        v2 = np.sort(np.dot(b2, dir))
        sw += s * norm(v1 - v2, ord)
        theta += s
    return sw / np.pi


def plot_clustering(points, labels, title=None, similarity=None, show=True, **kwargs):
    X = PCA(2).fit_transform(points) if points.shape[1] > 2 else points

    if similarity is None:
        fig, ax = plt.subplots(1, 1, **kwargs)
    else:
        fig, [ax, ax2] = plt.subplots(1, 2, **kwargs)
        sorted_labels = np.argsort(labels)
        similarity = [[similarity[i, j] for j in sorted_labels] for i in sorted_labels]
        ax2.pcolormesh(similarity, cmap=plt.cm.jet)
    fig.suptitle(title)
    ax.scatter(X[:, 0], X[:, 1], c=labels)
    ax.set_xlim(min(X[:, 0]), max(X[:, 0]))
    ax.set_ylim(min(X[:, 1]), max(X[:, 1]))
    for c in set(labels):
        for i, p in enumerate([x for label, x in zip(labels, X) if label == c]):
            ax.text(p[0], p[1], str(i), color="red", alpha=1, fontsize=12)
    if show:
        fig.show()
    return fig


def barcode_to_diagram(barcode):
    """
    Transform a list of tuple into a dionysus diagram
    :param barcode:
    :return:
    """
    l = []
    for bar in barcode:
        if len(bar) < 2:
            l.append((bar[0], np.inf))
        elif int(bar[0] * 1000) != int(bar[1] * 1000):
            l.append(bar)
    return di.Diagram(l)


def get_ph_neuron(neuron, neurite_type='all', feature='radial_distances', **kwargs):
    """
    Compute the persistence diagram of a neuron using tmd
    :param neuron:
    :param neurite_type:
    :param feature:
    :param kwargs:
    :return:
    """
    ph_all = []

    if neurite_type == 'all':
        neurite_list = ['neurites']
    else:
        neurite_list = [neurite_type]

    for t in neurite_list:
        for tr in getattr(neuron, t):
            ph_all = ph_all + get_degree_0_persistence(tr, feature=feature, **kwargs)

    return ph_all


def _compress(union_find, u):
    while not union_find[u] == union_find[union_find[u]]:
        union_find[u] = union_find[union_find[u]]
    return union_find[u]


def get_degree_0_persistence(tree, feature='radial_distances', **kwargs):
    """
    Compute the degree 0 persistence of a tree according to the given feature
    Runs in O(log(m)*m) where m is the number of sections in that tree.
    :param tree:
    :param feature:
    :param kwargs:
    :return:
    """
    from queue import PriorityQueue

    ph = []

    f = getattr(tree, 'get_point_' + feature)(**kwargs)
    filtration = PriorityQueue()

    # sort simplices in lexico-graphical order
    # Given, u and v such that f(u) < f(v),
    # one has {u} < {v} < {u, v}
    parents = tree.p

    filtration.put((f[0], 1, 0))
    for i in range(1, len(parents)):
        filtration.put((f[i], 1, i))
        p = parents[i]
        filtration.put((max(f[i], f[p]), 2, {i, p}))

    union_find = {}

    # when \sigma = {u}, add u to the union-find structure V
    # when \sigma = {u, v}, assumes wlog, that f[u] < f[v]
    #   merge e_v in e_u, add (f[e_v], max(f[u], f[v]) to the barcode
    while not filtration.empty():
        val, d, sigma = filtration.get()
        if d == 1:
            if sigma not in union_find:
                union_find[sigma] = sigma
        else:
            u, v = sigma
            e_u, e_v = _compress(union_find, u), _compress(union_find, v)
            if e_u != e_v:
                if f[e_u] > f[e_v]:
                    e_u, e_v = e_v, e_u
                union_find[e_v] = e_u
                if val != f[e_v]:
                    ph.append((f[e_v], val))

    # extend the barcode with bars (f[e_u], max(ph)) for every unmerged component
    M = max(ph, key=lambda bar: bar[1])[1]
    ph.extend([(f[e_u], M) for e_u in set(map(lambda u: _compress(union_find, u), union_find.keys()))])

    return ph


def discretize(barcodes, M):
    x_1, x_M = np.inf, -np.inf
    y_1, y_M = np.inf, -np.inf

    for barcode in barcodes:
        for b in barcode:
            if b[0] > x_M:
                x_M = b[0]
            if b[0] < x_1:
                x_1 = b[0]
            if b[1] > y_M:
                y_M = b[1]
            if b[1] < y_1:
                y_1 = b[1]

    discretized_barcodes = []

    for barcode in barcodes:
        v = np.zeros(M)
        for a, b in barcode:
            l1 = int(np.ceil((a - x_1) / (x_M - x_1) * M))
            l2 = int(np.floor((b - y_1) / (y_M - y_1) * M))

            for l in range(l1, l2):
                v[l] += 1
        discretized_barcodes.append(v)

    return np.array(discretized_barcodes, copy=False), (x_1, x_M), (y_1, y_M)


class BarcodePCA():
    def __init__(self, M, n_components=None):
        self.M = M
        self.n_components = n_components
        self.weights = None
        self._explained_variance_ratio = None
        self.ksi = 0

    @staticmethod
    def _weight(x, y):
        return (y - x) / (1 + (y - x) ** 3)

    def fit(self, X, y=None):
        v, (x_1, x_M), (y_1, y_M) = discretize(X, self.M)
        v -= np.mean(v, axis=0)

        D = np.zeros((len(v), len(v)))

        self.weights = np.array(map(BarcodePCA._weight, zip(np.linspace(x_1, x_M, self.M)),
                                    np.linspace(y_1, y_M, self.M)), copy=False)

        for i, v_i in enumerate(v):
            for j, v_j in enumerate(v[:i + 1]):
                D[i, j] = np.sum(v_i * v_j * self.weights)

        D += D.T

        vals, w = eigh(D)
        vals, w = vals[::-1], w[::-1]

        ksi = np.zeros((len(w), len(v)))
        for j, w_j in enumerate(w):
            ksi[j] += np.sum([a_i * v_i for a_i, v_i in zip(w_j, v)], axis=1)

        self.ksi = ksi

        n_components = len(w) if self.n_components is None else self.n_components

        self._explained_variance_ratio = sum(vals[:n_components]) / sum(vals)

        return self

    def fit_transform(self, X, y=None):
        v, (x_1, x_M), (y_1, y_M) = discretize(X, self.M)
        v -= np.mean(v, axis=0)

        D = np.zeros((len(v), len(v)))

        self.weights = np.array(map(BarcodePCA._weight, zip(np.linspace(x_1, x_M, self.M)),
                                    np.linspace(y_1, y_M, self.M)), copy=False)

        for i, v_i in enumerate(v):
            for j, v_j in enumerate(v[:i + 1]):
                D[i, j] = np.sum(v_i * v_j * self.weights)

        D += D.T

        vals, w = eigh(D)
        vals, w = vals[::-1], w[::-1]

        ksi = np.zeros((len(w), len(v)))
        for j, w_j in enumerate(w):
            ksi[j] += np.sum([a_i * v_i for a_i, v_i in zip(w_j, v)], axis=1)

        self.ksi = ksi

        n_components = len(w) if self.n_components is None else self.n_components

        self._explained_variance_ratio = sum(vals[:n_components]) / sum(vals)

        s = np.zeros((len(v), n_components))
        for i in range(len(v)):
            for j, w_j in enumerate(w[:n_components]):
                s[i, j] += np.sum(np.sum(w_j * D[:, i]))
                s[i, j] /= np.sqrt(np.matmul(np.matmul(w_j.T, D), w_j))

        return s

    def tranform(self, X):
        v, (x_1, x_M), (y_1, y_M) = discretize(X, self.M)
        v -= np.mean(v, axis=0)

        n_components = len(self.ksi) if self.n_components is None else self.n_components

        s = np.zeros((len(v), n_components))
        for i, v_i in enumerate(v):
            for j, ksi_j in enumerate(self.ksi[:n_components]):
                s[i, j] = np.sum(v_i * ksi_j * self.weights)

        return s


def pca_barcode(barcodes, M, n_components=None):
    v, (x_1, x_M), (y_1, y_M) = discretize(barcodes, M)
    v -= np.mean(v, axis=0)

    D = np.zeros((len(v), len(v)))

    weights = np.array(map(lambda x, y: 1 / (1 + (y - x) ** 2),
                           zip(np.linspace(x_1, x_M, M)), np.linspace(y_1, y_M, M)), copy=False)

    for i, v_i in enumerate(v):
        for j, v_j in enumerate(v[:i + 1]):
            D[i, j] = np.sum(v_i * v_j * weights)

    D += D.T

    vals, w = eigh(D)
    vals, w = vals[::-1], w[::-1]

    ksi = np.zeros((len(w), len(v)))
    for j, w_j in enumerate(w):
        ksi[j] += np.sum([a_i * v_i for a_i, v_i in zip(w_j, v)], axis=1)

    if n_components is None:
        n_components = len(w)

    explained_variance_ratio = sum(vals[:n_components]) / sum(vals)

    s = np.zeros((len(v), n_components))
    for i in range(len(v)):
        for j, w_j in enumerate(w[:n_components]):
            s[i, j] += np.sum(np.sum(w_j * D[:, i]))
            s[i, j] /= np.sqrt(np.matmul(np.matmul(w_j.T, D), w_j))

    return s
