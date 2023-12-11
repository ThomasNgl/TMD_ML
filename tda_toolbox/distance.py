import os.path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import sklearn.svm as svm
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import tmd
import data

from diagram import sliced_wasserstein_distance as sw
from plot_svm import plot_svm


def compute_barcodes(dataset, output=None, forced=False, feature='radial_distances', neurite_list=['apical'], **kwds):
    """
    Compute the barcodes of the given neurons and save it to output
    :param dataset:
    :param output:
    :param forced:
    :return:
    """
    neurons = []
    barcodes = []

    if forced or not output or not os.path.isfile(output):
        print "Loading neurons"

        for filename in dataset:
            neurons.append(tmd.io.load_neuron(filename))

        print "Computing barcodes"

        for neuron in neurons:
            barcode = []
            for neurite_type in neurite_list:
                barcode += tmd.methods.get_ph_neuron(neuron, neurite_type=neurite_type, feature=feature, **kwds)
            barcodes.append(barcode)
        if output:
            np.save(output, barcodes)
    else:
        barcodes = np.load(output)

    return barcodes


def compute_distances(dataset, output=None, output_barcode=None, feature='radial_distances',
                      neurite_list=['apical'],
                      d=lambda bar1, bar2: sw(bar1, bar2, 20), forced=False, **kwds):
    """
    Compute the distance matrix corresponding to the given dataset
    :param dataset: an array of neuron files
    :param output: a filename to save the distance matrix
    :param d: the distance function to use
    :param forced: force the recomputation of the distance matrix
    :return: the distance matrix
    """

    D = np.array([])

    if forced or not output or not os.path.isfile(output):
        barcodes = compute_barcodes(dataset, output=output_barcode, forced=forced,
                                    neurite_list=neurite_list, feature=feature, **kwds)

        print "Computing distances"

        D = np.zeros((len(barcodes), len(barcodes)))

        for i, bar1 in enumerate(barcodes):
            for j, bar2 in enumerate(barcodes[i + 1:]):
                D[i, i + 1 + j] = d(bar1, bar2)

        D += np.transpose(D)

        if output:
            np.save(output, D)
    else:
        D = np.load(output)

    return D


class Distance:
    def __init__(self, dataset, output=None, output_barcode=None,
                 feature='radial_distances', neurite_list=['apical'],
                 transform=lambda barcodes: barcodes,
                 d=lambda bar1, bar2: sw(bar1, bar2, 20), forced=False):
        self.dataset = dataset
        self.output = output
        self.output_barcode = output_barcode
        self.feature = feature
        self.neurite_list = neurite_list
        self.transform = transform
        self.d = d
        self.forced = forced
        self.neurons = []
        self.barcodes = None
        self.barcodes_t = None
        self.similarity = None
        self.distances = None

        self.labels = []
        for filename in data.datasets[self.dataset]:
            label = filename.split('\\')[1]
            self.labels.append(label)

    def compute_barcodes(self):
        if self.forced or not self.output_barcode or not os.path.isfile(self.output_barcode):
            print "Loading neurons"

            filenames = [os.path.join(data.neurons_location, f.replace('\\', os.path.sep)) for f in data.datasets[self.dataset]]

            for filename in filenames:
                self.neurons.append(tmd.io.load_neuron(filename))

            print "Computing barcodes"

            self.barcodes = []

            for neuron in self.neurons:
                barcode = []
                for neurite_type in self.neurite_list:
                    barcode += tmd.methods.get_ph_neuron(neuron, neurite_type=neurite_type, feature=self.feature)
                    self.barcodes.append(barcode)
            if self.output_barcode:
                np.save(self.output_barcode, self.barcodes)
        else:
            self.barcodes = np.load(self.output_barcode)

        return self.barcodes

    def transform_barcodes(self):
        self.barcodes_t = self.transform(self.compute_barcodes()) if self.barcodes is None \
            else self.transform(self.barcodes)
        return self.barcodes_t

    def compute_distances(self):
        D = np.array([])

        if self.forced or not self.output or not os.path.isfile(self.output):
            barcodes = self.transform_barcodes() if self.barcodes_t is None else self.barcodes_t

            print "Computing distances"

            D = np.zeros((len(barcodes), len(barcodes)))

            for i, bar1 in enumerate(barcodes):
                for j, bar2 in enumerate(barcodes[i + 1:]):
                    D[i, i + 1 + j] = self.d(bar1, bar2)

            D += np.transpose(D)

            if self.output:
                np.save(self.output, D)
        else:
            D = np.load(self.output)

        self.distances = D
        return D

    def compute_similarity(self, gamma):
        distances = self.compute_distances() if self.distances is None else self.distances
        self.similarity = np.exp(-gamma * distances)
        return self.similarity

    @staticmethod
    def plot_similarity(ax, similarity, labels=None):
        if not labels is None:
            sorted_labels = np.argsort(labels)
            similarity = [[similarity[i, j] for j in sorted_labels] for i in sorted_labels]
        return ax.pcolormesh(similarity, cmap=plt.cm.jet)

    @staticmethod
    def fit(points, labels, param_grid=None, random_state=42, shuffle=True):

        if param_grid is None:
            param_grid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'],
                 'class_weight': ['balanced', None]},
            ]

        cv = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=shuffle)

        svc = svm.SVC()
        clf = GridSearchCV(svc, param_grid, cv=cv)
        clf.fit(points, labels)
        return [clf.cv_results_[p][clf.best_index_] for p in ['mean_test_score', 'std_test_score', 'params']]

    @staticmethod
    def unsupervised_fit(points, labels, n_clusters):
        colors = {l: i for i, l in enumerate(set(labels))}
        labels_int = [colors[l] for l in labels]

        clf = KMeans(n_clusters=n_clusters)
        k_labels = clf.fit_predict(points)

        return metrics.fowlkes_mallows_score(labels_int, k_labels)

    @staticmethod
    def plot_svm(ax, points, labels, params={}):
        pca = PCA(2)
        # kpca = KernelPCA(2, kernel='rbf', gamma=0.001)
        label_to_int = {l: i for i, l in enumerate(set(labels))}
        y = [label_to_int[label] for label in labels]
        X = pca.fit_transform(points, y)

        clf = svm.SVC()
        clf.set_params(**params)
        clf.fit(X, y)

        plot_svm(ax, clf, X, y)
