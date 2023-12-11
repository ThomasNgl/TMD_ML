def extract_results(csvfile, dialect="excel-tab"):
    import csv

    with open(csvfile, 'r') as f:
        reader = csv.reader(f, dialect=dialect)
        reader.next()
        return list(reader)


def plot_accuracies(csvfile_or_results, out=None, show=True, dialect="excel-tab", n_clusters=None, filter_fn=None,
                    **kwargs):
    def _name_to_ix(name):
        if "bottleneck" in name:
            return "yellow"
        if "wasserstein" in name:
            return "red"
        if "sw" in name:
            return "orange"
        if "landscape" in name:
            return "green"
        if "gaussian" in name:
            return "purple"
        if "signature" in name:
            return "pink"
        return "blue"

    import matplotlib.pyplot as plt

    if isinstance(csvfile_or_results, str):
        results = extract_results(csvfile_or_results, dialect)
    else:
        results = csvfile_or_results

    if filter is not None:
        results = filter(filter_fn, results)

    y_pos = range(len(results))
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111, )
    ax.barh(y_pos, map(lambda r: float(r[1]), results),
            xerr=map(lambda r: float(r[2]), results),
            color=map(lambda r: _name_to_ix(r[0]), results))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(map(lambda r: r[0], results))
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, 1)

    if n_clusters is not None:
        ax.plot([1. / n_clusters, 1. / n_clusters], [ymin, ymax], 'r--')
    if out is not None:
        fig.savefig(out)

    if show:
        fig.show()

    return fig


def plot_sphere(csvfile_or_results, out=None, show=True, dialect="excel-tab", n_clusters=None, filter_fn=None,
                **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from mpl_toolkits.mplot3d import Axes3D

    if isinstance(csvfile_or_results, str):
        results = extract_results(csvfile_or_results, dialect)
    else:
        results = csvfile_or_results

    if filter is not None:
        results = filter(filter_fn, results)

    X = []
    accuracies = []

    for result in results:
        if "projection-" in result[0]:
            X.append([float(x) for x in eval(result[0][result[0].find("projection-") + len("projection-"):])])
            accuracies.append(float(result[1]))

    X = np.array(X)
    accuracies = np.array(accuracies)

    delau = ConvexHull(X)
    colors = np.mean(accuracies[delau.simplices], axis=1)

    fig = plt.figure(**kwargs)
    ax = fig.gca(projection="3d")

    collec = ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=delau.simplices,
                             cmap=plt.cm.jet, antialiased=False)
    collec.set_array(colors)
    collec.set_clim(1. / n_clusters if n_clusters is not None else 0, 1)

    fig.colorbar(collec)

    if out is not None:
        fig.savefig(out)
    if show:
        fig.show()

    return fig


def plot_accuracy_vs_projnum(dataset, csvfile_or_results, dist, n_clusters=None, out=None, show=True, dialect="excel-tab",
                             **kwargs):
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    import diagram as dg

    h5file = "dgms.h5"

    if isinstance(csvfile_or_results, str):
        results = np.array(extract_results(csvfile_or_results, dialect=dialect), copy=False)
    else:
        results = np.array(csvfile_or_results, copy=False)

    results = results[np.where(map(lambda r: dist + "-projection" in r[0], results))]

    n_runs = 10
    random_state = 42
    features = [f[len(dist + "-"):] for f in results[:, 0]]

    cv = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)

    with h5py.File(h5file) as f:
        n = len(f[dataset][features[0]]['dists'][dist].value)
        kernels = np.zeros((len(features), n, n))
        labels_true = []
        for label in f[dataset][features[0]].keys():
            if label == 'dists':
                continue
            labels_true.extend([label] * len(f[dataset][features[0]][label]))

        best_proj = results[np.argmax(results[:, 1])]
        best_proj_accuracy = float(best_proj[1])
        sigma = np.sqrt(np.median(f[dataset][features[0]]['dists'][dist].value))

        for i, result, feature in zip(range(len(features)), results, features):
            key = '/'.join([dataset, feature, 'dists', dist])
            distances = f[key].value
            kernels[i] = dg.gaussian_kernel(distances, sigma)

    K = np.unique(np.logspace(0, np.log10(len(kernels)), 100, endpoint=False, dtype=int))
    means = []
    vs_means = []
    err = []
    for k in K:
        score = []
        vs_score = []
        std = []
        for i in range(n_runs):
            ixes = np.random.choice(len(kernels), k, replace=False)
            vs_score.append(100 * np.mean([float(s) for s in results[ixes, 1]]))
            kernel_sum = np.sum(kernels[ixes], axis=0)
            svc = SVC(kernel="precomputed")
            clf = GridSearchCV(svc, {"C": [0.01, 0.1, 1, 10, 100], 'class_weight': ['balanced', None]}, cv=cv)
            clf.fit(kernel_sum, labels_true)
            score.append(100 * clf.cv_results_['mean_test_score'][clf.best_index_])
            std.append(100 * clf.cv_results_['std_test_score'][clf.best_index_])

        means.append(np.mean(score))
        vs_means.append(np.mean(vs_score))
        err.append(np.sqrt(np.dot(std, std)) / n_runs)

    fig, ax = plt.subplots(**kwargs)
    fig.suptitle("Accuracy of the classification with {} distance of {} neurons vs number of projections used"
                 .format(dist, dataset))
    errorbar = ax.errorbar(K, means, yerr=err, fmt='--', label="Kernel sum")
    ax.plot(K, vs_means, "r-", K, len(K) * [100 * best_proj_accuracy], "k-")
    ax.set_xlabel("Number of projections (logscale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale('log')

    redpatch = mpatches.Patch(color='red', label='Average', linewidth=1)
    blackpatch = mpatches.Patch(color='black', label='Best projection', linewidth=1)
    plt.legend(handles=[errorbar, redpatch, blackpatch])

    if n_clusters is not None:
        ax.set_ylim((100. / n_clusters, 100))

    if out is not None:
        fig.savefig(out)
    if show:
        fig.show()

    return fig


if __name__ == '__main__':
    import sys
    from getopt import getopt

    opts, args = getopt(sys.argv[1:], "n:d:s", ["n_clusters=", "dialect=", "no_show"])

    options = {}
    for opt, arg in opts:
        if opt in ("--n_clusters", "-n"):
            options['n_clusters'] = int(arg)
        if opt in ("--dialect", "-d"):
            options['dialect'] = arg
        if opt in ("--no_show", "-s"):
            options['show'] = False

    csv = args[0]
    out = args[1] if len(args) > 1 else None
    plot_accuracies(csv, out, **options)
