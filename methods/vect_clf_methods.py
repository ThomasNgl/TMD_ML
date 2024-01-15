import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from methods.utils import compute_phs_from_pop_list, sort_bars

#Experiment with a vectorization and a clf from sklearn
def clf_experiment(X_vect, y,
                   clf = RandomForestClassifier(),
                   embedder = None,
                    cv = ShuffleSplit(n_splits=10, test_size=.20, random_state=0),
                    normalize_features = False):

    #preprocessing
    if normalize_features:
        feature_scaler = StandardScaler()
        X_vect = feature_scaler.fit_transform(X_vect)

    #dim reduction
    if embedder:
        X_vect = embedder.fit_transform(X_vect)

    #classification
    score = cross_validate(clf, X_vect, y, cv=cv, return_train_score =True)
    test_score = score['test_score']
    train_score = score['train_score']
    fit_time = np.mean(score['fit_time'])
    return train_score, test_score, fit_time

def vect_clf_experiment(pwd_list, vect_method, vect_params = None,
                        clf = RandomForestClassifier(),
                        embedder = None,
                        cv = ShuffleSplit(n_splits=10, test_size=.20, random_state=0),
                        sorted = True,
                        normalize_features = False, neurite_type='apical_dendrite', feature = "projection"):
    #tabularization
    X, y = compute_phs_from_pop_list(pwd_list=pwd_list, neurite_type=neurite_type, feature=feature)
    #preprocessing
    if sorted:
        X = sort_bars(X)
    
    #vectorization
    if vect_params:
        X_vect = vect_method(X, *vect_params)
    else:
        X_vect = vect_method(list_phs = X)
    
    train_score, test_score, fit_time = clf_experiment(X_vect, y,
                                                        clf = clf,
                                                        embedder = embedder,
                                                        cv = cv,
                                                        normalize_features = normalize_features)
    return train_score, test_score, fit_time

#distance matrix
def dist_clf_experiment(pwd_list, distance_method, clf, embedder = None, nb_splits = 50, sorted = True, neurite_type='apical_dendrite', feature = 'projection'):
    #tabularization
    X, y = compute_phs_from_pop_list(pwd_list=pwd_list, neurite_type=neurite_type, feature=feature)
    
    #preprocessing
    if sorted:
        X = sort_bars(X)

    train_score_list =[]
    test_score_list = []

    nb_neurons = len(y)
    len_train = int(nb_neurons*0.8)
    #distance
    X_dist = np.array([[distance_method(X[i], X[j]) for i in range(nb_neurons)] for j in range(nb_neurons)])
    
    #cv
    for _ in range(nb_splits):
        p = np.random.permutation(nb_neurons)
        X_processed = X_dist.copy()[p]
        y_processed = np.array(y).copy()[p]

        X_processed = X_processed[:, p[:len_train]]
        y_train = y_processed[:len_train]
        y_test = y_processed[len_train:]
        X_dist_train = X_processed[:len_train]
        X_dist_test = X_processed[len_train:]

        #Dim red
        if embedder:
            scaler = StandardScaler()
            X_dist_scaled_train = scaler.fit_transform(X_dist_train)
            X_dist_scaled_test = scaler.transform(X_dist_test)
            
            X_dist_train = embedder.fit_transform(X_dist_scaled_train)
            X_dist_test = embedder.transform(X_dist_scaled_test)
        #classification
        clf.fit(X_dist_train, y_train)
        train_score_list.append(clf.score(X_dist_train, y_train))
        test_score_list.append(clf.score(X_dist_test, y_test))

    return train_score_list, test_score_list