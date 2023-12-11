

def combine_vect_clf_experiment(pwd_list, vect_method_list, vect_params_list = None,
                        clf = RandomForestClassifier(),
                        embedder_list = None,
                        cv = ShuffleSplit(n_splits=10, test_size=.20, random_state=0),
                        sorted = True,
                        normalize_features_list = False, normalize_features_combined= False, embedder_combined= None):
    #tabularization
    X, y = compute_phs_from_pop_list(pwd_list=pwd_list)

    #preprocessing
    if sorted:
        X = sort_bars(X)

    nb_methods = len(vect_method_list)
    if not vect_params_list:
        vect_params_list = [False]*nb_methods
    
    if not normalize_features_list:
        normalize_features_list = [None]*nb_methods
    
    if not embedder_list:
        embedder_list = [None]*nb_methods

    #vectorizations
    X_vect_combined = []
    for i in range(nb_methods):
        if vect_params_list[i]:
            X_vect = vect_method_list[i](X.copy(), *vect_params_list[i])
        else:
            X_vect = vect_method_list[i](list_phs = X.copy())
    
        #preprocessing
        if normalize_features_list[i]:
            feature_scaler = StandardScaler()
            X_vect = feature_scaler.fit_transform(X_vect)

        #dim reduction
        if embedder_list[i]:
            X_vect = embedder_list[i].fit_transform(X_vect)

        X_vect_combined.append(X_vect)
    X_vect_combined = np.concatenate(X_vect_combined, axis = 1)
    
    #preprocessing
    if normalize_features_combined:
        feature_scaler = StandardScaler()
        X_vect_combined = feature_scaler.fit_transform(X_vect_combined)
    #dim reduction
    if embedder_combined:
        X_vect_combined = embedder_combined.fit_transform(X_vect_combined)
    #classification
    score = cross_validate(clf, X_vect_combined, y, cv=cv, return_train_score =True)
    test_score = score['test_score']
    train_score = score['train_score']
    fit_time = np.mean(score['fit_time'])
    
    return train_score, test_score, fit_time

def combine_vect_dist_clf_experiment(pwd_list, distance_method, clf, embedder = None, nb_splits = 50, sorted = True):
    #tabularization
    X, y = compute_phs_from_pop_list(pwd_list=pwd_list)
    
    #preprocessing
    if sorted:
        X = sort_bars(X)

    train_score_list =[]
    test_score_list = []

    nb_neurons = len(y)
    len_train = int(nb_neurons*0.8)
    #distance
    X_dist = np.array([[distance_method(X[i], X[j]) for i in range(nb_neurons)] for j in range(nb_neurons)])
    X_pima = PI(X, *[10, True, True, None, True])
    X_lifc = lifespan_curve(X, *[False, 20, False])
    X_vect_combined = np.concatenate([X_pima, X_lifc], axis = 1)

    #cv
    for _ in range(nb_splits):
        p = np.random.permutation(nb_neurons)
        X_processed = X_dist.copy()[p]        
        X_processed = X_processed[:, p[:len_train]]
        X_d_train, X_d_test = X_processed[:len_train], X_processed[len_train:]
        pca = PCA(n_components = 10)
        X_d_train = pca.fit_transform(X_d_train)
        X_d_test = pca.transform(X_d_test)
        
        X_vect = X_vect_combined.copy()[p]
        X_v_train, X_v_test = X_vect[:len_train], X_vect[len_train:]
        
        y_processed = np.array(y).copy()[p]        
        y_train = y_processed[:len_train]
        y_test = y_processed[len_train:]

        X_train = np.concatenate([X_v_train, X_d_train], axis = 1)
        X_test = np.concatenate([X_v_test, X_d_test], axis = 1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #Dim red

        if embedder:
            
            X_train = embedder.fit_transform(X_train)
            X_test = embedder.transform(X_test)
        
        #classification
        clf.fit(X_train, y_train)
        train_score_list.append(clf.score(X_train, y_train))
        test_score_list.append(clf.score(X_test, y_test))

    return train_score_list, test_score_list