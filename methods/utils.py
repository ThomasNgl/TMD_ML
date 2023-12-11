import numpy as np
import torch
import tmd
from scipy import stats
import random

#Preprocess 
#####
def shuffle(X, y):
        permutations = np.random.permutation(len(y))
        shuffle_X = []
        shuffle_y = []
        for p in permutations:
            shuffle_X.append(np.array(X[p]))
            shuffle_y.append(y[p])
        return shuffle_X, shuffle_y

def compute_phs_from_pop_list(pwd_list, shuffle_raws = True, use_morphio = True, neurite_type = "apical_dendrite", feature = "projection"):
    '''
    Params:
    pwd_list: a list of string.
            The list of pwds of the folders containing reconstructed neuron files.
    Return:
    X: list of np.array of shape (nb_bars, 2).
            The list of the phs/pds/barcode computed from the list of reconstructed neurons.
            nb_bars is the number of bars/points in a pd/barcode and 2 because those elements are represented by 2 real values.
    y: list of int.
            The list of the labels. 
            Labels are number not string. They correspond to their index in the pwd_list.
            '''
    population_list = []
    #ph_list
    X = []
    #population_names = []
    #list of indices, tabularized dataset
    y = []
    
    for pwd in pwd_list:
        
        population = tmd.io.load_population(pwd, use_morphio = use_morphio)
        population_list.append(population)
        #population_names.append(population.name)

        for neuron in population.neurons:
            ph = tmd.methods.get_ph_neuron(neuron, neurite_type = neurite_type, feature = feature)
            X.append(np.array(ph))
            y.append(len(population_list)-1)

    if shuffle_raws:
        X, y = shuffle(X, y) 
    
    return X, y

def zero_padding(X):
    #find the maxi nb of features
    max_len = 0
    for sample in X:
        if max_len < len(sample):
            max_len = len(sample)
    #fill w zeros
    nul_bar = np.array([[0.,0.]])
    for i in range(len(X)):
        dif = max_len - len(X[i])
        if dif != 0:
            list_nul_bar = np.repeat(nul_bar, dif, 0)
            X[i] = torch.tensor(np.concatenate((X[i], list_nul_bar), axis = 0), dtype = torch.float32)
        else:
            X[i] = torch.tensor(X[i], dtype = torch.float32)
    X = torch.stack((X), axis = 0)
    return X
#####

# For vect methods
#####
def sort_bars(list_phs):
    #Sort bars to respect birth < death
    list_phs_sorted = []
    for ph in list_phs:
        list_phs_sorted.append(np.sort(ph, axis = 1))
    list_phs = list_phs_sorted
    return list_phs

def define_limits(list_phs):
    #tmd.analysis.get_limits
    """Return the x-y coordinates limits (min, max) for a list of persistence diagrams."""
    ph = tmd.analysis.collapse(list_phs)
    xlim = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylim = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]

    return xlim, ylim
#####

# PI 
#####
def pi_function(ph, xlim, ylim, std = None, resolution =100):
    #def get_persistence_image_data(ph, norm_factor=None, xlim=None, ylim=None, bw_method=None):
    res = complex(0, resolution)
    X, Y = np.mgrid[xlim[0] : xlim[1] : res, ylim[0] : ylim[1] : res]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values, bw_method=std)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    return Z
#####

# template_function
#####
def tent_functions(x, center, delta=1):
	lifetime = x[:,1] - x[:,0]
	return np.sum(np.maximum(0, 1 - (1/delta)*np.maximum(np.abs(x[:,0] - center[0]), np.abs(lifetime - center[1]))))    
#####

# Perslay
#####
def shuffle_split_perslay(X, y, folds = 40, test_fraction = 0.2):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    length = len(y)
    max_index = int((1-test_fraction)*length)
    for i in range(folds):
        dataset = list(zip(X, y))
        random.shuffle(dataset)
        new_X, new_y = zip(*dataset)
        X_train = new_X[:max_index]
        X_test = new_X[max_index:]

        y_train = new_y[:max_index]
        y_test = new_y[max_index:]

        X_train_list.append(list(X_train))
        X_test_list.append(list(X_test))
        y_train_list.append(torch.tensor(y_train, dtype= torch.long))
        y_test_list.append(torch.tensor(y_test, dtype= torch.long))
    
    y_train_list = torch.stack((y_train_list), axis=0)
    y_test_list = torch.stack((y_test_list), axis=0)
    return X_train_list, X_test_list, y_train_list, y_test_list

def select_best_perslay_experiments(dict_results, min_acc):
    best_experiments = {}
    for key in dict_results:
        val_acc_list = dict_results[key][2]
        val_score_mean = np.mean(val_acc_list, axis=0)
        max_val_acc = np.max(val_score_mean)
        if max_val_acc>min_acc:
            best_experiments[key] = dict_results[key]
    return best_experiments
#####