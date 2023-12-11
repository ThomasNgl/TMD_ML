import numpy as np
import torch
from torch.utils.data import TensorDataset

# From the repo
from methods.utils import compute_phs_from_pop_list, sort_bars
from methods.vect_methods import PI
from others.CNN import CNNet 
from others.cocob import COCOBBackprop
from others.Trainer import Trainer
#CNN

def cnn_experiment(pwd_list, neurite_type = "apical_dendrite", feature = "projection"):
    final_results = {}
    train_acc_list_layer = []
    val_acc_list_layer = []
    for layer in pwd_list.keys():
        print(pwd_list[layer])
        X, y = compute_phs_from_pop_list(pwd_list=pwd_list[layer], neurite_type = neurite_type, feature = feature)
        X_sorted = sort_bars(X)
        resolution = 50
        X_vect = PI(X_sorted, resolution = resolution, flatten = False)
        X = torch.tensor(X_vect, dtype = torch.float32)
        X = torch.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
        y = torch.tensor(y, dtype= torch.long)

        folds = 50
        test_fraction = 0.2
        dataset = TensorDataset(X, y)
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        for _ in range(folds):
            dataset_train, dataset_test = torch.utils.data.random_split(dataset, [1-test_fraction, test_fraction])
            X_train_list.append(dataset_train[:][0])
            X_test_list.append(dataset_test[:][0])
            y_train_list.append(dataset_train[:][1])
            y_test_list.append(dataset_test[:][1])
            mx_epochs = 75

        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []
        for i in range(folds):    
            model = CNNet(n_classes=len(pwd_list[layer]), image_size=resolution, bn =False)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = COCOBBackprop(model.parameters())
            trainer = Trainer(model, criterion, optimizer)
            result = trainer.fit_val(X_train_list[i], X_test_list[i],
                                        y_train_list[i], y_test_list[i], 
                                        max_epochs=mx_epochs, batch_size=8)
            train_acc_list.append(result[0])
            train_loss_list.append(result[1])
            val_acc_list.append(result[2])
            val_loss_list.append(result[3])

        val_score_mean = np.mean(val_acc_list, axis=0)
        val_score = val_score_mean[-1]
        final_results['pi_cnn_'+layer] = val_score
        train_acc_list_layer.append(train_acc_list)
        val_acc_list_layer.append(val_acc_list)
    return final_results, train_acc_list_layer, val_acc_list_layer