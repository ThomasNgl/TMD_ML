import numpy as np
import torch

from methods.utils import compute_phs_from_pop_list, sort_bars, zero_padding, shuffle_split_perslay, define_limits
from torch.utils.data import TensorDataset
import copy
from perslay_model.Perslay import PerslayModel
from others.cocob import COCOBBackprop
from others.Trainer import Trainer

#Perslay
def perslay_experiment(pwd_list, padding0, test_fraction, folds, 
                       point_transform, weight_function, operator,
                       hiddens,
                       max_epochs = 75, batch_size = 8,
                       nb_classes = 4,
                       copie = True, neurite_type = "apical_dendrite", feature = "projection"):
    
    X, y = compute_phs_from_pop_list(pwd_list=pwd_list, neurite_type = neurite_type, feature = feature)
    X = sort_bars(X)

    if padding0:
        X = zero_padding(X)
        y = torch.tensor(y, dtype= torch.long)
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

    else: 
        X_train_list, X_test_list, y_train_list, y_test_list = shuffle_split_perslay(X, y, test_fraction=test_fraction, folds=folds)
    
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    parameters_list = []
    for i in range(len(y_train_list)):
        if copie:
            weight_function_copy = copy.deepcopy(weight_function)
            weight_function_copy.__init__(*weight_function.attributes)

            point_transform_copy = copy.deepcopy(point_transform)
            point_transform_copy.__init__(*point_transform.attributes)
        else:
            weight_function_copy = weight_function
            point_transform_copy = point_transform
            
        xlim, ylim = define_limits(X_train_list[i])
        min_bound, max_bound = np.min((xlim, ylim)), np.max((xlim, ylim))
        weight_function_copy.min_bound = torch.tensor(min_bound)
        weight_function_copy.max_bound = torch.tensor(max_bound)

        model = PerslayModel(weight_function=weight_function_copy, 
                            point_transform=point_transform_copy,
                            operator=operator, 
                            output_dim=nb_classes, hiddens_dim=hiddens,
                            is_tensor = padding0)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = COCOBBackprop(model.parameters())
        trainer = Trainer(model, criterion, optimizer)
        result = trainer.fit_val(X_train_list[i], X_test_list[i],
                                    y_train_list[i], y_test_list[i], 
                                    max_epochs, batch_size)
        train_acc_list.append(result[0])
        train_loss_list.append(result[1])
        val_acc_list.append(result[2])
        val_loss_list.append(result[3])
        
        parameters = trainer.model.parameters()
        parameters = [p for p in parameters]
        parameters_list.append(parameters)

    return train_acc_list, train_loss_list, val_acc_list, val_loss_list, parameters_list  

# From the repo
from perslay_model.WeightFunction import WeightFunction
from perslay_model.PointTransformations import GaussPT
from perslay_model.Operators import CombineOP, SumOP, MeanOP, QuantileOP


def best_perslay_experiment(pwd_list, neurite_type = "apical_dendrite", feature = "projection"):
    wf = WeightFunction(grid=2, denominator=10000)
    t = torch.linspace(-500, 1500, 10)
    pt = GaussPT(t=t, sigma = 100)
    op_list = [SumOP(), MeanOP(), QuantileOP(q=0.5), QuantileOP(q=0.), QuantileOP(q=0.1), QuantileOP(q=0.25), QuantileOP(q=0.75), QuantileOP(q=0.9), QuantileOP(q=1.)]
    op = CombineOP(op_list)
    final_results = {}
    train_acc_list_layer = []
    val_acc_list_layer = []
    for layer in pwd_list.keys():
        print(pwd_list[layer])
        train_acc_list, _, val_acc_list, _, _ = perslay_experiment(pwd_list[layer], padding0 = True, test_fraction = 0.2, folds = 50,
                    point_transform = pt, weight_function = wf, operator = op,
                    hiddens = [16],
                    max_epochs = 75, batch_size = 8, nb_classes=len(pwd_list[layer]), neurite_type = neurite_type, feature = feature)
        
        #plot_train_val(train_acc_list, val_acc_list)
        val_score_mean = np.mean(val_acc_list, axis=0)
        val_score = val_score_mean[-1]
        final_results['Perslay_'+layer] = val_score
        train_acc_list_layer.append(train_acc_list)
        val_acc_list_layer.append(val_acc_list)
    return final_results, train_acc_list_layer, val_acc_list_layer