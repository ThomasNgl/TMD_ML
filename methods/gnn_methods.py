import torch
from tqdm import tqdm
import numpy as np

import morphoclass as mc 
from morphoclass.data import MorphologyDataset
from morphoclass.data.morphology_data_loader import MorphologyDataLoader
from morphoclass.training.trainers import Trainer
from morphoclass.models import ManNet

def gnn_experiment(path, layers_list, pre_transform):
    final_results = {}
    # A list that contains the lists (one for each layer) that contains the accuracies for each fold
    # One for the training and one for the validation
    train_acc_list_layer = []
    val_acc_list_layer = []
    # For each layer of the cortex
    for layer in layers_list:
        # Get the dataset that contains the graph of the neurons of the layer 
        dataset = MorphologyDataset.from_structured_dir(path, layer=layer, pre_transform=pre_transform)
        mc.utils.make_torch_deterministic()
        mc.training.reset_seeds(numpy_seed=0, torch_seed=0)
        #STRATIFIED
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
        
        n_splits = 2
        # 80 % of the samples in dataset are use for training
        max_index_train = int(0.8*len(dataset))
        
        # A list that contains the accuracies for each split
        # One for the training and one for the validation
        train_acc_list = []
        val_acc_list = []
        for _ in range(n_splits):
            # Get the number of m-types
            nb_classes = max(dataset.ys)+1
            
            # Initialize the Trainer of the gnn model 
            model = ManNet(n_classes=nb_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
            man_net_trainer = Trainer(model, dataset, optimizer, MorphologyDataLoader)
            
            # Get the samples used for training and the samples used for validation
            # Shuffle and split
            indices = torch.randperm(len(dataset))
            train_idx = indices[:max_index_train]
            val_idx = indices[max_index_train:]

            # Fit/Train the model on the training set and evaluate with validation set
            history = man_net_trainer.train(batch_size = 16, train_idx=train_idx, val_idx = val_idx, n_epochs=75, progress_bar=tqdm)
            
            train_acc_list.append(history['train_acc'])
            val_acc_list.append(history['val_acc'])
        
        # Get the mean accuracy of the gnn for each layer 
        val_score_mean = np.mean(val_acc_list, axis=0)
        # Validation score at the last epoch
        val_score = val_score_mean[-1]
        final_results['gnn_' + layer] = val_score

        train_acc_list_layer.append(train_acc_list)
        val_acc_list_layer.append(val_acc_list)
    return final_results, train_acc_list_layer, val_acc_list_layer