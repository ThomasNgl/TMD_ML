import torch
import torch.nn as nn
import random

# init: un model, un optimizer, un criterion
# fit_val: fit le model sur un train set et test sur un val set
# retourne l'accuracy et la loss
class Trainer(nn.Module):
    def __init__(self, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def evaluate(self, x, y):
        pred = self.model.forward(x)
        loss = self.criterion(pred, y).item()
        accuracy = (pred.argmax(1) == y).type(torch.float).sum().item()
        accuracy /= len(x)
        return accuracy, loss

    def fit_val(self, x_train, x_val, y_train, y_val, max_epochs, batch_size):
        val_acc_list = []
        train_acc_list = []

        val_loss_list = []
        train_loss_list = []
        
        train_size = len(x_train)
        iterations = int(train_size/batch_size)
        for epoch in range(max_epochs):
            if type(x_train) == list:
                random.shuffle(y_train)
                random.shuffle(x_train)
            else:
                permutation = torch.randperm(y_train.shape[0])
                y_train=y_train[permutation]
                x_train=x_train[permutation]
            self.model.eval()
            if epoch % 1 == 0 or epoch==0:
                with torch.no_grad():
                    acc_train, loss_train = self.evaluate(x_train, y_train)
                    train_acc_list.append(acc_train)
                    train_loss_list.append(loss_train)

                    acc_val, loss_val = self.evaluate(x_val, y_val)
                    val_acc_list.append(acc_val)
                    val_loss_list.append(loss_val)

            self.model.train()
            for i in range(iterations):
                self.optimizer.zero_grad()

                local_X, local_y = x_train[i*batch_size: (i+1)*batch_size], y_train[i*batch_size: (i+1)*batch_size]
                pred = self.model.forward(local_X)
                loss = self.criterion(pred, local_y)
                loss.backward()
                self.optimizer.step()

        return train_acc_list, train_loss_list, val_acc_list, val_loss_list