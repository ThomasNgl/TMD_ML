import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Put in utils or plots
def plot_train_val(train_score, val_score, title= 'accuracy of Perslay clf through learning'):
    train_score_mean = np.mean(train_score, axis=0)
    train_score_std = np.std(train_score, axis = 0)

    val_score_mean = np.mean(val_score, axis=0)
    val_score_std = np.std(val_score, axis = 0)

    x = range(len(train_score_mean))
    plt.plot(x, train_score_mean, 'k-')
    plt.fill_between(x, train_score_mean-train_score_std, train_score_mean+train_score_std, alpha = 0.5, label = 'Train set')
    plt.plot(x, val_score_mean, 'k-')
    plt.fill_between(x, val_score_mean-val_score_std, val_score_mean+val_score_std, alpha = 0.5, label = 'Validation set')
    plt.ylim([0.2,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.grid(axis='y')
    plt.legend()
    plt.show()

#plot model and each layer accuracy and weighted average accuracy
def plot_model_layers(results_list, color_continuous_scale = px.colors.sequential.Rainbow,range_color=None, width = None, height = None, figure_name='figure'):
    l2_acc_list = []
    l3_acc_list = []
    l4_acc_list = []
    l5_acc_list = []
    l6_acc_list = []

    mean_acc_list = []
    method_list = []
    i = False
    for results in results_list:
            for key in results.keys():

                if key[:key.index('_L')] not in method_list:
                    method_list.append(key[:key.index('_L')])
               
                layer = int(key[key.index('L')+1])
                acc = round(2*results[key],1)/2
                if layer == 2:
                    l2_acc_list.append(acc)
                if layer == 3:
                    l3_acc_list.append(acc)
                if layer == 4:
                    l4_acc_list.append(acc)
                if layer == 5:
                    l5_acc_list.append(acc)
                if layer == 6:
                    l6_acc_list.append(acc)
                    i = True
                if i :
                    mean_acc_list.append((41*l2_acc_list[-1]+62*l3_acc_list[-1]+70*l4_acc_list[-1]+159*l5_acc_list[-1]+130*l6_acc_list[-1])/(41+62+70+159+130))
                    i =False
        
    df = pd.DataFrame({'method': method_list, 'L2':l2_acc_list, 'L3':l3_acc_list, 'L4':l4_acc_list, 'L5':l5_acc_list, 'L6':l6_acc_list, 'mean':mean_acc_list})

    fig = px.parallel_categories(df, dimensions=['method', 'L2','L3','L4','L5','L6'],
                    color="mean", color_continuous_scale=color_continuous_scale,range_color=range_color,
                    labels={'mean':'mean', 'method':'method', 'layer':'layer'}, width=width, height=height)
    
    fig.write_image(figure_name+'.png')
    fig.show()
