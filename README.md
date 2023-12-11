# TMD_ML
Classification of neuron morphologies using TMD and supervised learning.
This repository contains tools to classify neuron morphologies with respect to their topological morphology descriptor, i.e., the persistence homology of the graph of a neuron.
[Kanari, L., Dłotko, P., Scolamiero, M., Levi, R., Shillcock, J., Hess, K., & Markram, H. (2018). A topological representation of branching neuronal morphologies. Neuroinformatics, 16, 3-13.](https://doi.org/10.1007/s12021-017-9341-1)
These methods were compared with 2 other methods. One is classic, some morphometrics are computed from the reconstructed neurons. Then, the extracted features are used by a XGB to classify the neurons. The other method uses the graph of the reconstructed neurons and a GNN to classify them. 

The experiments:

With TMD:
1) the vectorization methods are combined with supervised Ml algorithms (e.g. random forest,  LDA ...) to classify neuron morphologies with respect to their TMD. (In the special case of persistence image vectorization the ouput is a 2D array thus can be provided to a CNN for classification).
2) The Perslay model learns the vectorization and the classification, thus classify the TMD of the neuron morphologies.

Without TMD:
3) Features from reconstructed neurons are collected and a XGB is used to classify the neuron morphologies with respect to the selected morphometrics.
4) A GNN classify the graph of the reconstructed neurons. 

Repository content:
1) data:
    Contains the reconstructed PCs and INs, their pwds and a file of morphometrics config. TODO [Link to reconstructed morpho](URL)
2) experiments:
    Contains the classification experiments for PCs and INs. The results are stored in the folder results.
    There are two ploting experiments and the resulting figures are stored in experiments.
    TODO save figures in folder results.
3) methods:
    Contains the methods used to run the experiments with the different models and classifiers.
    Most of the methods call the TMD (https://github.com/BlueBrain/TMD.git)
    in gnn_methods.py a GNN (ManNet) and other methods to process dataset are called from morphoclass library (https://github.com/BlueBrain/morphoclass.git)
    vect_methods.py contains a decade of vectorization methods defined in [Ali, D., Asaad, A., Jimenez, M. J., Nanda, V., Paluzo-Hidalgo, E., & Soriano-Trigueros, M. (2023). A survey of vectorization methods in topological data analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence.](10.1109/TPAMI.2023.3308391)

4) others:
    Contains some classes to run experiments. 
    CNN.py contains an implementation of a conv net work from (https://github.com/BlueBrain/morphoclass.git)
    cocob.py contains a Pytorch implementation of the cocob optimizer (https://github.com/nocotan/cocob_backprop.git)
    Trainer.py contains a class Trainer for pytorch models.

5) perslay_model:
    Contains a Pytorch version of the Perslay model. [Carriere, M., Chazal, F., Datashape, I. S., Ike, Y., Lacombe, T., Royer, M., & Umeda, Y. (2019). Perslay: A simple and versatile neural network layer for persistence diagrams. stat, 1050, 5.](https://github.com/MathieuCarriere/perslay.git)
    The authors provide an implementation of their model with tensorflow library.

6) results:
    Store the results from the experiments in pkl files

7) src_morphoclass: (https://github.com/BlueBrain/morphoclass.git)

8) tda_toolbox: TODO cite the github 



