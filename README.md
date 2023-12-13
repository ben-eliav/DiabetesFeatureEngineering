# Feature Engineering Project

This project's goal is to try and find out whether GNNs can be used in the feature engineering process in order to 
estimate mutual information in heterogeneous populations. Heterogeneous populations are populations that can be 
divided into several subpopulations where each subpopulation has slightly different behaviors. The goal is to return 
for each subpopulation an ordered list of features that are most important in determining some target value for each 
subpopulation.

## Generating Synthetic Data
In order to generate synthetic data, we can run the file synthetic_data.py. Running this file will create a csv file 
containing entirely synthetic data. The synthetic data will have specific behaviors for each subpopulation, while 
still preserving a general idea of how each feature is supposed to behave. "synthetic_data.py" will also 
generate a text file describing how the synthetic data was created. Parameters that 
can be entered into synthetic_data:
1. name: The name of the file that will be created.
2. num_features: The amount of features in the dataset.
3. num_labels: The number of options for labels in the dataset.
4. num_samples: Expected number of samples in the dataset.
5. num_subpops: The number of subpopulations in the dataset.
6. seed: The seed for the random number generator.

Synthetic data is also generated using some hyperparameters which can be changed in the syntheticParametersConfigs.json file. The parameters are:
1. Subpop_size_std: Standard deviation of the subpopulation sizes.
2. Random_dist_weight1: Weight of the random distribution when creating a new distribution based on 
   base_distribution. This is regarding the distribution of the labels.
3. Random_dist_weight2: Weight of the random distribution when creating a new distribution for a feature that is 
   based on base_distribution. This is used for features, not labels. This is how much the feature distribution is 
   to be different in different subpopulations.
4. Max_bias: Determines the maximum "similarity" that a label-based feature can have to the distribution that the 
   label dictated for the feature. Number between 0 and 1, such that 0 would mean that the feature would have 
   nothing to do with the label, and 1 would mean that the bias could be any random number between 0 and 1.
5. Feature_important_prob: Probability of a feature being important. When we have a feature that is important, its 
   value will affect the label. This will be used when we have a feature that is based on more than one other 
   previous feature.

## Creating a Graph
Before we can train any models on a graph, we must create the graph first. We can use the file create_graph.py in 
order to turn any data that is saved as a csv file into a multiplex graph of features and subpopulations. It is 
important to define which features will define the subpopulations in the dataConfigs.json file. We can also bin 
numeric data, define null values, and specify which feature is the label. The graph creates a node for each feature 
in each subpopulation, such that there are (#features) * (#subpopulations) in the graph. We can use different 
methods to construct the edges between the nodes, the default method being defining each layer of the multiplex 
graph as a complete graph. The other methods are not recommended because they take MI between different features 
into consideration, which is of course supposed to be unknown during the training phase. 

"create_graph.py" takes the following parameters in order to build the graph:
1. task: The name of the task that the graph is being built for (the data).
2. method: The way that the edges are constructed between nodes. There are three options: complete, rank and filter. 
   Rank takes the top K most similar features using MI and filter takes all features with a certain MI (above a 
   given threshold) and passes an edge between the nodes.
3. k: In the case of "rank", defines how many of each node's most similar neighbors are taken into consideration 
   (for passing an edge). If the method is not rank, k is not used.
4. threshold: In the case of "filter", defines the MI threshold for passing an edge between two nodes. If the 
   method is not filter, threshold is not used.
5. self_loops: Boolean value that determines whether to add self loops to the graph.
6. directed: Boolean value that determines whether the graph is directed.
7. one_hot: Boolean value that determines whether to use one-hot encoding for the labels. If false, the encoding 
   will be the MI between the feature and all other features. This is less recommended because it uses MI with 
   features that are technically unknown during the training phase.

In the end, it will save the graph into a PyTorch (.pt) file. The graph is built as a heterogeneous graph. It also 
creates train, validation and test sets for the nodes of the graph. The different sets are created at random using 
the PyTorch built-in function RandomNodeSplit.

## Training a Model
Now that the graph is created, we can use a GNN model to predict MI values of each feature. We can use the file 
"main.py" to train a learning model on a graph that we have built. There are two parameters that can be entered into 
main.py:
1. graph: The name of the graph that we want to train on (file path).
2. model: The GNN model that we wish to work with (GAT, GraphSAGE, etc.).

The "main.py" file uses a class called Trainer in order to train the model. Trainer uses heterogeneous GNN's to train. 
The loss that is used is MSE and the optimizer is Adam. The model is trained for a certain number of epochs, and in 
each epoch the validation score using NDCG is calculated on the validation set. The best performing model's parameters 
are saved to another PyTorch (.pt) file. The model also compares itself to a baseline which does not utilize GNN's 
and plots the loss and NDCG in a graph.

---
Ben Eliav, 2023