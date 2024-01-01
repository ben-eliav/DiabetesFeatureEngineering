import torch
from sklearn.metrics import mutual_info_score
import constants
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


class GraphCreator:

    def __init__(self, subpops, subpop_cols, target, directed=True, start_build=True, one_hot=False):
        self.subpops = subpops  # List of pandas dataframes
        self.subpop_cols = subpop_cols  # List of column names that describe the subpopulations
        self.directed = directed
        self.target = target
        self.one_hot = one_hot
        self.data = HeteroData()  # TODO: Implement undirected graph
        self.layers = len(subpops)
        self.features = [i for i in list(subpops[0].columns) if i not in [self.target, *self.subpop_cols]]
        self.layer_size = len(self.features)
        if start_build:
            self.features_to_nodes()
            self.add_inter_layer()

    @property
    def subpops_titles(self):
        return [''.join(list(subpop[self.subpop_cols].iloc[0])) for subpop in self.subpops]

    def mi_between_features(self, subpop_num):
        """
        Finds the mutual information between each pair of features in the given subpopulation.
        """
        mi = []
        for i, feature in enumerate(self.features):
            mi.append([])
            for j, other_feature in enumerate(self.features):
                if j < i:
                    mi[i].append(mi[j][i])  # Symmetric matrix
                else:
                    subpop = self.subpops[subpop_num]
                    mi[i].append(mutual_info_score(subpop[feature], subpop[other_feature]))
        return mi

    @constants.timeit
    def features_to_nodes(self):
        """
        Converts all features in the subpopulations to nodes in the graph.
        """
        for i, subpop in enumerate(self.subpops):
            target_values = []
            subpop_name = self.subpops_titles[i]
            self.data[subpop_name].num_nodes = self.layer_size
            for j, feature in enumerate(self.features):
                target_values.append(mutual_info_score(subpop[feature], subpop[self.target]))
            if self.one_hot:
                self.data[subpop_name].x = torch.eye(self.layer_size)
            if not self.one_hot:
                self.data[subpop_name].x = torch.Tensor(self.mi_between_features(i))
            self.data[subpop_name].y = torch.Tensor(target_values).unsqueeze(-1)

    @constants.timeit
    def add_inter_layer(self):
        """
        Adds edges between nodes and their peers in different layers.
        """
        for i, subpop_title in enumerate(self.subpops_titles):
            for j, other_title in enumerate(self.subpops_titles):
                if i == j:  # No inter layer in same layer
                    continue
                if self.directed or j > i:
                    edges = [list(range(self.layer_size)), list(range(self.layer_size))]
                    self.data[subpop_title, 'inter', other_title].edge_index = torch.tensor(edges, dtype=torch.long)
                else:  # Not directed and j > i - j will only increase. No need to continue
                    break

    def rank_intra(self, k, subpop_num, self_loops=True):
        """
        Finds the K nearest neighbors of each feature (based on mutual information) in the given subpopulation.
        """
        neighbors = []
        mi_all = self.mi_between_features(subpop_num) if self.one_hot \
            else self.data[self.subpops_titles[subpop_num]].x
        for i in range(self.layer_size):  # For each feature run over MI with all the other features
            neighbors.append([])
            for j in range(self.layer_size):
                if self_loops or i != j:
                    mi = float(mi_all[i][j])
                    if mi != 0:
                        neighbors[i].append((j, mi))
            neighbors[i].sort(key=lambda x: x[1], reverse=True)
            if len(neighbors[i]) > k:
                neighbors[i] = neighbors[i][:k]
        return neighbors

    @constants.timeit
    def filter_intra(self, threshold, subpop_num, self_loops=True):
        """
        Connects feature in subpopulation to all other features that have MI higher than certain threshold
        """
        neighbors = []
        mi_all = self.mi_between_features(subpop_num) if self.one_hot \
            else self.data[self.subpops_titles[subpop_num]].x
        for i in range(self.layer_size):  # For each feature run over MI with all the other features
            neighbors.append([])
            for j in range(self.layer_size):
                if self_loops or i != j:
                    mi = float(mi_all[i][j])
                    if mi > threshold:
                        neighbors[i].append((j, mi))
        return neighbors

    def complete_intra(self, self_loops=True):
        """
        Connects feature in subpopulation to all the other features
        """
        neighbors = []
        for i in range(self.layer_size):  # For each feature run over MI with all the other features
            neighbors.append([])
            for j in range(self.layer_size):
                if self_loops or i != j:
                    neighbors[i].append((j, 1))
        return neighbors

    def add_intra_layer(self, complete=False, **kwargs):
        """
        Adds edges between nodes and their peers in the same layer. Notice graph is undirected therefore a node could
        technically be connected to more than K neighbors in the case of ranking (KNN).

        :param complete: If True, connects each node to all other nodes in the same layer
        :param kwargs: K in the case of ranking or threshold in the case of filtering. Dictates the method that will be
        used
        """

        def neighbors(x):
            loops = kwargs['self_loops'] if 'self_loops' in kwargs else True
            if complete:
                return self.complete_intra(loops)
            try:
                return self.rank_intra(kwargs['k'], x, loops) if 'k' in kwargs \
                    else self.filter_intra(kwargs['threshold'], x, loops)
            except KeyError:
                raise KeyError('Must include parameter k or threshold in kwargs, or method must be complete')

        for i, name in enumerate(self.subpops_titles):
            intra_edges = [[], []]
            connect = neighbors(i)
            for j in range(self.layer_size):
                for neighbor in connect[j]:
                    intra_edges[0].append(j)
                    intra_edges[1].append(neighbor[0])
            self.data[name, 'intra', name].edge_index = torch.tensor(intra_edges, dtype=torch.long)

    def add_masks(self):
        splitter = T.RandomNodeSplit(num_val=0.2, num_test=0.2)
        self.data = splitter(self.data)


def main():
    import preprocessing
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Synthetic1', help='Dataset to be used')
    parser.add_argument('--method', type=str, default='complete', help='Method to be used for intra layer edges')
    parser.add_argument('--k', type=int, default=constants.K, help='Number of neighbors to connect to')
    parser.add_argument('--threshold', type=float, default=constants.THRESHOLD, help='MI threshold for edge existence')
    parser.add_argument('--self_loops', type=bool, default=True, help='Whether to connect nodes to themselves')
    parser.add_argument('--directed', type=bool, default=True, help='Whether to create directed graph')
    parser.add_argument('--one_hot', type=bool, default=True, help='Whether to use one hot encoding')
    args = parser.parse_args()

    configs = json.load(open('dataConfigs.json'))
    if 'Synthetic' not in args.task:
        task = configs[args.task]
        preprocessor = preprocessing.Preprocessor(task)
    else:
        task = configs['Synthetic']
        preprocessor = preprocessing.Preprocessor(task, args.task[9:])
    subpops = preprocessor.interpret_data()
    print('Done with preprocessing')
    graph_creator = GraphCreator(subpops, task['subpops'], task['target'], args.directed, one_hot=args.one_hot)
    if args.method == 'filter':
        graph_creator.add_intra_layer(threshold=args.threshold, self_loops=args.self_loops)
    elif args.method == 'rank':
        graph_creator.add_intra_layer(k=args.k, self_loops=args.self_loops)
    elif args.method == 'complete':
        graph_creator.add_intra_layer(complete=True)
    print("Adding masks")
    graph_creator.add_masks()
    print('Starting to save')
    if args.method == 'complete':
        file = f'Graphs/{args.task}_complete_{"selfLoops" if args.self_loops else ""}' \
               f'_{"directed" if args.directed else ""}' \
               f'_{"oneHot" if args.one_hot else ""}.pt'
    elif args.method == 'filter':
        file = f'Graphs/{args.task}_filter_{args.threshold}_{"selfLoops" if args.self_loops else ""}' \
               f'_{"directed" if args.directed else ""}' \
               f'_{"oneHot" if args.one_hot else ""}.pt'
    elif args.method == 'rank':
        file = f'Graphs/{args.task}_rank_{args.k}_{"selfLoops" if args.self_loops else ""}' \
               f'_{"directed" if args.directed else ""}' \
               f'_{"oneHot" if args.one_hot else ""}.pt'
    else:
        file = f'Graphs/{args.task}_unknown_method.pt'

    torch.save(graph_creator.data, file)


if __name__ == '__main__':
    main()
