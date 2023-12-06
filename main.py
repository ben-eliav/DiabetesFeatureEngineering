from preprocessing import Preprocessor
from create_graph import GraphCreator
from constants import *
from train_eval import Trainer
import models
import torch


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default='Synthetic1_complete_selfLoops_directed_oneHot.pt',
                        help='Graph to be used')
    parser.add_argument('--model', type=str, default='GAT', help='Model to be used')
    args = parser.parse_args()
    print('Loading graph...')
    try:
        graph = torch.load(f'Graphs/{args.graph}')
        print('Done loading graph.')
        print('Beginning training...')
        model = getattr(models, args.model)()
        trainer = Trainer(graph, model, args.graph)
        trainer.learn()
        print('Done training.')
    except FileNotFoundError:
        print("Graph not found. Create graph using create_graph.py")


if __name__ == '__main__':
    main()
