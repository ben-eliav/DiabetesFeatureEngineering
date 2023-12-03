from preprocessing import Preprocessor
from createGraph import GraphCreator
from constants import *
from train_eval import Trainer
import models
import torch

if __name__ == '__main__':
    print('Loading graph...')
    graph = torch.load('Graphs/Synthetic1_complete_selfLoops_directed_oneHot.pt')
    print('Done loading graph.')
    print('Beginning training...')
    trainer = Trainer(graph, models.GAT(), 'Synthetic1_complete_selfLoops_directed_oneHot')
    trainer.learn()
    print('Done training.')
