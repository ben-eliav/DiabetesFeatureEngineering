from preprocessing import Preprocessor
from createGraph import GraphCreator
from constants import *
from train_eval import Trainer
import models
import torch

if __name__ == '__main__':
    graph = torch.load('Graphs/Synthetic1_complete_selfLoops_directed_oneHot.pt')
    trainer = Trainer(graph, models.GraphSAGE(), 'Synthetic1_complete_selfLoops_directed_oneHot')
    print(trainer.name)
    trainer.learn()
