import matplotlib.pyplot as plt
from constants import *


def plot_loss_ndcg(loss, ndcg, baseline=None, log=True):
    plt.plot(list(range(EPOCHS)), loss)
    plt.title('Training loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plot = plt.semilogy if log else plt.plot
    plot(list(range(EPOCHS)), ndcg)
    plt.title('Validation NDCG per epoch')
    plt.ylabel('NDCG')
    plt.xlabel('Epoch')
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='-', linewidth=0.5)
    plt.show()
