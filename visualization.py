import matplotlib.pyplot as plt
from constants import *


def plot_loss_ndcg(losses, ndcgs, baseline=None, log=True):
    plot = plt.semilogy if log else plt.plot
    plot(list(range(EPOCHS)), losses[0], label='Training loss')
    plot(list(range(EPOCHS)), losses[1], label='Validation loss')
    plt.legend()
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(list(range(EPOCHS)), ndcgs[0], label='Training NDCG')
    plt.plot(list(range(EPOCHS)), ndcgs[1], label='Validation NDCG')
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='-', linewidth=0.5, label='Baseline NDCG')
    plt.legend()
    plt.title('NDCG per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.show()
