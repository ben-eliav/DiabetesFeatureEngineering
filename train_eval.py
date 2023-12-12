import visualization
from constants import *
import torch
from torch_geometric.nn import to_hetero
from sklearn.metrics import ndcg_score


class Trainer:
    def __init__(self, data, model, name):
        self.name = f'{model.name}_{name}'
        self.data = data
        self.model = to_hetero(model, self.data.metadata())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=DECAY)
        self.criterion = torch.nn.MSELoss()

    def train_subpop(self, subpop):
        """
        Train model on single subpopulation.

        :return: Loss value based on "self.criterion" (MSELoss)
        """
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)[subpop]
        loss = self.criterion(out[self.data[subpop].train_mask], self.data[subpop].y[self.data[subpop].train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss), out

    def train(self):
        loss_vals = {}
        ndcg = {}
        for key in self.data.x_dict:
            loss_vals[key], out = self.train_subpop(key)
            ndcg[key] = ndcg_score(self.data[key].y[self.data[key].train_mask].detach().numpy().reshape(1, -1),
                                   out[self.data[key].train_mask].detach().numpy().reshape(1, -1))
        return loss_vals, sum(loss_vals.values()), ndcg

    def test_subpop(self, subpop, val):
        """
        Calculate SSE in certain layer.

        :param val: True if validation, False if test
        """
        self.model.eval()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)[subpop]
        mask = self.data[subpop].val_mask if val else self.data[subpop].test_mask
        sse = (out[mask] - self.data[subpop].y[mask]).pow(2).sum().item()
        return ndcg_score(self.data[subpop].y[mask].detach().numpy().reshape(1, -1),
                          out[mask].detach().numpy().reshape(1, -1)), sse

    def test(self, validation=True):
        """
        Testing each layer separately

        :param validation: True if validation, False if test
        :return: NDCG values for each subpop, sum of NDCG values, sse loss
        """
        self.model.eval()
        ndcg = {}
        sse = 0
        for key in self.data.x_dict:
            ndcg_subpop, sse_subpop= self.test_subpop(key, validation)
            ndcg[key] = ndcg_subpop
            sse += sse_subpop
        return ndcg, sum(ndcg.values()), sse

    def baseline_prediction(self):
        """
        Calculate baseline prediction (mean of training labels)

        :return: Tensor of predictions for all features
        """
        layer_size = self.data.x_dict[list(self.data.x_dict.keys())[0]].shape[0]
        predictions = []
        for feature_index in range(layer_size):
            labels = [self.data[subpop].y[feature_index] for subpop in self.data.x_dict
                      if self.data[subpop].train_mask[feature_index]]
            predictions.append(sum(labels) / len(labels))
        return torch.tensor(predictions, dtype=torch.float32)

    def baseline_ndcg(self, validation=True):
        """
        Calculate NDCG for baseline prediction

        :param validation: True if validation, False if test
        """
        predictions = torch.tensor(self.baseline_prediction())

        ndcg = {}
        for key in self.data.x_dict:
            mask = self.data[key].val_mask if validation else self.data[key].test_mask
            ndcg[key] = ndcg_score(self.data[key].y[mask].detach().numpy().reshape(1, -1),
                                   predictions[mask].numpy().reshape(1, -1))
        return ndcg, sum(ndcg.values())

    def learn(self):
        best_ndcg = -1
        loss_epochs = []
        val_ndcg_epochs = []
        for epoch in range(EPOCHS):
            _, loss, train_ndcg = self.train()
            loss_epochs.append(loss)
            val_ndcg_dict, val_ndcg, val_sse = self.test()
            val_ndcg_epochs.append(val_ndcg)
            print(f'Epoch: {epoch}, Train Loss: {loss}, Val Loss: {val_sse}')
            print(f'Train NDCG: {train_ndcg}')
            print(f'Val NDCG: {val_ndcg_dict}')
            if best_ndcg < val_ndcg:
                best_ndcg = val_ndcg
                torch.save(self.model.state_dict(), f'Models/{self.name}.pt')
        visualization.plot_loss_ndcg(loss_epochs, val_ndcg_epochs, self.baseline_ndcg()[1])
