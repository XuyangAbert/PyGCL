import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset,Planetoid


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        z = torch.cat(zs, dim=1)
        z = self.project(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z = self.encoder(x, edge_index, batch)
        z1 = self.encoder(x1, edge_index1, batch)
        z2 = self.encoder(x2, edge_index2, batch)
        return z, z1, z2

def train_node(encoder_model, contrast_model, data, optimizer):
  encoder_model.train()
  optimizer.zero_grad()
  z, z1, z2 = encoder_model(data.x, data.edge_index, data.batch)
  loss = contrast_model(h1=z1, h2=z2, batch=data.batch)
  loss.backward()
  optimizer.step()
  return loss.item()

def test_node(encoder_model, data):
  encoder_model.eval()
  z, _, _ = encoder_model(data.x, data.edge_index, data.batch)
  split = get_split(num_samples=z.size()[0], train_ratio=0.6, test_ratio=0.2)
  result = SVMEvaluator(linear=True)(z, data.y, split)
  return result

def node_classification():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    input_dim = max(dataset.num_features, 1)

    # aug1 = A.Identity()
    aug1 = A.PPRDiffusion(alpha=0.2)
    aug2 = A.FOSR(max_iterations=20)
    gconv = GConv(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.0001)
    test_results = []

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train_node(encoder_model, contrast_model, data, optimizer)
            if epoch % 5 == 0:
                test_result = test_node(encoder_model, data)
                test_results.append([epoch,
                                     test_result['micro_f1'],
                                     test_result['macro_f1'],
                                     loss])
            pbar.set_postfix({'loss': loss})
            pbar.update()
    return test_results
if __name__ == '__main__':
    test_results_node = node_classification()
