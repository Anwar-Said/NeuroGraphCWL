import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import os,random
from utils import *
import torch
import argparse
import numpy as np
import csv
import sys
from NeuroGraph.datasets import NeuroGraphStatic

name = sys.argv[1]
name = "data/HCPGender.pt"
epochs = 3
seed= 13
batch_size = 16
num_layers = 3
hidden = 64
lr = 1e-5
weight_decay = 0.0005
dropout = 0.5
hidden_mlp = 64

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu" # using only cpu
# dataset = NeuroGraphStatic(root = "data/",dataset_name=name)

data, slices = torch.load(name)
num_samples = slices['x'].size(0)-1
dataset = []
for i in range(num_samples):
    start_x = slices['x'][i]
    end_x = slices['x'][i + 1]
    x= data.x[start_x:end_x,:]
    start_ei = slices['edge_index'][i]
    end_ei = slices['edge_index'][i + 1]
    edge_index = data.edge_index[:,start_ei:end_ei]
    y = data.y[i]
    data_sample = Data(x =x, edge_index = edge_index, y=y)
    dataset.append(data_sample)

labels = [d.y.item() for d in dataset]

train_tmp, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2, stratify=labels,random_state=123,shuffle= True)
tmp = [dataset[index] for index in train_tmp] 
train_labels = [d.y.item() for d in tmp]
train_indices, val_indices = train_test_split(list(range(len(train_labels))),
 test_size=0.125, stratify=train_labels,random_state=123,shuffle = True)
train_dataset = [tmp[index] for index in train_indices]
val_dataset = [tmp[index] for index in val_indices]
test_dataset = [dataset[index] for index in test_indices]
print("dataset {} loaded with train {} val {} test {} splits".format(name,len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
num_features,num_classes = 1000, 2


criterion = torch.nn.CrossEntropyLoss()
def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:  
        data = data.to(device)
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y) 
        total_loss +=loss
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
    return total_loss/len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:  
        data = data.to(device)
        out = model(data)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset) 
model = ResidualGNNs(num_features,hidden,hidden_mlp,num_layers, num_classes).to(device)
print(model)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss, test_acc = [],[]
best_val_acc,best_val_loss = 0.0,0.0
val_acc_history, test_acc_history = [],[]

for epoch in range(epochs):
    loss = train(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    # if epoch%10==0:
    print("epoch: {}, loss: {}, val_acc:{}, test_acc:{}".format(epoch, np.round(loss.item(),6), np.round(val_acc,2),np.round(test_acc,2)))
    val_acc_history.append(val_acc)
    test_acc_history.append(test_acc)
print("best test and val acc:", round(np.max(test_acc_history),3),round(np.max(val_acc_history),3))
with open("results.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([np.max(val_acc_history), np.max(test_acc_history)])
