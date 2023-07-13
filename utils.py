import torch
from torch import nn
from torch_geometric.nn import aggr
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv,GATConv,MLP
import math
from torch.nn import Conv1d, MaxPool1d, ModuleList
import torch.nn.functional as F

softmax = torch.nn.LogSoftmax(dim=1)


class ResidualGNNs(torch.nn.Module):
    def __init__(self, num_features, hidden_channels,hidden, num_layers, num_classes,k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        
        if num_layers>0:
            self.convs.append(GCNConv(num_features, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        # self.attention = Attention(input_dim1, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), num_classes),
        )
    

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                # xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                xx = self.aggr(xx,batch)
                # h.append(torch.stack([t.flatten() for t in xx]))
                h.append(xx)
        
        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        # x = torch.stack(h, dim=0)
        x = torch.cat((x,h),dim=1)
        x = self.mlp(x)
        return x

class GNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels, num_layers, k=0.6):
        super().__init__()
        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.sort_aggr = aggr.SortAggregation(self.k)
        self.convs = ModuleList()
       
        self.convs.append(GATConv(train_dataset.num_features, hidden_channels))
#         self.convs.append(GNN(1433, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        self.convs.append(GATConv(hidden_channels, 1))
#         self.lin = Linear(hidden_channels*(num_layers+1), dataset.num_classes)
        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, 32, args.num_classes], dropout=0.5, batch_norm=False)
    def reset_parameters(self):
        # for conv in self.convs:
        #     conv.reset_parameter()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
#             x = conv(x, edge_index).tanh()
        x = torch.cat(xs[1:], dim=-1)

        
        x = self.sort_aggr(x,batch)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        x = self.mlp(x)
        return x


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x