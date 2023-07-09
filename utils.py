import torch
from torch import nn
from torch_geometric.nn import aggr
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv
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