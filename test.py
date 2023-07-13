from torch_geometric.loader import DataLoader
from utils import ResidualGNNs
import torch
import csv
import sys

test_data = sys.argv[1]
# test_data = "data/test_data.pt"
num_layers = 3
hidden = 64
num_classes= 2
hidden_mlp = 64
num_features = 1000

model = ResidualGNNs(num_features,hidden,hidden_mlp,num_layers, num_classes)
print(model)

model.load_state_dict(torch.load("data/gcn_trained.pkl"))

test_dataset = torch.load(test_data)
test_loader = DataLoader(test_dataset, 1, shuffle= False)

def test(loader):
    model.eval()
    predictions = []
    for data in loader:  
        data = data
        out = model(data)  
        pred = out.argmax(dim=1)  
        predictions.append(pred.item())
    return  predictions



pred = test(test_loader)
with open("predictions.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["predictions"])
        writer.writerow([pred])
print("preditions have been saved to the predictions file!")






