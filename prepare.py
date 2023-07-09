from NeuroGraph import utils
import numpy as np
import sys,os
import pickle
import pandas as pd
from torch_geometric.data import Data
import torch

if __name__== "__main__":
    
    ids_file = sys.argv[1]
    labels = sys.argv[2]
    dataDir = sys.argv[3]
    # ids_file = "data/ids.pkl"
    # labels = "data/labels.csv"
    # dataDir = "processed"
   
    # file_names = [file for file in os.listdir(dataDir) if file.endswith('corr.npy')]
    with open(ids_file,'rb') as f:
        ids = pickle.load(f)
    
    labels = pd.read_csv(labels).set_index('Subject')['Gender']
    labels = labels.to_dict()
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    ids = ids[:3]
    dataset = []
    for iid in ids:
        corr = np.load(os.path.join(dataDir, (iid + "_corr.npy")))
        label = labels.get(int(iid))
        y = 1 if label=="M" else 0
        data = utils.construct_data(corr, y, threshold= 10)
        dataset.append(data)
    print("dataset processed!")
    torch.save(dataset,"dataset_graph.pt")
    print("dataset has been preprocessed!")
        


   