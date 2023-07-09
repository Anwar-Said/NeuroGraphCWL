from NeuroGraph import utils
import numpy as np
import sys,os


if __name__== "__main__":
    dataDir = sys.argv[1]
    # atlas_path = atlas_path
    # dataDir = "processed/"
   
    file_names = [file for file in os.listdir(dataDir) if file.endswith('_3.npy')]

    for f in file_names:
        Y = np.load(os.path.join(dataDir,f))
        Y = utils.construct_corr(Y)
        print("Processing completed for file ", f)
        np.save(os.path.join("processed",f.split("_")[0]+"_corr.npy"),Y)




