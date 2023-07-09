from NeuroGraph import utils
import numpy as np
import sys,os


if __name__== "__main__":
    dataDir = sys.argv[1]
    # atlas_path = atlas_path
    # dataDir = "processed/"
   
    file_names = [file for file in os.listdir(dataDir) if file.endswith('_1.npy')]

    for f in file_names:
        Y = np.load(os.path.join(dataDir,f))
        Y = utils.remove_drifts(Y)
        print("Processing completed for file ", f)
        np.save(os.path.join("processed",f.split("_")[0]+"_2.npy"),Y)




