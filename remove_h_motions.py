from NeuroGraph import utils
import numpy as np
import sys,os


if __name__== "__main__":
    dataDir = sys.argv[1]
    # reg_path = "data/raw"
    # dataDir = "processed/"
    reg_path = sys.argv[2]
   
    file_names = [file for file in os.listdir(dataDir) if file.endswith('_2.npy')]

    for f in file_names:
        Y = np.load(os.path.join(dataDir,f))
        regs = np.loadtxt(os.path.join(reg_path,f.split("_")[0]+".txt"))
        M = utils.regress_head_motions(Y, regs)
        print("Processing completed for file ", f)
        np.save(os.path.join("processed",f.split("_")[0]+"_3.npy"),M)




