
import nibabel as nib
import numpy as np
import sys,os
from nilearn.image import load_img
from pathlib import Path


def parcellation(fmri, atlas_path):
    """
    Prepfrom brain parcellation

    Args:

    fmri (numpy array): fmri image
    rois (int): {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}, optional,
    Number of regions of interest. Default=1000.
    
    """
    # roi = fetch_atlas_schaefer_2018(n_rois=n_rois,yeo_networks=17, resolution_mm=2)
    # atlas = load_img(roi['maps'])
    # atlas_path = 'data/roi/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
    atlas = nib.load(atlas_path)

    volume = atlas.get_fdata()
    subcor_ts = []
    for i in np.unique(volume):
        if i != 0: 
            bool_roi = np.zeros(volume.shape, dtype=int)
            bool_roi[volume == i] = 1
            bool_roi = bool_roi.astype(bool)
            roi_ts_mean = []
            for t in range(fmri.shape[-1]):
                roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
            subcor_ts.append(np.array(roi_ts_mean))

    Y = np.array(subcor_ts).T
    return Y

if __name__== "__main__":
    dataDir = sys.argv[1]
    atlas_path = sys.argv[2]
    # atlas_path = atlas_path
    
    # dataDir = "data/raw/"
   
    file_names = [file for file in os.listdir(dataDir) if file.endswith('.nii.gz')]
    file_names_all = [file for file in os.listdir(dataDir)]

    # print("file names:", file_names_all)
    # print(Path.cwd())

    if not os.path.exists("processed"):
        os.mkdir("processed")
    for f in file_names:
        img_path = os.path.join(dataDir,  f)
        # img_path = "data/raw/"+f

        print(img_path)
        img = load_img(img_path)
        fmri = img.get_fdata()
        Y = parcellation(fmri,atlas_path)
    

        print("Processing completed for file ", f)
        np.save(os.path.join("processed",f.split(".")[0]+"_1.npy"),Y)

    
    