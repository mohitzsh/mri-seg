import os
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from nilearn import image

TRAIN_VOL = [1,2,3,4,5,6,7,8,9,10,11,12,13]
VAL_VOL = [14,15,16,17,18]
currdir = os.path.dirname(os.path.realpath(__file__))
VIS_VOL = 1
prfx = "IBSR"
imgsfx = "ana_strip.nii.gz"
clssfx = "segTRI_ana.nii.gz"
perm = (2,1,0)

"""
    Argument Parser
"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("datadir",
                        help="A directory containing img (Images) and cls (GT Segmentation) folder.")
    parser.add_argument("--mode",choices=("vis","dump"),default="dump",
                        help="visualization vs dump mode")
    parser.add_argument("--imgsfx",default="_ana_strip.nii.gz",
                        help="Suffix for Image File")
    parser.add_argument("--labsfx",default="_segTRI_ana.nii.gz",
                        help="Suffix for Label File")

    return parser.parse_args()

"""
    Return dump filename from file index
"""
def idx2name(idx,ftype="img"):
    if ftype=="img":
        return "{}_{:02d}_{}".format(prfx,idx,imgsfx)
    else:
        return "{}_{:02d}_{}".format(prfx,idx,clssfx)

"""
    Return absolute path of the 3d volume file
"""
def fname2path(fname,ftype="img"):
    return os.path.join(args.datadir,ftype,fname)

"""
    Directory with visualization files
"""
def visdir():
    return os.path.join(currdir,"..","data","2d","vis")

"""
    Directory with 2d Data files
"""
def slicedatadir():
    return os.path.join(currdir,"data","2d","data")

"""
    Path to visualization files for mriidx
"""
def idxvisdir(mriidx):
    return os.path.join(visdir(),"IBSR_{}".format(mriidx))

"""
    Return filename for saved combined img and cls
"""
def slicename(mriidx,sliceidx):
    return "{:02d}_{:03d}.png".format(mriidx,sliceidx)
"""
    Dump all slices of Img and Cls files
"""
def dump_slices(idx):
    plt.ion()
    imgf = fname2path(idx2name(idx,"img"),"img")
    clsf = fname2path(idx2name(idx,"cls"),"cls")

    img = image.load_img(imgf)
    cls = image.load_img(clsf)

    imgnp = img.get_data()
    imgnp = np.squeeze(imgnp,axis=3)
    imgnp = np.transpose(imgnp,perm)

    clsnp = cls.get_data()
    clsnp = np.squeeze(clsnp,axis=3)
    clsnp = np.transpose(clsnp,perm)

    # import pdb; pdb.set_trace()

    nslices,_,_ = imgnp.shape

    if not os.path.exists(idxvisdir(idx)):
        os.makedirs(idxvisdir(idx))

    [savet2dfiles(imgnp[n],clsnp[n],slicedatadir(),slicename(idx,n)) for n in range(nslices)]

"""
    Save Numpy array as .png file
"""
def savet2dfiles(imgnp,clsnp,dirname,slicename):
    img = Image.fromarray(imgnp,'P')
    cls = Image.fromarray(clsnp,'P')

    img.save(os.path.join(dirname,"img",slicename))

    cls.save(os.path.join(dirname,"cls",slicename))

args = parse_args()

def main():
    if args.mode == "vis":
        dump_slices_vis(VIS_VOL)
if __name__ == "__main__":
    main()
