import os
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from nilearn import image
from nilearn import plotting

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

    parser.add_argument("--datadir",
                        help="A directory containing img (Images) and cls (GT Segmentation) folder.")
    parser.add_argument("--imgsfx",default="_ana_strip.nii.gz",
                        help="Suffix for Image File")
    parser.add_argument("--labsfx",default="_segTRI_ana.nii.gz",
                        help="Suffix for Label File")
    parser.add_argument("--outdir",
                        help="Output DIrectory for slices.")
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
    Directory with 2d Data files
"""
def slicedatadir():
    if not os.path.exists(os.path.join(args.outdir,"data")):
        os.makedirs(os.path.join(args.outdir,"data","img"))
        os.makedirs(os.path.join(args.outdir,"data","cls"))
    return os.path.join(args.ourdir,"data")

"""
    Directory for 2d Vis files
"""
def slicevisdir():
    if not os.path.exists(os.path.join(args.outdir,"vis")):
        os.makedirs(os.path.join(args.outdir,"vis","img"))
        os.makedirs(os.path.join(args.outdir,"vis","cls"))
    return os.path.join(args.outdir,"vis")

"""
    Returns name for slice in TIF and PNG Format
    NOTE: USE TIF extension only for images. For Labels, keep using PNG
"""
def slicename(mriidx,sliceidx):
    return "{:02d}_{:03d}.tif".format(mriidx,sliceidx), "{:02d}_{:03d}.png".format(mriidx,sliceidx)

"""
    Dump all slices of Img and Cls files
"""
def dump_slices(idx):
    import pdb; pdb.set_trace()
    imgf = fname2path(idx2name(idx,"img"),"img")
    clsf = fname2path(idx2name(idx,"cls"),"cls")

    img = image.load_img(imgf)
    cls = image.load_img(clsf)

    imgnp = img.get_data()
    clsnp = cls.get_data()

    imgnp = np.squeeze(imgnp,axis=3)
    clsnp = np.squeeze(clsnp,axis=3)
    imgnp = np.transpose(imgnp,perm)
    clsnp = np.transpose(clsnp,perm)

    nslices,_,_ = imgnp.shape


    for n in range(nslices):
        savet2dfiles(imgnp[n],clsnp[n],slicedatadir(),slicevisdir(),slicename(idx,n))

"""
    Save Numpy array as .png file
"""
def savet2dfiles(imgnp,clsnp,datadirname,visdirname,slicename):
    tif = slicename[0]
    png = slicename[1]
    if not np.all(np.unique(clsnp)==0):
        # First save data files
        img = Image.fromarray(imgnp)
        img.save(os.path.join(datadirname,"img",tif))
        plt.imsave(os.path.join(datadirname,"cls",png),clsnp)

        # Now save the visualization data
        plt.imsave(os.path.join(visdirname,"img",png),imgnp,cmap='RdGy')
        plt.imsave(os.path.join(visdirname,"cls",png),clsnp)

        print("img/{} saved".format(slicename))
        print("cls/{} saved".format(slicename))

args = parse_args()

def main():
    VOL_LIST = TRAIN_VOL + VAL_VOL
    for vol in VOL_LIST:
        dump_slices(vol)
if __name__ == "__main__":
    main()
