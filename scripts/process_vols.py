import argparse
import nibabel as nib
import numpy as np
from subprocess import call
import sys
import os
from scipy.ndimage import affine_transform
from scipy.ndimage import gaussian_filter
import shutil
from scipy import stats
from skimage import exposure
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_name")
    parser.add_argument("--out_name")
    parser.add_argument("--out_affine")
    parser.add_argument("--atlas")
    parser.add_argument("--in_affine")
    parser.add_argument("--mode",choices=("iso_img","iso_cls","reg","resample_img","resample_cls","sgauss","ahe","center"))
    # parser.add_argument("--input_dir",
    #                     help="Directory with img and cls folders")
    # parser.add_argument("--output_dir",
    #                     help="Output Directory. It will create img, cls folders with processed brain volumes")
    # parser.add_argument("--norm",action='store_true',
    #                     help="Normalize input to be between 0 and 1")
    # parser.add_argument("--affine",action="store_true",
    #                     help="Affine register the Image to ICMB-MNI152 nonlinear atlas")
    # parser.add_argument("--atlas",
    #                     help="Path to ICMB-MNI152 non-linear atlas")
    # parser.add_argument("--isotropic",action="store_true",
    #                     help="Make the volumes isotropic")
    # parser.add_argument("--threshmin",type=float,default=0.02,
    #                     help="Set all intensities below this to 0")
    # parser.add_argument("--n4",action="store_true",
    #                     help="Apply N4 Bias field correction")
    # parser.add_argument("--img_suffix",default="_ana_strip.nii.gz",
    #                     help="Image Suffix")
    # parser.add_argument("--cls_suffix",default="_segTRI_ana.nii.gz",
    #                     help="Class Suffix")
    # parser.add_argument("--skip_images",action="store_true")
    # parser.add_argument("--histeq",action="store_true")
    return parser.parse_args()

"""
    Returns vol, affine
"""
def read_vol(in_filepath):
    img = nib.load(in_filepath)
    img_np = img.get_data()
    img_affine = img.affine
    img_zoom = img.header.get_zooms()[:3]
    return img_np,img_affine,img_zoom

def write_vol(data, affine, out_filepath):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, out_filepath)
    return out_filepath

def sub_gauss_smooth_img(in_fname,out_fname):
    perm = (2,1,0)
    data,affine,_ = read_vol(in_fname)
    data = np.transpose(data,perm)

    # Calculate the gaussian kernel 31x31
    kernel = np.zeros((31,31),dtype='float32')
    kernel[15,15] = 1
    kernel = gaussian_filter(kernel,sigma=5)

    # Convolve the filter on each slice
    data = data[:,None,:,:] # Fake channel dimension
    data = torch.from_numpy(data).float()
    kernel = kernel[None,None,:,:]
    kernel = torch.from_numpy(kernel).float()
    if torch.cuda.is_available():
        data = data.cuda()
        kernel = kernel.cuda()
    data_smooth = F.conv2d(Variable(data),Variable(kernel),padding=15)

    new_data = data - data_smooth.data

    if torch.cuda.is_available():
        new_data = new_data.cpu().numpy()
    else:
        new_data = new_data.numpy()
    new_data = np.squeeze(new_data)
    new_data = np.transpose(new_data,perm)
    write_vol(new_data,affine,out_fname)

"""
    Copy file from one folder to another
"""

def copy_file(source,destination):
    call([
        "cp",
        source,
        destination
    ])
"""
    Apply Bias Field correction to the images
"""
def N4BiasField(fname):
    out_filename = add_suffix(fname,"n4")
    call([
        "N4BiasFieldCorrection",
        "-v",
        "-d",str(3),
        "-i",fname,
        "-o",out_filename,
        "-s",str(4),
        "-b","[200]",
        "-c", "[50x50x50x50,0.000001]",
        "-r",str(1)
    ])

    return out_filename

"""
    Apply Histogram Equalization
"""
def histogram_equalization(in_fname,out_fname):
    perm = (2,1,0)
    data,affine,_ = read_vol(in_fname)
    data = np.transpose(data,perm)
    img = sitk.GetImageFromArray(np.copy(data))
    img = sitk.AdaptiveHistogramEqualization(img)
    data_clahe = sitk.GetArrayFromImage(img)
    data = np.transpose(data_clahe,perm)
    return write_vol(data,affine,out_fname)

def standardize(in_fname,out_fname,apply_std=True,eps=10e-5):
    perm = (2,1,0)
    data,affine,_ = read_vol(in_fname)
    data = data.astype('float')
    data = np.transpose(data,perm)

    mean = np.zeros(data.shape,dtype='float')
    mean[...] = np.mean(data.reshape((data.shape[0],-1)),axis=1)[:,None,None]

    if apply_std:
        std = np.zeros(data.shape,dtype='float')
        std[...] = np.std(data.reshape((data.shape[0],-1)),axis=1)[:,None,None]
        data[std!=0] = (data[std!=0] - mean[std!=0])/std[std!=0]
    else:
        data = data - mean
    data = np.transpose(data,perm)
    return write_vol(data,affine,out_fname)

def affine_registration(in_img_fname,in_ref_fname,out_img_fname,out_affine_fname):
    call(["reg_aladin",
        "-noSym", "-speeeeed", "-ref", in_ref_fname ,
        "-flo", in_img_fname,
        "-res", out_img_fname,
        "-aff", out_affine_fname,
        "-pad",str(0),
        "-maxit",str(10),
        "-ln",str(6),])

"""
    mode = 0: NN
    mode = 1: Linear
    mode = 2: bilinear
    mode = 3: Cubic
    mode = 4: Not sure

    NOTE: use mode = 0 for labels and mode = 2 for MRI Images
"""
def affine_transform_nifty(in_img_fname,in_ref_fname,in_affine_fname,out_fname,mode=0):
    call(["reg_resample",
        "-ref", in_ref_fname,
        "-flo", in_img_fname,
        "-res", out_fname,
        "-trans", in_affine_fname,
        "-inter", str(mode)])
"""
    Reslice without dipy
"""
def reslice(data,affine,zooms,new_zooms,order=1, mode='constant', cval=0):
    new_zooms = np.array(new_zooms, dtype='f8')
    zooms = np.array(zooms, dtype='f8')
    R = new_zooms / zooms
    new_shape = zooms / new_zooms * np.array(data.shape[:3])
    new_shape = tuple(np.round(new_shape).astype('i8'))
    kwargs = {'matrix': R, 'output_shape': new_shape, 'order': order,
              'mode': mode, 'cval': cval}
    if data.ndim == 3:
        data2 = affine_transform(input=data, **kwargs)
    if data.ndim == 4:
        data2 = np.zeros(new_shape+(data.shape[-1],), data.dtype)
        for i in range(data.shape[-1]):
            affine_transform(input=data[..., i], output=data2[..., i],
                                **kwargs)
    Rx = np.eye(4)
    Rx[:3, :3] = np.diag(R)
    affine2 = np.dot(affine, Rx)
    return data2, affine2

"""
    Make the volumes isotropic.
    order=0 used for labels
    order=1 used for images
"""
def isotropic(in_fname,out_fname,order=0):
    data,affine,zoom = read_vol(in_fname)
    new_data, new_affine = reslice(data, affine, zoom, (1., 1., 1.), order)
    write_vol(new_data,new_affine,out_fname)

"""
    Rescale the valume with zero padding
"""
def rescale(vol,size):
    orig_size = vol.shape
    assert(orig_size[0] <= size[0] and orig_size[1] <= size[1] and orig_size[2] <= size[2])
    new_vol = np.zeros(size)
    new_vol[:orig_size[0],:orig_size[1],:orig_size[2]] = vol
    return new_vol

def clip_low(vol):
    return np.clip(vol,0,1)

def main():
    args = parse_args()

    if args.mode == "iso_img":
        isotropic(args.in_name,args.out_name,order=0)
    if args.mode == "iso_cls":
        isotropic(args.in_name,args.out_name,order=1)
    if args.mode == "reg":
        affine_registration(args.in_name,args.atlas,args.out_name,args.out_affine)
    if args.mode == "resample_img":
        affine_transform_nifty(args.in_name,args.atlas,args.in_affine,args.out_name,3)
    if args.mode == "resample_cls":
        affine_transform_nifty(args.in_name,args.atlas,args.in_affine,args.out_name,0)
    if args.mode == "sgauss":
        sub_gauss_smooth_img(args.in_name,args.out_name)
    if args.mode == "ahe":
        histogram_equalization(args.in_name,args.out_name)
    if args.mode == "center":
        standardize(args.in_name,args.out_name,apply_std=False)

if __name__ == "__main__":
    main()
