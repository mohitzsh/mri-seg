import argparse
import nibabel as nib
import numpy as np
from subprocess import call
import sys
import os
from scipy.ndimage import affine_transform
import shutil
from scipy import stats
from skimage import exposure

#### TOOD: medpy.filter.IntensityRangeStandardization  (can try this to transform intensities to the same intensity range)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        help="Directory with img and cls folders")
    parser.add_argument("--output_dir",
                        help="Output Directory. It will create img, cls folders with processed brain volumes")
    parser.add_argument("--norm",action='store_true',
                        help="Normalize input to be between 0 and 1")
    parser.add_argument("--affine",action="store_true",
                        help="Affine register the Image to ICMB-MNI152 nonlinear atlas")
    parser.add_argument("--atlas",
                        help="Path to ICMB-MNI152 non-linear atlas")
    parser.add_argument("--isotropic",action="store_true",
                        help="Make the volumes isotropic")
    parser.add_argument("--threshmin",type=float,default=0.02,
                        help="Set all intensities below this to 0")
    parser.add_argument("--n4",action="store_true",
                        help="Apply N4 Bias field correction")
    parser.add_argument("--img_suffix",default="_ana_strip.nii.gz",
                        help="Image Suffix")
    parser.add_argument("--cls_suffix",default="_segTRI_ana.nii.gz",
                        help="Class Suffix")
    parser.add_argument("--skip_images",action="store_true")
    parser.add_argument("--histeq",action="store_true")
    return parser.parse_args()

"""
    Add suffix to the filename
"""
def add_suffix(filename,suffix,new_ext=".nii.gz", old_ext=".nii.gz"):
    prefix = filename.split(old_ext)[0]
    return prefix + "_" + suffix +new_ext
"""
    Returns vol, affine
"""
def read_vol(in_filepath,squeeze=True):
    img = nib.load(in_filepath)
    img_np = img.get_data()
    if squeeze:
        img_np = np.squeeze(img_np,axis=3)
    img_affine = img.affine
    img_zoom = img.header.get_zooms()[:3]
    return img_np,img_affine,img_zoom

def write_vol(data, affine, out_filepath):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, out_filepath)
    return out_filepath

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
    Apply Histogram Equalization to each z slice
"""
def intensity_normalization_histeq(fname,squeeze):
    out_filename = add_suffix(fname,"histeq")
    img_np,img_affine,_ = read_vol(fname,squeeze=squeeze)
    z_size = img_np.shape[2]
    for slice_idx in range(z_size):
        slice_z = img_np[:,:,slice_idx]
        slice_z[slice_z!=0] = exposure.equalize_adapthist(slice_z[slice_z != 0])
        img_np[:,:,slice_idx] = slice_z
    out_fname = add_suffix(fname,"histeq")
    return write_vol(img_np,img_affine,out_fname)
"""
    Affine transform volumes and labels to MNI space using affine (pre-) registration
"""
def affine_transformation(fname,atlas,affine_path,args,label=False):
    out_filename = add_suffix(fname,"affine")
    if not label:
        affine_path = fname.split(args.img_suffix)[0] + "_affine_transform.txt"
    else:
        affine_path = affine_path.split(args.cls_suffix)[0] + "_affine_transform.txt"
    if not label:
        call(["reg_aladin",
            "-noSym", "-speeeeed", "-ref", atlas ,
            "-flo", fname,
            "-res", out_filename,
            "-aff", affine_path,
            "-pad",str(0)])
    else:
        call(["reg_resample",
            "-ref", atlas,
            "-flo", fname,
            "-res", out_filename,
            "-trans", affine_path,
            "-inter", str(0)])

    return out_filename


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
    Assuming that the input is [0,M] for the entire volume
"""
def normalize(fname,squeeze):
    out_filename = add_suffix(fname,"norm")
    img_np,img_affine,_ = read_vol(fname,squeeze=squeeze)
    if squeeze:
        img_np = np.squeeze(img_np,axis=3)
    img_np = (img_np - np.min(img_np))/(np.max(img_np) - np.min(img_np))

    return write_vol(img_np,img_affine,out_filename)

"""
    Make the volumes isotropic.
    order=0 used for labels
    order=1 used for images
"""
def isotropic(data,affine,zoom,order=0):
    data, affine = reslice(data, affine, zoom, (1., 1., 1.), order)
    return data,affine

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
    # NOTE
    # 1) Reshape the volumes at the end
    # 2) normalize the volumes before applying histogram equalization
    # 3) Take care of the NAN values in the MRI volumes
    args = parse_args()
    assert(os.path.exists(args.input_dir))
    img_files = [f for f in os.listdir(os.path.join(args.input_dir,"img")) if (f.split(".")[-1] == "gz" and f.split(".")[-2] == "nii")]
    cls_files = [f for f in os.listdir(os.path.join(args.input_dir,"cls")) if (f.split(".")[-1] == "gz" and f.split(".")[-2] == "nii")]
    # Preprocess the image files before affine registration to MNI Space
    for im_name in img_files:
        if args.skip_images:
            continue
        # Make appropriate directories
        if not os.path.exists(os.path.join(args.output_dir,"temp","img")):
            os.makedirs(os.path.join(args.output_dir,"temp","img"))
        if not os.path.exists(os.path.join(args.output_dir,"img")):
            os.makedirs(os.path.join(args.output_dir,"img"))

        im_path = os.path.abspath(os.path.join(args.input_dir,"img",im_name))
        img_np, affine,zoom = read_vol(im_path)
        if args.isotropic:
            img_np, affine = isotropic(img_np,affine,zoom,order=1)

        # Write the volume to the temporary folder
        tempfname = write_vol(img_np,affine,os.path.join(args.output_dir,"temp","img",im_name))
        # Call external functions from now on
        if args.affine:
            tempfname = affine_transformation(tempfname,args.atlas,os.path.join(args.output_dir,"temp","img",im_name),args)
        if args.n4:
            tempfname = N4BiasField(tempfname)
        if args.norm:
            tempfname = normalize(tempfname,squeeze=False)
        if args.histeq:
            tempfname = intensity_normalization_histeq(tempfname,squeeze=False)
        # Reshape the volumes to 256x256x256
        img_np, affine,zoom = read_vol(tempfname,squeeze=False)
        # img_np = rescale(img_np,(256,256,256))
        write_vol(img_np,affine,tempfname)
        # Copy the file from temp folder to the "img" folder now
        copy_file(tempfname,os.path.join(args.output_dir,"img",im_name))
        print("{} Processed!".format(im_name))

    # Process the label files before affine registration to MNI Space
    for cls_name in cls_files:
        # Setup the folders
        if not os.path.exists(os.path.join(args.output_dir,"cls")):
            os.makedirs(os.path.join(args.output_dir,"cls"))
        if not os.path.exists(os.path.join(args.output_dir,"temp","cls")):
            os.makedirs(os.path.join(args.output_dir,"temp","cls"))

        cls_path = os.path.abspath(os.path.join(args.input_dir,"cls",cls_name))
        label_np, affine,zoom = read_vol(cls_path)

        if args.isotropic:
            label_np, affine = isotropic(label_np,affine,zoom,order=0)

        label_np = rescale(label_np,(256,256,256))
        # Write the temp file used for command line calls
        tempfname = write_vol(label_np,affine,os.path.join(args.output_dir,"temp","cls",cls_name))
        if args.affine:
            tempfname = affine_transformation(tempfname,args.atlas,os.path.join(args.output_dir,"temp","img",cls_name),args,label=True)

        # Copy the files from the temp folder to the destination folder
        copy_file(tempfname,os.path.join(args.output_dir,"cls",cls_name))
        print("{} Processed!".format(cls_name))
    # Remove the temp directory
    shutil.rmtree(os.path.join(args.output_dir,'temp'))

if __name__ == "__main__":
    main()
