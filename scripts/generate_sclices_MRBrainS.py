import os
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from nilearn import image
import nibabel as nib

perm = (2,1,0)

img_files = [
    "11_T1_shifted.nii",
    "11_T1_IR_shifted.nii",
    "11_T2_FLAIR_shifted.nii",
    "111_T1.nii",
    "111_T1_IR.nii",
    "111_T2_FLAIR.nii",
    "22_T1_shifted.nii",
    "22_T1_IR_shifted.nii",
    "22_T2_FLAIR_shifted.nii",
    "222_T1.nii",
    "222_T1_IR.nii",
    "222_T2_FLAIR.nii",
    "33_T1_shifted.nii",
    "33_T1_IR_shifted.nii",
    "33_T2_FLAIR_shifted.nii",
    "333_T1.nii",
    "333_T1_IR.nii",
    "333_T2_FLAIR.nii",
    "44_T1_shifted.nii",
    "44_T1_IR_shifted.nii",
    "44_T2_FLAIR_shifted.nii",
    "444_T1.nii",
    "444_T1_IR.nii",
    "444_T2_FLAIR.nii"

]

cls_files = [
    "11.nii",
    "22.nii",
    "33.nii",
    "44.nii",
]

modality_map = {
    "T1" : 0,
    "T1_IR": 1,
    "T2_FLAIR": 2
}

def read_vol(fname):
    data = nib.load(fname)
    data = data.get_data()
    return np.transpose(data,perm)

def save_data_slice(data,fname):
    img = Image.fromarray(data)
    img.save(fname)

def save_visualization_slice(data,fname,cmap=None):
    if cmap is None:
        plt.imsave(fname,data)
    else:
        plt.imsave(fname,data,cmap=cmap)

def get_subject_name(fname):
    end_fname = fname.split("/")[-1]
    return end_fname.split("_")[0][0]

"""
    stage 0: stands for original image with gaussian smoothed image subtracted out
    stage 1: Full data preprocessing applied
"""
def get_pp_stage(fname):
    end_fname = fname.split("/")[-1]
    if len(end_fname.split("_")[0]) ==2:
        return "1"
    else:
        return "0"
"""
    0 : T1
    1 : T1_IR
    2 : T2_FLAIR
"""
def get_modality(fname):
    end_fname = fname.split("/")[-1].split(".")[0]
    if end_fname.split("_")[1] == "T1" and len(end_fname.split("_")) == 2:
        return "0"
    elif end_fname.split("_")[1] == "T1" and end_fname.split("_")[2] != "IR" :
        return "0"
    elif end_fname.split("_")[1] == "T1" and end_fname.split("_")[2] == "IR":
        return "1"
    else:
        return "2"


def process_img_vol(fname,out_data_dir,out_vis_dir,train_list_fname):
    data = read_vol(fname)
    nz,_,_ = data.shape

    f = open(train_list_fname,'a+')
    for idx in range(nz):
        subject_name = get_subject_name(fname)
        modality = get_modality(fname)
        pp = get_pp_stage(fname)
        slice_name = subject_name + '_' + modality + '_' + pp + '_' + str(idx)
        f.write("{}\n".format(slice_name))
        slice_data_path = os.path.join(out_data_dir,slice_name) + '.tif'
        slice_vis_path = os.path.join(out_vis_dir,slice_name) + '.png'
        save_data_slice(data[idx],slice_data_path)
        save_visualization_slice(data[idx],slice_vis_path,cmap='gray')
        print("img slice {} saved!".format(slice_name))
    f.close()

def process_cls_vol(fname,out_data_dir,out_vis_dir):
    data = read_vol(fname)
    nz,_,_ = data.shape
    for idx in range(nz):
        subject_name = get_subject_name(fname)
        slice_name = subject_name + '_' + str(idx)
        slice_data_path = os.path.join(out_data_dir,slice_name) + '.png'
        slice_vis_path = os.path.join(out_vis_dir,slice_name) + '.png'
        save_data_slice(data[idx],slice_data_path)
        save_visualization_slice(data[idx],slice_vis_path)
        print("cls slice {} saved!".format(slice_name))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",)
    parser.add_argument("--out_dir")
    parser.add_argument("--train_list")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(os.path.join(args.out_dir,"data","img")):
        os.makedirs(os.path.join(args.out_dir,"data","img"))
    if not os.path.exists(os.path.join(args.out_dir,"data","cls")):
        os.makedirs(os.path.join(args.out_dir,"data","cls"))
    if not os.path.exists(os.path.join(args.out_dir,"vis","img")):
        os.makedirs(os.path.join(args.out_dir,"vis","img"))
    if not os.path.exists(os.path.join(args.out_dir,"vis","cls")):
        os.makedirs(os.path.join(args.out_dir,"vis","cls"))

    out_data_dir_img = os.path.join(args.out_dir,"data","img")
    out_data_dir_cls = os.path.join(args.out_dir,"data","cls")
    out_vis_dir_img = os.path.join(args.out_dir,"vis","img")
    out_vis_dir_cls = os.path.join(args.out_dir,"vis","cls")
    "Save Image files"
    for name in img_files:
        fname = os.path.join(args.in_dir,"img",name)
        process_img_vol(fname,out_data_dir_img,out_vis_dir_img,args.train_list)
    for name in cls_files:
        fname = os.path.join(args.in_dir,"cls",name)
        process_cls_vol(fname,out_data_dir_cls,out_vis_dir_cls)

if __name__ == "__main__":
    main()
