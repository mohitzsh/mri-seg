from torchvision.transforms import Compose
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.autograd import Variable
from utils.transforms import OneHotEncode
import argparse
from utils.scores import dice_score
import os
import nibabel as nib

GPU = False
nclasses = 4
IMG_SUFFIX = "_ana_strip.nii.gz"
CLS_SUFFIX = "_segTRI_ana.nii.gz"

def save_vol(data,affine,path):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)

def basegrid(shape):
    theta = torch.FloatTensor([1, 0, 0, 0, 1, 0])
    theta = theta.view(2, 3)
    theta = theta.expand(1,2,3)
    grid = F.affine_grid(theta,shape)
    if GPU:
        return grid.cuda()
    return grid

def generate_grid(net,combslice):
    net.eval()

    disp = net(Variable(combslice,volatile=True))
    orig_size = disp.size()
    disp = disp.resize(orig_size[0],orig_size[2],orig_size[3],2)

    return disp

"""
    img1 : DxHxW tensor
    img2 : DxHxW tensor
    cls1 : DxHxW tensor
    cls2 : DxHxW tensor

    Return:
    out_img: DxHxW float tensor
    out_cls: DxHxWx4 float tensor
"""
def transform_vols(img1,img2,cls1,cls2,net):

    D,H,W = (img1.shape[0],img1.shape[1],img1.shape[2])
    out_img = torch.zeros(img1.shape).float()
    out_cls = torch.zeros((nclasses,) + cls1.shape).float()
    basegrid_img = basegrid(torch.Size((1,1,H,W)))
    basegrid_cls = basegrid(torch.Size((1,nclasses,H,W)))
    for idx in range(D):
        imgslice1 = img1[idx]
        imgslice2  = img2[idx]
        clsslice1 = cls1[idx]
        clsslice2 = cls2[idx]

        clsslice1_oh = OneHotEncode()(clsslice1.unsqueeze(0)).float()
        clsslice2_oh = OneHotEncode()(clsslice2.unsqueeze(0)).float()
        if GPU:
            clsslice1_oh = clsslice1_oh.cuda()
            clsslice2_oh = clsslice2_oh.cuda()
        combslice = torch.cat((imgslice1.unsqueeze(0),imgslice2.unsqueeze(0)),dim=0)

        disp = generate_grid(net,combslice.unsqueeze(0))

        grid_cls = basegrid_cls + disp
        grid_img = basegrid_img + disp

        imgslice1_t = F.grid_sample(imgslice1.unsqueeze(0).unsqueeze(0),grid_img)[0,0] # HxW
        clsslice1_oh_t = F.grid_sample(clsslice1_oh.unsqueeze(0),grid_cls)[0] # 4xHxW
        _,clsslice1_oh_t = torch.max(clsslice1_oh_t,dim=0)
        clsslice1_oh_t = OneHotEncode()(clsslice1_oh_t.data.unsqueeze(0))
        out_img[idx] = imgslice1_t.data
        out_cls[:,idx] = clsslice1_oh_t

    return out_img,out_cls.byte()

"""
    val_volume is the list of volumes to validate
"""
def validate(net,
            data_dir,
            train_vols,
            val_vols,
            img_suffix,
            cls_suffix,
            perm,
            gpu=True,
            ):
    global GPU
    GPU = gpu

    predictions = None
    gt = None
    for tvolname in train_vols:
        tvol_img = nib.load(os.path.join(data_dir,"img",tvolname+img_suffix))
        tvol_cls = nib.load(os.path.join(data_dir,"cls",tvolname+cls_suffix))

        tvol_img_affine = tvol_img.affine
        tvol_img = tvol_img.get_data()

        tvol_cls_affine = tvol_cls.affine
        tvol_cls = tvol_cls.get_data()

        tvol_img = torch.from_numpy(np.transpose(tvol_img,perm)).float()
        tvol_cls = torch.from_numpy(np.transpose(tvol_cls,perm)).float()

        if predictions is None:
            predictions = torch.zeros((len(val_vols),nclasses,) + tvol_img.size()).byte()
            gt = torch.zeros((len(val_vols),) + tvol_img.size()).byte()
        if GPU:
            tvol_img = tvol_img.cuda()
            tvol_cls = tvol_cls.cuda()
        for vvol_idx,vvolname in enumerate(val_vols):

            vvol_img = nib.load(os.path.join(data_dir,"img",vvolname+img_suffix))
            vvol_cls = nib.load(os.path.join(data_dir,"cls",vvolname+cls_suffix))

            vvol_img_affine = vvol_img.affine
            vvol_img = vvol_img.get_data()

            vvol_cls_affine = vvol_cls.affine
            vvol_cls = vvol_cls.get_data()

            vvol_img = np.transpose(vvol_img,perm)
            vvol_cls = np.transpose(vvol_cls,perm)

            vvol_img = torch.from_numpy(vvol_img).float()
            vvol_cls = torch.from_numpy(vvol_cls).float()

            gt[vvol_idx] = vvol_cls

            if GPU:
                vvol_img = vvol_img.cuda()
                vvol_cls = vvol_cls.cuda()

            out_img, out_cls_oh = transform_vols(tvol_img,vvol_img,tvol_cls,vvol_cls,net)
            predictions[vvol_idx] += out_cls_oh
    _,predictions = torch.max(predictions,dim=1)

    score = dice_score(gt.numpy(),predictions.numpy(),nclasses)
    print(score)




def parse_arguments():
    # TODO: Add argument to use a trained model
    parser = argparse.ArgumentParser()

    parser.add_argument("--no_train",action="store_true",
                        help="Just run validation with default identity transformation. Good for debug mode")
    parser.add_argument("--data_dir",
                        help="Data directory with img, cls folders")
    parser.add_argument("--train_vols",nargs='+',
                        help="Training Volume Names")
    parser.add_argument("--val_vols",nargs='+',
                        help="Validation volumes")
    parser.add_argument("--img_suffix",default="_ana_strip.nii.gz",
                        help="Suffix for the images")
    parser.add_argument("--cls_suffix",default="_segTRI_ana.nii.gz",
                        help="Suffix for the classes")
    parser.add_argument("--gpu",action="store_true",
                        help="If GPU should be use")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_arguments()
    if args.no_train:
        validate(None,args.data_dir,args.train_vols,args.val_vols,
                    args.img_suffix,args.cls_suffix,(2,1,0),args.gpu,
                )
