from torchvision.transforms import Compose
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from torch.autograd import Variable
from transforms import OneHotEncode
import argparse
from scores import dice_score
import os
H = 218
W = 182
D = 182

nclasses = 4
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

    if net is not None:
        net.eval()
    # Generate the base grid
    theta = torch.FloatTensor([1, 0, 0, 0, 1, 0])
    theta = theta.view(2, 3)
    theta = theta.expand(1,2,3)
    if gpu:
        theta = Variable(theta.cuda())
    else:
        theta = Variable(theta)
    basegrid_label = F.affine_grid(theta,torch.Size((1,nclasses,H,W)))

    tvol_size = (D,H,W)
    valcount = len(val_vols)
    prediction_vol = np.repeat(np.zeros(tvol_size,dtype=np.uint8)[:,:,:,None],nclasses,axis=3)
    predictions = np.repeat(prediction_vol[None,:,:,:,:],valcount,axis=0)
    ground_truth = np.zeros((valcount,D,H,W),dtype=np.uint8)
    # This dictionary contains predictions for each volume in val_volumes
    for tvolname in train_vols:
        tvol_img = nib.load(os.path.join(data_dir,"img",tvolname+img_suffix)).get_data()
        tvol_label = nib.load(os.path.join(data_dir,"cls",tvolname+cls_suffix)).get_data()

        if tvol_img.ndim == 4:
            tvol_img = np.squeeze(tvol_img,axis=3)
        if tvol_label.ndim == 4:
            tvol_label = np.squeeze(tvol_label,axis=3)

        tvol_img = np.transpose(tvol_img,perm)
        tvol_label = np.transpose(tvol_label,perm)
 # VxHxWxDxnClass


        for vvol_idx,vvolname in enumerate(val_vols):

            vvol_img = nib.load(os.path.join(data_dir,"img",vvolname+img_suffix)).get_data()
            vvol_cls = nib.load(os.path.join(data_dir,"cls",vvolname+cls_suffix)).get_data()
            if tvol_img.ndim == 4:
                vvol_img = np.squeeze(vvol_img,axis=3)
            if tvol_label.ndim == 4:
                vvol_label = np.squeeze(vvol_label,axis=3)
            vvol_img = np.transpose(vvol_img,perm)
            vvol_cls = np.transpose(vvol_cls,perm)

            ground_truth[vvol_idx,...] = vvol_cls
            # get predictions for currect val volume
            for slice_idx in range(vvol_img.shape[0]):
                train_img_slice = tvol_img[slice_idx]
                val_img_slice = vvol_img[slice_idx]
                train_label_slice = tvol_label[slice_idx]

                # Fake dimension added for channel
                combslice = np.concatenate((train_img_slice[None,:,:],val_img_slice[None,:,:]),axis=0)

                # Get tensors
                combslice = torch.from_numpy(combslice[None,:,:,:]) # Add a fake dimension for batch
                train_label_slice = torch.from_numpy(train_label_slice[None,:,:]) # Add fake dimension for channel
                train_label_slice_ohencode = OneHotEncode()(train_label_slice) # CxHxW
                train_label_slice_ohencode = train_label_slice_ohencode.unsqueeze(0) # 1xCxHxW
                if gpu:
                    combslice = Variable(combslice.cuda())
                    train_label_slice_ohencode = Variable(train_label_slice_ohencode.cuda())
                grid_label = basegrid_label
                if net is not None:
                    disp = net(combslice)
                    disp = nn.Sigmoid()(disp)*2 - 1

                    orig_size = disp.size()
                    disp = disp.resize(orig_size[0],orig_size[2],orig_size[3],2)

                    grid_label += disp

                # Check if this works. Otherwise, use a cpu version of the sampling function for labels
                soft_label_pred = F.grid_sample(train_label_slice_ohencode.float(),grid_label)[0] # Get rid of the batch dimension
                _,hard_label_pred = torch.max(soft_label_pred,0)

                hard_label_pred_ohencode = OneHotEncode()(hard_label_pred.data.cpu())
                predictions[vvol_idx,slice_idx] += torch.from_numpy(np.transpose(hard_label_pred_ohencode.cpu().numpy(),(1,2,0)))
    predictions = torch.from_numpy(predictions).byte()
    _,final_predictions = torch.max(predictions,4)


    # Compute the per_class dice score between final_predictions,ground_truth
    if gpu:
        final_predictions = final_predictions.cpu()
    scores = dice_score(ground_truth,final_predictions.numpy(),nclasses)

    # Print the scores
    for idx,valname in enumerate(val_vols):
        print("{}\n".format(valname))
        for c in range(nclasses):
            print("Class {},\tDice Score {}\n".format(c,scores[idx,c]*100))

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
