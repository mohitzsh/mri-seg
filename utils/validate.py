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
from torchvision.transforms import Compose

from utils.transforms import ToTensorLabel
from utils.transforms import ToTensorTIF


from PIL import Image
GPU = False
nclasses = 4
IMG_SUFFIX = "_ana_strip.nii.gz"
CLS_SUFFIX = "_segTRI_ana.nii.gz"

perm = (2,1,0)

mrbrains_train_img_dict = {}
mrbrains_train_cls_dict = {}
mrbrains_val_img_dict = {}
mrbrains_val_cls_dict = {}

def save_vol(data,affine,path):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)

def read_vol(fname):
    img = nib.load(fname)
    data = img.get_data()
    affine = img.affine
    return data,affine

def basegrid(shape):
    batch_size = shape[0]
    theta = torch.FloatTensor([1, 0, 0, 0, 1, 0])
    theta = theta.view(2, 3)
    theta = theta.expand(1,2,3).repeat(batch_size,1,1)

    grid = F.affine_grid(Variable(theta),torch.Size(shape))

    if torch.cuda.is_available():
        grid = grid.cuda()
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
        if torch.cuda.is_available():
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
def validate_ibsr(net,
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
    tvol_img_dict = {}
    tvol_img_affine_dict = {}
    tvol_cls_dict = {}
    tvol_cls_affine_dict = {}
    vvol_img_dict = {}
    vvol_img_affine_dict = {}
    vvol_cls_dict = {}
    vvol_cls_affine_dict = {}

    for tvolname in train_vols:
        tvol_img = nib.load(os.path.join(data_dir,"img",tvolname+img_suffix))
        tvol_cls = nib.load(os.path.join(data_dir,"cls",tvolname+cls_suffix))

        tvol_img_affine = tvol_img.affine
        tvol_cls_affine = tvol_cls.affine

        tvol_img = tvol_img.get_data()
        tvol_cls = tvol_cls.get_data()

        tvol_img = torch.from_numpy(np.transpose(tvol_img,perm)).float()
        tvol_cls = torch.from_numpy(np.transpose(tvol_cls,perm)).float()

        tvol_img_dict[tvolname] = tvol_img
        tvol_cls_dict[tvolname] = tvol_cls
        tvol_img_affine_dict[tvolname] = tvol_img_affine
        tvol_cls_affine_dict[tvolname] = tvol_cls_affine

    for vvolname in val_vols:
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

        vvol_img_dict[vvolname] = vvol_img
        vvol_cls_dict[vvolname] = vvol_cls
        vvol_img_affine_dict[vvolname] = vvol_img_affine
        vvol_cls_affine_dict[vvolname] = vvol_cls_affine



    for tvolname in train_vols:
        tvol_img = tvol_img_dict[tvolname]
        tvol_cls = tvol_cls_dict[tvolname]

        if predictions is None:
            predictions = torch.zeros((len(val_vols),nclasses,) + tvol_img.size()).byte()
            gt = torch.zeros((len(val_vols),) + tvol_img.size()).byte()
        if GPU:
            tvol_img = tvol_img.cuda()
            tvol_cls = tvol_cls.cuda()
        for vvol_idx,vvolname in enumerate(val_vols):

            vvol_img = vvol_img_dict[vvolname]
            vvol_cls = vvol_cls_dict[vvolname]

            gt[vvol_idx] = vvol_cls

            if GPU:
                vvol_img = vvol_img.cuda()
                vvol_cls = vvol_cls.cuda()

            out_img, out_cls_oh = transform_vols(tvol_img,vvol_img,tvol_cls,vvol_cls,net)
            predictions[vvol_idx] += out_cls_oh
    _,predictions = torch.max(predictions,dim=1)

    return  np.mean(dice_score(gt.numpy(),predictions.numpy(),nclasses),axis=1)


def validate_sim(net,data_dir,train_s_idx,train_e_idx,val_s_idx,val_e_idx,gpu):
    global GPU
    GPU = gpu
    predictions = None
    gt = None
    val_img_arr = []
    val_cls_arr = []
    train_img_arr = []
    train_cls_arr = []
    for v_idx in np.arange(val_s_idx,val_e_idx+1):
         val_img = Image.open(os.path.join(data_dir,"img",str(v_idx)+'.tif'))
         val_cls = Image.open(os.path.join(data_dir,"cls",str(v_idx)+'.png')).convert('P')
         val_img_arr.append(ToTensorTIF()(val_img))
         val_cls_arr.append(ToTensorLabel()(val_cls))

    for t_idx in np.arange(train_s_idx,train_e_idx+1):
        train_img = Image.open(os.path.join(data_dir,"img",str(t_idx)+'.tif'))
        train_cls = Image.open(os.path.join(data_dir,"cls",str(t_idx)+'.png')).convert('P')
        train_img = ToTensorTIF()(train_img)
        train_cls = ToTensorLabel()(train_cls)

        train_img_arr.append(train_img)
        train_cls_arr.append(train_cls)

    for v_idx in np.arange(val_s_idx,val_e_idx+1):
        v_idx = v_idx - val_s_idx
        val_img = val_img_arr[v_idx]
        val_cls = val_cls_arr[v_idx]

        if val_img.dim() == 2:
            val_img = val_img.unsqueeze(0)
        if val_cls.dim() == 2:
            val_cls = val_cls.unsqueeze(0)

        assert(val_img.dim() == 3)
        assert(val_cls.dim() == 3)
        if gpu:
            val_img = val_img.cuda()
            val_Cls = val_cls.cuda()
        if predictions is None:
            predictions = torch.zeros(((val_e_idx - val_s_idx +1),nclasses,) + val_img[0].size()).byte()
            gt = torch.zeros(((val_e_idx - val_s_idx + 1),) + val_img[0].size()).byte()
        gt[v_idx] = val_cls[0]
        # Select 50 Random Traning examples to propagate the labels from
        train_images = np.random.random_integers(train_s_idx,train_e_idx,50)

        for t_idx in np.arange(train_s_idx,train_e_idx+1):
            t_idx = t_idx - train_s_idx
            train_img = train_img_arr[t_idx]
            train_cls = train_cls_arr[t_idx]

            if train_img.dim() == 2:
                train_img = train_img.unsqueeze(0)
            if train_cls.dim() == 2:
                train_cls = train_cls.unsqueeze(0)
            if gpu:
                train_img = train_img.cuda()
                train_cls = train_cls.cuda()

            assert(train_img.dim() == 3)
            assert(train_cls.dim() == 3)

            out_img,out_cls_oh = transform_vols(train_img,val_img,train_cls,val_cls,net)
            predictions[v_idx] += out_cls_oh

    _,predictions = torch.max(predictions,dim=1)
    score = dice_score(gt.numpy(),predictions.numpy(),nclasses)
    score = np.average(score,axis=0)
    return score

"""
    Load the validation and training data once.
    mrbrains_train_img_dict
        "subject"
            "t1":
                "orig":
                "proc"
            "t1_ir":
                "orig":
                "proc":
            "t2"
                "orig":
                "proc"
    mrbrains_train_cls_dict
    mrbrains_val_img_dict
    mrbrains_val_cls_dict
"""
def load_mrbrains_dataset(data_dir,train_subjects,val_subjects):
    global mrbrains_train_img_dict
    global mrbrains_train_cls_dict
    global mrbrains_val_img_dict
    global mrbrains_val_cls_dict

    if len(mrbrains_train_img_dict) ==0:
        for sj in train_subjects:
            # Load all modalities and processing extent for images
            t1_orig = str(sj) + '_0_0.nii' # t1_sgauss
            t1_proc = str(sj) + '_0_1.nii' # t1_complete
            t1_ir_orig = str(sj) + '_1_0.nii' # t1_ir_sgauss
            t1_ir_proc = str(sj) + '_1_1.nii' #t1_ir_complete
            t2_orig = str(sj) + '_2_0.nii' # t2_sgauss
            t2_proc = str(sj) + '_2_1.nii' # t2_complete

            # Add t1 volumes
            mrbrains_train_img_dict[str(sj)] = {}
            mrbrains_train_img_dict[str(sj)]["t1"] = {}
            mrbrains_train_img_dict[str(sj)]["t1_ir"] = {}
            mrbrains_train_img_dict[str(sj)]["t2"] = {}

            tpath = os.path.join(data_dir,"img",t1_orig)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_train_img_dict[str(sj)]["t1"]["orig"] = data

            tpath = os.path.join(data_dir,"img",t1_proc)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_train_img_dict[str(sj)]["t1"]["proc"] = data

            # add t1_ir volumes
            tpath = os.path.join(data_dir,"img",t1_ir_orig)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_train_img_dict[str(sj)]["t1_ir"]["orig"] = data

            tpath = os.path.join(data_dir,"img",t1_ir_proc)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_train_img_dict[str(sj)]["t1_ir"]["proc"] = data

            # add t2 volume
            tpath = os.path.join(data_dir,"img",t2_orig)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_train_img_dict[str(sj)]["t2"]["orig"] = data

            tpath = os.path.join(data_dir,"img",t2_proc)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_train_img_dict[str(sj)]["t2"]["proc"] = data

            # Load the cls file
            tcls = str(sj)+str(sj)+'.nii'
            tclspath = os.path.join(data_dir,"cls",tcls)
            data,_ = read_vol(tclspath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).long()

            mrbrains_train_cls_dict[str(sj)] = data
        for sj in val_subjects:
            # Load all modalities and processing extent for images
            t1_orig = str(sj) + '_0_0.nii' # t1_sgauss
            t1_proc = str(sj) + '_0_1.nii' # t1_complete
            t1_ir_orig = str(sj) + '_1_0.nii' # t1_ir_sgauss
            t1_ir_proc = str(sj) + '_1_1.nii' #t1_ir_complete
            t2_orig = str(sj) + '_2_0.nii' # t2_sgauss
            t2_proc = str(sj) + '_2_1.nii' # t2_complete

            # Add t1 volumes
            mrbrains_val_img_dict[str(sj)] = {}
            mrbrains_val_img_dict[str(sj)]["t1"] = {}
            mrbrains_val_img_dict[str(sj)]["t1_ir"] = {}
            mrbrains_val_img_dict[str(sj)]["t2"] = {}

            tpath = os.path.join(data_dir,"img",t1_orig)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_val_img_dict[str(sj)]["t1"]["orig"] = data

            tpath = os.path.join(data_dir,"img",t1_proc)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_val_img_dict[str(sj)]["t1"]["proc"] = data

            # add t1_ir volumes
            tpath = os.path.join(data_dir,"img",t1_ir_orig)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_val_img_dict[str(sj)]["t1_ir"]["orig"] = data

            tpath = os.path.join(data_dir,"img",t1_ir_proc)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_val_img_dict[str(sj)]["t1_ir"]["proc"] = data

            # add t2 volume
            tpath = os.path.join(data_dir,"img",t2_orig)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_val_img_dict[str(sj)]["t2"]["orig"] = data

            tpath = os.path.join(data_dir,"img",t2_proc)
            data,_ = read_vol(tpath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).float()
            mrbrains_val_img_dict[str(sj)]["t2"]["proc"] = data

            # Load the cls file
            tcls = str(sj)+str(sj)+'.nii'
            tclspath = os.path.join(data_dir,"cls",tcls)
            data,_ = read_vol(tclspath)
            data = np.transpose(data,perm)
            data = torch.from_numpy(data).long()

            mrbrains_val_cls_dict[str(sj)] = data
"""
    Combine two mri volumes to transform the entire vol1 to vol2 in single forward pass.
"""
def combine_vol(vol1,vol2):
    if torch.cuda.is_available():
        if not vol1.is_cuda:
            vol1 = vol1.cuda()
        if not vol2.is_cuda:
            vol2 = vol2.cuda()

    # combine vol1 and vol2 as a minibatch along z direction
    shape = vol1.shape
    combvol = torch.zeros((shape[0],)+(2,)+shape[1:]).float()
    if torch.cuda.is_available():
        combvol = combvol.cuda()
    combvol[:,0,:,:] = vol1
    combvol[:,1,:,:] = vol2

    return combvol
"""
    combine img1 and img2 using combine_vol, forward pass through net, generate displacement vectors,
    apply F.grid_sample on the img1 and cls1
"""
def transform_faster(img1,img2,cls1,net):
    net.eval()

    combimg = combine_vol(img1,img2) # Dx2xHxW

    # Generate transformation field
    basegrid_img = basegrid((combimg.shape[0],1,)+img1.shape[1:])
    basegrid_cls = basegrid((combimg.shape[0],nclasses,)+img1.shape[1:])
    disp = generate_grid(net,combimg)
    grid_cls = disp + basegrid_cls
    grid_img = disp + basegrid_img

    # One hot encoded class labels
    # This only works for 3d vols
    # given a vol with shape DxHxW, cl1_oh will be DxCxHxW
    idx = np.arange(nclasses).reshape(nclasses,1)[:,:,None]
    idx = torch.from_numpy(idx).long()
    if torch.cuda.is_available():
        idx = idx.cuda()
    cl1_oh = (cls1[:,None,:,:] == idx)

    # Transform labels
    cls1oht = F.grid_sample(Variable(cl1_oh.float()),grid_cls)
    img1t = F.grid_sample(Variable(img1[:,None,:,:]),grid_cls)

    return img1t.data,cls1oht.data.long(),disp.data

"""
    Register all validation subjects for a given modality and data_processing
    returns:
        All of these are tensors, V is #validation vols, T is #training volumes

        predictions: VxDxHxW
        gt : VxDxHxW
        scores : 4,
        transformation_field : VxTxDxHxWx2
"""
def register_mrbrains(net,train_subjects,val_subjects,modality,data_proc):
    global mrbrains_train_img_dict
    global mrbrains_train_cls_dict
    global mrbrains_val_img_dict
    global mrbrains_val_cls_dict

    predictions = None
    gt = None
    transformation_field = None
    for vidx,vj in enumerate(val_subjects):
        val_cls = mrbrains_val_cls_dict[str(vj)]
        if predictions is None:
            predictions = torch.zeros((len(val_subjects),)+val_cls.shape)
            gt = torch.zeros((len(val_subjects),)+val_cls.shape)
            transformation_field = torch.zeros((len(val_subjects),len(train_subjects),)+val_cls.shape + (2,))
            if torch.cuda.is_available():
                predictions = predictions.cuda()
                gt = gt.cuda()
                transformation_field = transformation_field.cuda()
        gt[vidx] = val_cls
        for mod_val in modality:
            for proc_val in data_proc:
                val_t1_img = mrbrains_val_img_dict[str(vj)][mod_val][proc_val]
                prediction = torch.zeros((nclasses,)+val_cls.shape).long()
                if torch.cuda.is_available():
                    val_t1_img = val_t1_img.cuda()
                    prediction = prediction.cuda()
                for tidx,tj in enumerate(train_subjects):
                    train_cls = mrbrains_train_cls_dict[str(tj)]
                    for mod_train in modality:
                        for proc_train in data_proc:
                            train_t1_img = mrbrains_train_img_dict[str(tj)][mod_train][proc_train]
                            if torch.cuda.is_available():
                                train_cls = train_cls.cuda()
                                train_t1_img = train_t1_img.cuda()
                            _,val_t1_cls_oht,disp  = transform_faster(train_t1_img,val_t1_img,train_cls,net)
                            transformation_field[vidx,tidx] = disp
                            prediction += val_t1_cls_oht
        _,prediction = torch.max(prediction,dim=0)
        predictions[vidx] = prediction
    if torch.cuda.is_available():
        predictions = predictions.cpu()
        gt = gt.cpu()
    score = torch.from_numpy(dice_score(gt.numpy(),predictions.numpy(),nclasses))
    if torch.cuda.is_available():
        predictions = predictions.cuda()
        predictions = predictions.cuda()
        score = score.cuda()

    score = torch.mean(score,dim=0,keepdim=True)
    return predictions,gt,score[0],transformation_field

def validate_mrbrains(net,data_dir,train_subjects,val_subjects,logger,epoch):
    load_mrbrains_dataset(data_dir,train_subjects,val_subjects)

    # Use scores from all modalities separately for processed data
    predictions_t1, gt, score_t1, field_t1 = register_mrbrains(net,train_subjects,val_subjects,["t1"],["proc"])
    predictions_t1_ir,_,score_t1_ir,field_t1_ir = register_mrbrains(net,train_subjects,val_subjects,["t1_ir"],["proc"])
    predictions_t2,_,score_t2,field_t2 = register_mrbrains(net,train_subjects,val_subjects,["t2"],["proc"])
    # Add these scores to the logger
    logger.add_scalars('Dice Scores t1',{'1' : score_t1[1],'2': score_t1[2], '3': score_t1[3]},epoch)
    logger.add_scalars('Dice Scores t1_ir',{'1' : score_t1_ir[1],'2': score_t1_ir[2], '3': score_t1_ir[3]},epoch)
    logger.add_scalars('Dice Scores t2',{'1' : score_t2[1],'2': score_t2[2], '3': score_t2[3]},epoch)

    # visualize, the distribution of the displacement field

    return score_t1

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
