import argparse
import numpy as np
import os
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from utils.transforms import ToTensorLabel
from utils.transforms import RandomRotation
from utils.transforms import Rotation
from utils.transforms import ToTensorTIF
from torch.utils.data import DataLoader
from datasets.ibsr1 import IBSRv1
from datasets.ibsr2 import IBSRv2
from datasets.sim import Sim
from parameternet.parameternet import ParaNet
from parameternet.unet import UNet
from parameternet.unet_small import UNetSmall
from utils.plot import plot_displacement_distribution
from utils.lr_scheduling import poly_lr_scheduler
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.interpolations import grid_sample_labels
from utils.grid import forward_diff
from utils.losses import cc
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
from utils.validate import validate
from utils.validate import validate_sim
import tensorboardX as tbx

homedir = os.path.dirname(os.path.realpath(__file__))

H = 218
W = 182
nclasses = 4
train_vols = ['IBSR_01','IBSR_02','IBSR_03','IBSR_04','IBSR_05',
                'IBSR_06','IBSR_07','IBSR_08','IBSR_09','IBSR_10',
            'IBSR_11','IBSR_12','IBSR_13','IBSR_14']

val_vols = ['IBSR_15','IBSR_16','IBSR_17','IBSR_18',]

val_img_src = "01_083"
val_img_target = "02_083"
lab_palette = [0,0,0,
    255,0,0,
    0,255,0,
    0,0,255,
    255,255,0
]
perm = (2,1,0)
"""
    img: Nx1xHxW cpu tensor
"""
def saveimg(img,name):
    i = ToPILImage()(img[0]).convert('L')
    i.save(name)

"""
    label: NxHxW cpu tensor
"""
def savelabel(label,name):
    l = ToPILImage()(label[0].unsqueeze(0).byte()).convert('P')
    l.putpalette(lab_palette)
    l.save(name)

"""
    Make a video from the transformations
"""
def make_video(outname,exp_name,keep_imgs=False,fps=100,is_color=True,format='XVID'):
    dir_path = os.path.abspath("../brain_img")
    fmt = dir_path + '/' + exp_name + '_'
    subprocess.call(['ffmpeg','-s','200x200','-r','{:d}/1'.format(fps),'-i','{}%05d_{}.png'.format(fmt,val_img),'-vcodec','mpeg4','-y','{}'.format(outname)])
    if not keep_imgs:
        # Delete all the .png files
        subprocess.call('rm {}/*.png'.format(dir_path),shell=True)
"""
    Make a with output size as
    size: NxCxHxW
"""
def basegrid(size,gpu):
    theta = torch.FloatTensor([1, 0, 0, 0, 1, 0])
    theta = theta.view(2, 3)
    theta = theta.expand(size[0],2,3)
    if gpu:
        theta = Variable(theta.cuda())
    else:
        theta = Variable(theta)
    return F.affine_grid(theta,torch.Size(size))

def snapshot(model,prefix,snapshot_dir):
    snapshot = {
        'model' : model.state_dict()
    }
    torch.save(snapshot,os.path.join(snapshot_dir,"{}.pth.tar").format(prefix))

def free_vars(*args):
    for var in list(args):
        if var is not None:
            del var

def view_img_transformation(img1,img1t,img2,itr,gpu):
    # import pdb; pdb.set_trace()
    for idx in range(img1.shape[0]):
        img1_idx = img1[idx]
        img1t_idx = img1t[idx]
        img2_idx = img2[idx]

        if gpu:
            x1 = img1_idx.data.squeeze(0).cpu().numpy()
            x2 = img1t_idx.data.squeeze(0).cpu().numpy()
            x3 = img2_idx.data.squeeze(0).cpu().numpy()
        else:
            x1 = img1_idx.data.squeeze(0).numpy()
            x2 = img1t_idx.data.squeeze(0).numpy()
            x3 = img2_idx.data.squeeze(0).numpy()
        h,w = x1.shape
        # Cretae a single numpy array with three images in a row
        x_final = np.zeros((h,3*w))
        x_final[:,:w] = x1
        x_final[:,w:2*w] = x2
        x_final[:,2*w:] = x3
        plt.imsave(os.path.join(os.path.abspath("../brain_img"),str(idx)+'_'+str(itr)+".png"), x_final, cmap='gray')

"""
    Parse Arguments
"""
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir_2d",
                        help="Path to data directory for 2d slices")
    parser.add_argument("--data_dir_3d",
                        help="Path to data dir for 3d volumes")
    parser.add_argument("--exp_name",
                        help="name of the experiments for prefixing saved models, images and video")
    parser.add_argument("--max_epoch",default=1,type=int,
                        help="Max epochs for training")
    parser.add_argument("--gpu",action='store_true',
                        help="GPU training")
    parser.add_argument("--lr",default=0.0001,type=float,
                        help="Lr for training")
    parser.add_argument("--lambdaseg",default=0,type=float,
                        help="Weight for segmentation loss")
    parser.add_argument("--lambdareg",default=0,type=float,
                        help="weight for regularization")
    parser.add_argument("--batch_size",default=4,type=int,
                        help="Batch size for training")
    parser.add_argument("--visualize",action='store_true',
                        help="Flag to save transformed images after each iteration")
    parser.add_argument("--train_vols",nargs='+',default=train_vols,
                        help="Training Volume Names")
    parser.add_argument("--val_vols",nargs='+',default=val_vols,
                        help="Validation volumes")
    parser.add_argument("--img_suffix",default="_ana_strip.nii.gz",
                        help="Suffix for the images")
    parser.add_argument("--cls_suffix",default="_segTRI_ana.nii.gz",
                        help="Suffix for the classes")
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--error_iter",default=500,type=int)
    parser.add_argument("--simulate",action="store_true")
    parser.add_argument("--train_s_idx",type=int,default=0)
    parser.add_argument("--train_e_idx",type=int,default=1000)
    parser.add_argument("--val_s_idx",type=int,default=5991)
    parser.add_argument("--val_e_idx",type=int,default=5999)
    parser.add_argument("--similarity",choices=('l2','cc'),default='cc')
    parser.add_argument("--log_dir",default="runs")
    parser.add_argument("--l2_weight",type=float,default=10e-4)
    parser.add_argument("--nker",type=int,default=8)
    return parser.parse_args()

def main():
    args = parse_arguments()
    args_str = str(vars(args))

    logger = tbx.SummaryWriter(log_dir = args.log_dir,comment=args.exp_name)
    logger.add_text('training details',args_str,0)
    #############################################
    # TRAINING DATASET: GENERIC TRANSFORMATION ##
    #############################################
    img_transform = [ToTensorTIF()]
    label_transform = [ToTensorLabel()]
    if args.simulate:
        trainset = Sim(args.data_dir_2d,args.train_s_idx,args.train_e_idx,co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform),drop_last=True)
    else:
        trainset = IBSRv1(homedir,args.data_dir_2d,co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    trainloader = DataLoader(trainset,batch_size =args.batch_size,shuffle=True,drop_last=True)

    #####################
    # PARAMETER NETWORK #
    ####################

    net = UNetSmall(args.nker)
    if args.gpu:
        net = nn.DataParallel(net).cuda()

    #############
    # OPTIMIZER #
    #############
    opt = optim.Adam(filter(lambda p: p.requires_grad, \
                net.parameters()),lr = args.lr,weight_decay=args.l2_weight)

    ##################################################################
    ### GENERATE A BASE GRID USING IDENTITY AFFINE TRANSFORMATION ####
    ##################################################################
    theta = torch.FloatTensor([1, 0, 0, 0, 1, 0])
    theta = theta.view(2, 3)
    theta = theta.expand(args.batch_size,2,3)
    if args.gpu:
        theta = Variable(theta.cuda())
    else:
        theta = Variable(theta)
    basegrid_img = F.affine_grid(theta,torch.Size((args.batch_size,1,H,W)))
    basegrid_label = F.affine_grid(theta,torch.Size((args.batch_size,nclasses,H,W)))

    #################################
    ## TRAINING PROCESS STARTS NOW ##
    #################################
    for epoch in range(args.max_epoch):
        for batch_id, ((img1,label1,ohlabel1,fname1),(img2,label2,ohlabel2,fname2)) in enumerate(trainloader):
            if img1 is None or label1 is None or img2 is None or label2 is None or ohlabel1 is None:
                continue
            net.train()
            itr = len(trainloader)*(epoch) + batch_id
            if args.gpu:
                img1, label1, img2, label2,combimg,ohlabel1 = Variable(img1.cuda()),\
                        Variable(label1.cuda()), Variable(img2.cuda()), Variable(label2.cuda()),\
                        Variable(torch.cat((img1,img2),1).cuda()), Variable(ohlabel1.cuda())
            else:
                img1, label1, img2, label2,combimg, ohlabel1 = Variable(img1), Variable(label1),\
                        Variable(img2), Variable(label2), Variable(torch.cat((img1,img2),1)), Variable(ohlabel1)

            disp = net(combimg)
            orig_size = disp.size()
            n,h,w = (disp.shape[0],disp.shape[2],disp.shape[3])
            disp = disp.resize(n,h,w,2)
            ##########################
            ## IMAGE TRANSFORMATION ##
            ##########################
            grid_img = basegrid_img + disp

            img1t = F.grid_sample(img1,grid_img)
            if args.similarity == 'cc':
                Lsim = cc(img1t.data,img2.data)
            elif args.similarity == 'l2':
                Lsim = nn.MSELoss()(img1t,img2)

            if args.plot:
                view_img_transformation(img1,img1t,img2,itr,args.gpu)
            ###########################
            ### LABEL TRANSFORMATION ##
            ###########################
            Lseg = Variable(torch.Tensor([0]),requires_grad=True)
            if args.gpu:
                Lseg = Variable(torch.Tensor([0]).cuda(),requires_grad=True)
            if args.lambdaseg != 0:
                grid_label = basegrid_label + disp
                cprob2 = F.grid_sample(ohlabel1.float(),grid_label)
                logcprob2 = nn.LogSoftmax()(cprob2)
                Lseg = nn.NLLLoss()(logcprob2,label2)

            ###################
            ## REGULARIZATON ##
            ###################
            Lreg = Variable(torch.Tensor([0]),requires_grad=True)
            target = torch.zeros(1)
            if args.gpu:
                Lreg = Variable(torch.Tensor([0]).cuda(),requires_grad=True)
            if args.lambdareg != 0:
                disp = disp.view(-1,2,h,w)
                dx = torch.abs(disp[:,:,1:,:] -disp[:,:,:-1,:])
                dy = torch.abs(disp[:,:,:,1:] - disp[:,:,:,:-1])
                # Implement L1 penalty for now
                dx_mean = torch.mean(dx)
                dy_mean = torch.mean(dy)
                target = torch.zeros(1)
                if args.gpu:
                    target = target.cuda()
                Lreg = nn.L1Loss()((dx_mean+dy_mean)/2,Variable(target))

            ######################
            ## PARAMETER UPDATE ##
            ######################
            Ltotal = Lsim + args.lambdareg*Lreg + args.lambdaseg*Lseg
            opt.zero_grad()
            opt,lr=poly_lr_scheduler(opt, args.lr, itr,max_iter=len(trainloader)*args.max_epoch)
            Ltotal.backward()
            opt.step()
            logger.add_scalars("Loss",{"Total":Ltotal.data[0],"Similarity":Lsim.data[0],"Regularization":args.lambdareg*Lreg.data[0]},itr)
            logger.add_scalar("lr",lr,itr)
            # Delete the variables
            free_vars(img1,img2,label1,label2,combimg,ohlabel1,ohlabel2,disp,target,img1t)
            # VISULAIZE THE TRANSFORMATION


            if args.visualize and (itr % 20 == 0):
                visualize_img(net,val_img_src_tensor.data,val_img_target_tensor.data,args.exp_name+"_{:05d}".format(itr),args.gpu)

            if itr % args.error_iter ==0:
                print("[{}][{}] Ltotal: {:.6} Lsim: {:.6f} Lseg: {:.6f} Lreg: {:.6f} ".\
                    format(epoch,itr,Ltotal.data[0],Lsim.data[0],args.lambdaseg*(Lseg.data[0]),args.lambdareg*(Lreg.data[0])))

        if args.simulate:
            score = validate_sim(net,args.data_dir_2d,args.train_s_idx,args.train_e_idx,args.val_s_idx,args.val_e_idx,args.gpu)
        else:
            validate(net,args.data_dir_3d,args.train_vols,args.val_vols,args.img_suffix,args.cls_suffix,perm,args.gpu)
        logger.add_scalars('Dice Scores',{'Class 0': score[0],'Class 1' : score[1],'Class 2': score[2], 'Class 3': score[3]},epoch)
    if args.visualize:
        make_video('{}.mp4'.format(args.exp_name),args.exp_name,keep_imgs=False)
    logger.close()
if __name__ == "__main__":
    main()
