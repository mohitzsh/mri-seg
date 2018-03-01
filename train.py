import argparse
import numpy as np
import os

# Transforms
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from utils.transforms import ToTensorLabel
from utils.transforms import RandomRotation
from utils.transforms import Rotation
from utils.transforms import ToTensorTIF

# Data related imports
from torch.utils.data import DataLoader
from datasets.ibsr1 import IBSRv1
from datasets.ibsr2 import IBSRv2

from parameternet.parameternet import ParaNet
from utils.plot import plot_displacement_distribution
from utils.lr_scheduling import poly_lr_scheduler
# Other Torch Related imports
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.interpolations import grid_sample_labels
from utils.grid import forward_diff

# torchvision transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
# Images and Video related imports
from PIL import Image
import subprocess

homedir = os.path.dirname(os.path.realpath(__file__))

H = 200
W = 200
crop_size = (0,0,200,200)
val_img = "01_134"
lab_palette = [0,0,0,
    255,0,0,
    0,255,0,
    0,0,255,
    255,255,0
]

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
    Visualize the original Image and/or Label
    mode : 0 Only image is transformed
           1 Both image and labels are transformed
"""
def visualize_orig(fname,args,img_transform,outprefix="",mode=0):
    with open(os.path.join(args.datadir,"img",fname+".png"),'rb') as f:
        img = Image.open(f).convert('L')
    img = img_transform(img)
    img = img.unsqueeze(0)
    saveimg(img,"img_"+fname+".png")

    if mode == 1:
        with open(os.path.join(args.datadir,"cls",fname1+".png"),'rb') as f:
            label = Image.open(f).convert('P')
            label.putpalette(lab_palette)
        label.save("lab_"+fname+".png")

"""
    Visualize the identity transformation
    img: 1x1xHxW
"""
def visualize_img(net,img1,img2,outprefix,gpu):
    net.eval()
    img_shape = img1.shape

    if gpu:
        combimg = Variable(torch.cat((img1,img2),1).cuda())
    else:
        combimg = Variable(torch.cat((img1,img2),1))

    disp = net(combimg)
    disp = nn.Sigmoid()(disp)*2 - 1

    orig_size = disp.size()
    disp = disp.resize(orig_size[0],orig_size[2],orig_size[3],2)

    grid_img = basegrid(img_shape,gpu) + disp
    if gpu:
        img1 = Variable(img1.cuda())
    else:
        img1 = Variable(img1)
    # Image Transformation
    img1t = F.grid_sample(img1,grid_img)
    #############
    ## TODO: Make a row of three images (SOURCE,TRANSFORMED,TARGET)
    ###########
    if gpu:
        x1 = img1[0].data.squeeze(0).cpu().numpy()
        x2 = img1t[0].data.squeeze(0).cpu().numpy()
        x3 = img2[0].squeeze(0).cpu().numpy()
    else:
        x1 = img1[0].data.squeeze(0).numpy()
        x2 = img1t[0].data.squeeze(0).numpy()
        x3 = img2[0].squeeze(0).numpy()
    ## SAVE THE RESULTS
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(x1,cmap='inferno')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.imshow(x2,cmap='inferno')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(x3,cmap='inferno')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(os.path.join(os.path.abspath("../brain_img"),outprefix+"_"+val_img+".png"),bbox_inches='tight')
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

"""
    Parse Arguments
"""
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("datadir",
                        help="Link to data directory for 2d slices")
    parser.add_argument("exp_name",
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
    return parser.parse_args()

def main():
    args = parse_arguments()
    ##############################################
    ## TRAINING DATASET: GENERIC TRANSFORMATION ##
    ##############################################
    # img_transform = [CenterCrop(128),ToTensor()]
    # label_transform = [CenterCrop(128),ToTensorLabel()]
    # trainset = IBSRv1(homedir,args.datadir,mode="train",co_transform=Compose([]),
    #                 img_transform=Compose(img_transform),label_transform=Compose(label_transform))

    ###############################################
    # TRAINING DATASET : IDENTITY TRANSFORMATION ##
    ###############################################
    # img_transform = [CenterCrop(128),ToTensor()]
    # label_transform = [CenterCrop(128),ToTensorLabel()]
    # trainset = IBSRv2(homedir,args.datadir,mode="train",co_transform=Compose([]),
    #                 img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    # trainloader = DataLoader(trainset,batch_size =args.batch_size,drop_last=True)

    #################################
    ## TRAINING DATASET : ROTATION ##
    #################################
    # img_transform = [RandomRotation(10),ToTensorTIF()]
    img_transform = [ToTensorTIF()]
    # label_transform = [RandomRotation(10,resample=Image.NEAREST),ToTensorLabel()]
    label_transform = [ToTensorLabel()]

    # img_transform_val_src = [Rotation(0),ToTensorTIF()]
    img_transform_val_src = [ToTensorTIF()]
    # label_transform_val_src = [Rotation(5,resample=Image.NEAREST),ToTensorLabel()]
    label_transform_val_src = [ToTensorLabel()]

    # img_transform_val_target = [Rotation(15),ToTensorTIF()]
    img_transform_val_target = [ToTensorTIF()]
    # label_transform_val_target = [Rotation(-5,resample=Image.NEAREST),ToTensorLabel()]
    label_transform_val_target = [ToTensorLabel()]

    trainset = IBSRv2(homedir,args.datadir,mode="train",co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    trainloader = DataLoader(trainset,batch_size =args.batch_size,drop_last=True)
    ######################
    # VALIDATION DATASET #
    ######################
    # valset = IBSRv2(homedir,args.datadir,mode='val',co_transform=Compose([]),
    #                 img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    #
    # valloader = DataLoader(valset,batch_size=1)

    #####################
    # PARAMETER NETWORK #
    ####################

    net = ParaNet()
    if args.gpu:
        # net = nn.DataParallel(net).cuda()
        net = net.cuda()

    #############
    # OPTIMIZER #
    #############
    opt = optim.Adam(filter(lambda p: p.requires_grad, \
                net.parameters()),lr = args.lr,weight_decay=0.0001)

    ############ GENERATE A BASE GRID USING IDENTITY AFFINE TRANSFORMATION
    theta = torch.FloatTensor([1, 0, 0, 0, 1, 0])
    theta = theta.view(2, 3)
    theta = theta.expand(args.batch_size,2,3)
    if args.gpu:
        theta = Variable(theta.cuda())
    else:
        theta = Variable(theta)
    basegrid_img = F.affine_grid(theta,torch.Size((args.batch_size,1,H,W)))
    basegrid_label = F.affine_grid(theta,torch.Size((args.batch_size,5,H,W)))
    ######################
    ## SETUP VALIDATION ##
    ######################
    with open(os.path.join(args.datadir,"img",val_img+".tif"),'rb') as f:
        img = Image.open(f).crop(crop_size)
    # import pdb; pdb.set_trace()
    val_src_img_tensor = Compose(img_transform_val_src)(img).unsqueeze(0)
    val_tar_img_tensor = Compose(img_transform_val_target)(img).unsqueeze(0)
    if args.gpu:
        val_src_img_tensor = Variable(val_src_img_tensor.cuda())
        val_tar_img_tensor = Variable(val_tar_img_tensor.cuda())
    else:
        val_src_img_tensor = Variable(val_src_img_tensor)
        val_tar_img_tensor = Variable(val_tar_img_tensor)

    #################################
    ## TRAINING PROCESS STARTS NOW ##
    #################################
    for epoch in range(args.max_epoch):
        for batch_id, ((img1,label1,_),(img2,label2,_),ohlabel1) in enumerate(trainloader):
            net.train()
            itr = len(trainloader)*(epoch) + batch_id
            if args.gpu:
                img1, label1, img2, label2,combimg,ohlabel1 = Variable(img1.cuda()),\
                        Variable(label1.cuda()), Variable(img2.cuda()), Variable(label2.cuda()),\
                        Variable(torch.cat((img1,img2),1).cuda()), Variable(ohlabel1.cuda())
            else:
                img1, label1, img2, label2,combimg, ohlabel1 = Variable(img1), Variable(label1),\
                        Variable(img2), Variable(label2), Variable(torch.cat((img1,img2),1)), Variable(ohlabel1)
            # import pdb; pdb.set_trace()
            disp = net(combimg)
            disp = nn.Sigmoid()(disp)*2 - 1 # Displacement is [-1,1]

            orig_size = disp.size()
            disp = disp.resize(orig_size[0],orig_size[2],orig_size[3],2)

            ##########################
            ## IMAGE TRANSFORMATION ##
            ##########################
            grid_img = basegrid_img + disp
            img1t = F.grid_sample(img1,grid_img)
            Lsim = nn.MSELoss(size_average=False)(img1t,img2)

            ###########################
            ### LABEL TRANSFORMATION ##
            ###########################
            Lseg = Variable(torch.Tensor([0]))
            if args.gpu:
                Lseg = Variable(torch.Tensor([0]).cuda())
            if args.lambdaseg != 0:
                grid_label = basegrid_label + disp
                cprob2 = F.grid_sample(ohlabel1.float(),grid_label)
                logcprob2 = nn.LogSoftmax()(cprob2)
                Lseg = nn.NLLLoss()(logcprob2,label2)

            ###################
            ## REGULARIZATON ##
            ###################
            Lreg = Variable(torch.Tensor([0]))
            if args.gpu:
                Lreg = Variable(torch.Tensor([0]).cuda())
            if args.lambdareg != 0:
                disp = disp.view(-1,2,H,W)
                dispgrad = forward_diff(disp)
                zeros = Variable(torch.from_numpy(np.zeros(disp.data.shape)))
                target = torch.cat((zeros,zeros),1).cuda().float() # Target has to be a float
                Lreg = nn.L1Loss(size_average=False)(dispgrad,target)

            ######################
            ## PARAMETER UPDATE ##
            ######################
            Ltotal = Lsim + args.lambdareg*Lreg + args.lambdaseg*Lseg
            opt.zero_grad()
            poly_lr_scheduler(opt, args.lr, itr)
            Ltotal.backward()
            opt.step()

            # VISULAIZE THE TRANSFORMATION
            if args.visualize and (itr % 20 == 0):
                visualize_img(net,val_src_img_tensor.data,val_tar_img_tensor.data,args.exp_name+"_{:05d}".format(itr),args.gpu)
            print("[{}][{}] Ltotal: {:.4} Lsim: {:.4f} Lseg: {:.4f} Lreg: {:.4f} ".\
                format(epoch,itr,Ltotal.data[0],Lsim.data[0],args.lambdaseg*(Lseg.data[0]),args.lambdareg*(Lreg.data[0])))

    make_video('{}.mp4'.format(args.exp_name),args.exp_name,keep_imgs=False)
if __name__ == "__main__":
    main()
