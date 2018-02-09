import argparse
import numpy as np
import os

# Transforms
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from utils.transforms import ToTensorLabel

# Data Related imports
from torch.utils.data import DataLoader
from datasets.ibsr1 import IBSRv1
from datasets.ibsr2 import IBSRv2

from parameternet.parameternet import ParaNet
from utils.plot import plot_displacement_distribution

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

homedir = os.path.dirname(os.path.realpath(__file__))

H = 128
W = 128

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
    parser.add_argument("--max_epoch",default=1,type=int,
                        help="Max epochs for training")
    parser.add_argument("--gpu",action='store_true',
                        help="GPU training")
    parser.add_argument("--lr",default=0.0001,type=float,
                        help="Lr for training")
    parser.add_argument("--lambdaseg",default=0.01,type=float,
                        help="Weight for segmentation loss")
    parser.add_argument("--lambdareg",default=0.01,type=float,
                        help="weight for regularization")
    parser.add_argument("--batch_size",default=4,type=int,
                        help="Batch size for training")

    return parser.parse_args()

def main():
    args = parse_arguments()

    img_transform = [CenterCrop(128),ToTensor()]
    label_transform = [CenterCrop(128),ToTensorLabel()]
    # trainset = IBSRv1(homedir,args.datadir,mode="train",co_transform=Compose([]),
    #                 img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    ####################
    # TRAINING DATASET #
    ###################
    trainset = IBSRv2(homedir,args.datadir,mode="train",co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    trainloader = DataLoader(trainset,batch_size =args.batch_size)
    ######################
    # VALIDATION DATASET #
    ######################
    valset = IBSRv2(homedir,args.datadir,mode='val',co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform))

    valloader = DataLoader(valset,batch_size=1)

    #####################
    # PARAMETER NETWORK #
    ####################

    net = ParaNet()
    if args.gpu:
        net = nn.DataParallel(net).cuda()

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


    for epoch in range(args.max_epoch):
        net.train()

        for batch_id, ((img1,label1,_),(img2,label2,_),ohlabel1) in enumerate(trainloader):
            itr = len(trainloader)*(epoch) + batch_id
            if args.gpu:
                img1, label1, img2, label2,combimg,ohlabel1 = Variable(img1.cuda()),\
                        Variable(label1.cuda()), Variable(img2.cuda()), Variable(label2.cuda()),\
                        Variable(torch.cat((img1,img2),1).cuda()), Variable(ohlabel1.cuda())
            else:
                img1, label1, img2, label2,combimg, ohlabel1 = Variable(img1), Variable(label1),\
                        Variable(img2), Variable(label2), Variable(torch.cat((img1,img2),1)), Variable(ohlabel1)

            disp = net(combimg)
            disp = nn.Sigmoid()(disp)*2 - 1 # Displacement is [-1,1]
            # # Generate Base grid
            # base_grid = torch.FloatTensor(disp.size())
            # linear_points = torch.linspace(-1, 1, W)
            # base_grid[:,1,:,:] = torch.ger(torch.ones(H),linear_points).expand_as(base_grid[:,0, :, :])
            # linear_points = torch.linspace(-1, 1, H)
            # base_grid[:,0,:,:] = torch.ger(linear_points,torch.ones(W)).expand_as(base_grid[:,1,:,:])

            # Check if base_grid is correct
            # base_grid[0][0][10][20]*W//2 + W//2 ~ 20
            # base_grid[0][1][10][20]*H//2 + H//2 ~ 10
            # YAYYYYYY

            orig_size = disp.size()
            disp = disp.resize(orig_size[0],orig_size[2],orig_size[3],2)
            # ORIGINAL
            # ----------
            grid_img = basegrid_img + disp
            grid_label = basegrid_label + disp

            # Image Transformation
            img1t = F.grid_sample(img1,grid_img)

            #### NOTE: Not sure if correct
            cprob2 = F.grid_sample(ohlabel1.float(),grid_label)

            logcprob2 = nn.LogSoftmax()(cprob2)

            # Loss Calculations
            Lsim = nn.MSELoss()(img1t,img2)
            Lseg = nn.NLLLoss()(logcprob2,label2)

            # Regularization for the transformation parameters
            disp = disp.view(-1,2,H,W)
            dispgrad = forward_diff(disp)

            # Try Soft L1 Loss for the grid gradient
            # zeros = Variable(torch.from_numpy(np.zeros(disp.data.shape)))
            # target = torch.cat((zeros,zeros),1).cuda().float() # Target has to be a float
            # Lreg = nn.SmoothL1Loss()(dispgrad,target)

            # DEBUG: Try using dispgrad as the regularization term
            Lreg = torch.sum(torch.abs(dispgrad))
            Ltotal = Lsim + args.lambdaseg*Lseg + args.lambdareg*Lreg
            opt.zero_grad()
            Ltotal.backward()

            opt.step()
            print("[{}][{}] Ltotal: {:.4} Lsim: {:.4f} Lseg: {:.4f} Lreg: {:.4f} ".format(epoch,itr,Ltotal.data[0],Lsim.data[0],args.lambdaseg*(Lseg.data[0]),args.lambdareg*(Lreg.data[0])))
        snapshot(net,"{}".format(epoch),os.path.join(homedir,'snapshots'))
        import pdb; pdb.set_trace()
if __name__ == "__main__":
    main()
