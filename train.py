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
from datasets.mrbrains import MRBrainS
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
from utils.validate import validate_ibsr
from utils.validate import validate_sim
from utils.validate import validate_mrbrains
import tensorboardX as tbx
from torch.optim.lr_scheduler import StepLR
from parameternet.unet_v1 import UNetV1
import visdom
from utils.grid import process_disp
homedir = os.path.dirname(os.path.realpath(__file__))

H = 229
W = 193
nclasses = 4
w_init = None
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

    if torch.cuda.is_available():
        theta = theta.cuda()

    theta = Variable(theta)

    return F.affine_grid(theta,torch.Size(size))

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

def snapshot_path(base_snap_dir,exp,ext='.pth'):
    return os.path.join(base_snap_dir,exp+ext)

def take_snapshot(best_score,curr_score,opt,net,epoch,snapshot_path):
    if np.any(np.array(curr_score)[1:] > np.array(best_score)[1:]):
        print("Best score improved for at least one of the foreground class.\n Taking a snapshot")
        best_score = curr_score
        snapshot = {
            'epoch': epoch,
            'net': net.state_dict(),
            'best_score': best_score,
            'optimizer' : opt.state_dict()
        }
        torch.save(snapshot,snapshot_path)

    return best_score

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        if w_init == "xavier_uniform":
            nn.init.xavier_uniform(m.weight.data)
        elif w_init == "xavier_normal":
            nn.init.xavier_normal(m.weight.data)
        elif w_init == "he_uniform":
            nn.init.kaiming_uniform(m.weight.data,mode='fan_out')
        elif w_init =="he_normal":
            nn.init.kaiming_normal(m.weight.data,mode='fan_out')

def visualize_loss(Y,X,win,vis):
    vis.line(Y=Y,X=X,win=win,update='append')

def visualize_dice(Y,X,win,vis,opts):
    vis.line(Y=Y,X=X,win=win,update='append',opts=opts)

def visualize_transform_heatmap(X,title,vis):
    vis.heatmap(X=X,opts=dict(title=title))

"""
    Parse Arguments
"""
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir_2d",
                        help="Path to data directory for 2d slices")
    parser.add_argument("--data_dir_3d",
                        help="Path to data dir for 3d volumes")
    parser.add_argument("--dataset",choices=("ibsr","sim","mrbrains"),default="mrbrains")
    parser.add_argument("--snapshot_dir")
    parser.add_argument("--resume")
    parser.add_argument("--exp_name",
                        help="name of the experiments for prefixing saved models, images and video")
    parser.add_argument("--max_epoch",default=1,type=int,
                        help="Max epochs for training")
    parser.add_argument("--start_epoch",default=0,type=int)
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
    parser.add_argument("--l2_weight",type=float,default=10e-4)
    parser.add_argument("--nker",type=int,default=8)
    parser.add_argument("--step_lr_step_size",type=int,default=1)
    parser.add_argument("--step_lr_gamma",type=float,default=0.1)
    parser.add_argument("--weight_init",choices=('default','xavier_uniform','xavier_normal','he_uniform','he_normal'),default='default')
    parser.add_argument("--visdom_server",default="http://cuda.cs.purdue.edu")
    parser.add_argument("--visdom_port",type=int,default=1235)
    return parser.parse_args()

def main():
    args = parse_arguments()
    args_str = str(vars(args))
    global w_init
    w_init = args.weight_init

    # Setup tensorboardX logger
    # logger = tbx.SummaryWriter(log_dir = args.log_dir,comment=args.exp_name)
    # logger.add_text('training details',args_str,0)

    # Setup Visdom Logger
    vis = visdom.Visdom(server=args.visdom_server,port = int(args.visdom_port),env=args.exp_name)
    vis.close(win=None) # Close all existing windows from the current environment

    vis.text(args_str)
    ##############################################
    # Visdom Windows for Transformation Heatmaps #
    ##############################################

    #############################################
    # TRAINING DATASET: GENERIC TRANSFORMATION ##
    #############################################
    img_transform = [ToTensorTIF()]
    label_transform = [ToTensorLabel()]
    if args.dataset == "sim":
        trainset = Sim(args.data_dir_2d,args.train_s_idx,args.train_e_idx,co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    if args.dataset == "ibsr":
        trainset = IBSRv1(homedir,args.data_dir_2d,co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    if args.dataset == "mrbrains":
        trainset = MRBrainS(homedir,args.data_dir_2d,co_transform=Compose([]),
                    img_transform=Compose(img_transform),label_transform=Compose(label_transform))
    trainloader = DataLoader(trainset,batch_size =args.batch_size,shuffle=True,drop_last=True)
    print("Dataset Loaded")

    #####################
    # PARAMETER NETWORK #
    ####################

    # net = UNetV1(args.nker)
    net = UNetSmall(args.nker)
    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
    print("Network Loaded")
    net.apply(weight_init)

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
    if torch.cuda.is_available():
        theta = theta.cuda()

    theta = Variable(theta)
    basegrid_img = F.affine_grid(theta,torch.Size((args.batch_size,1,H,W)))
    basegrid_label = F.affine_grid(theta,torch.Size((args.batch_size,nclasses,H,W)))

    best_score = [0,0,0,0]

    ##############################################
    # Resume training is args.resume is not None #
    ##############################################
    scheduler = StepLR(opt, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma)
    if args.resume is not None:
        print("Resuming Training from {}".format(args.resume))
        snapshot = torch.load(args.resume)
        args.start_epoch = snapshot['epoch']
        best_score = snapshot['best_score']
        net.load_state_dict(snapshot['net'])
        opt.load_state_dict(snapshot['optimizer'])

    else:
        print("No Checkpoint Found")

    #####################
    # VISDOM LOSS SETUP #
    #####################
    win_loss_total = vis.line(Y=np.empty(1),opts=dict(title='loss_total'))
    win_loss_sim = vis.line(Y=np.empty(1),opts=dict(title='loss_sim'))
    win_loss_reg = vis.line(Y=np.empty(1),opts=dict(title='loss_reg'))
    win_loss_seg = vis.line(Y=np.empty(1),opts=dict(title='loss seg'))

    #####################
    # VISDOM DICE SETUP #
    #####################
    dice_opts_t1 = dict(
        title='dice_t1',
        legend=['0','1','2','3'],
    )
    dice_opts_t1_ir = dict(
        title='dice_t1_ir',
        legend=['0','1','2','3'],
    )
    dice_opts_t2 = dict(
        title='dice_t2',
        legend=['0','1','2','3'],
    )
    # empty_data = np.empty((1,nclasses))
    # empty_data[...] = np.nan
    # win_dice_t1 = vis.line(Y=empty_data,opts=dice_opts_t1)
    # win_dice_t1_ir = vis.line(Y=empty_data,opts=dice_opts_t1_ir)
    # win_dice_t2 = vis.line(Y=empty_data,opts=dice_opts_t2)
    win_dice_t1 = None
    win_dice_t1_ir = None
    win_dice_t2 = None

    #############
    ## TRAINING #
    #############


    ##########
    # DEBUG  #
    #########
    #val_results = validate_mrbrains(net,args.data_dir_3d,[1,2,3],[4])
    ####################################

    for epoch in np.arange(args.start_epoch,args.max_epoch):
        scheduler.step()

        loss_total = []
        loss_sim = []
        loss_reg = []
        loss_seg = []
        steps = []

        for batch_id, ((img1,label1,ohlabel1,fname1),(img2,label2,ohlabel2,fname2)) in enumerate(trainloader):
            if img1 is None or label1 is None or img2 is None or label2 is None or ohlabel1 is None:
                continue
            net.train()
            itr = len(trainloader)*(epoch) + batch_id
            steps.append(itr)
            ####################
            # Debug Snapshot code
            #################
            if torch.cuda.is_available():
                img1, label1, img2, label2,combimg,ohlabel1 = Variable(img1.cuda()),\
                        Variable(label1.cuda()), Variable(img2.cuda()), Variable(label2.cuda()),\
                        Variable(torch.cat((img1,img2),1).cuda()), Variable(ohlabel1.cuda())
            else:
                img1, label1, img2, label2,combimg, ohlabel1 = Variable(img1), Variable(label1),\
                        Variable(img2), Variable(label2), Variable(torch.cat((img1,img2),1)), Variable(ohlabel1)
            # (disp0,disp1,disp2) = net(combimg)
            # disp0 = reshape_transform(disp0)
            # disp1 = reshape_transform(disp1)
            # disp2 = reshape_transform(disp2)
            disp = net(combimg)
            disp = process_disp(disp)

            ##########################
            ## IMAGE TRANSFORMATION ##
            ##########################
            # grid_img0 = basegrid_img + disp0
            # grid_img1 = basegrid_img + disp1 + disp0
            # grid_img2 = basegrid_img + disp2 + disp1 + disp0
            grid_img = basegrid_img + disp

            # img1t0 = F.grid_sample(img1,grid_img0)
            # img1t1 = F.grid_sample(img1,grid_img1)
            # img1t2 = F.grid_sample(img1,grid_img2)
            img1t = F.grid_sample(img1,grid_img)

            if args.similarity == 'cc':
                # Lsim0 = cc(img1t0.data,img2.data)
                # Lsim1 = cc(img1t1.data,img2.data)
                # Lsim2 = cc(img1t2.data,img2.data)
                # Lsim = Lsim0 + Lsim1 + Lsim2
                Lsim = cc(img1t,img2)
            elif args.similarity == 'l2':
                Lsim = nn.MSELoss()(img1t,img2)


            ###########################
            ### LABEL TRANSFORMATION ##
            ###########################
            Lseg = Variable(torch.Tensor([0]),requires_grad=True)
            if torch.cuda.is_available():
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
            if torch.cuda.is_available():
                Lreg = Variable(torch.Tensor([0]).cuda(),requires_grad=True)
            if args.lambdareg != 0:
                # disp = disp.view(-1,2,h,w)
                dx = disp[:,1:,:,:] -disp[:,:-1,:,:]
                dy = disp[:,:,1:,:] - disp[:,:,:-1,:]
                # Implement L1 penalty for now
                # Try to constrain the second derivative
                d2dx2 = torch.abs(dx[:,1:,:,:] - dx[:,:-1,:,:])
                d2dy2 = torch.abs(dy[:,:,1:,:] - dy[:,:,:-1,:])
                d2dxdy = torch.abs(dx[:,:,1:,:] - dx[:,:,:-1,:])
                d2dydx = torch.abs(dy[:,1:,:,:] - dy[:,:-1,:,:])

                d2_mean = (torch.mean(d2dx2) + torch.mean(d2dy2) + torch.mean(d2dxdy) + torch.mean(d2dydx))/4

                # dx_mean = torch.mean(dx)
                # dy_mean = torch.mean(dy)
                # target = torch.zeros(1)
                # if args.gpu:
                #     target = target.cuda()
                # Lreg = nn.L1Loss()((dx_mean+dy_mean)/2,Variable(target))
                Lreg = d2_mean
            ######################
            ## PARAMETER UPDATE ##
            ######################
            Ltotal = Lsim + args.lambdareg*Lreg + args.lambdaseg*Lseg
            opt.zero_grad()
            # opt,lr=poly_lr_scheduler(opt, args.lr, itr,max_iter=len(trainloader)*args.max_epoch)
            Ltotal.backward()
            opt.step()

            free_vars(img1,img2,label1,label2,combimg,ohlabel1,ohlabel2,target)


            loss_total.append(Ltotal.data[0])
            loss_reg.append(Lreg.data[0])
            loss_sim.append(Lsim.data[0])
            loss_seg.append(Lseg.data[0])

            if itr % args.error_iter ==0:
                print("[{}][{}] Ltotal: {:.6} Lsim: {:.6f} Lseg: {:.6f} Lreg: {:.6f} ".\
                    format(epoch,itr,Ltotal.data[0],Lsim.data[0],args.lambdaseg*(Lseg.data[0]),Lreg.data[0]))

        ############
        # VALIDATE #
        ############
        if args.dataset == "sim":
            score = validate_sim(net,args.data_dir_2d,args.train_s_idx,args.train_e_idx,args.val_s_idx,args.val_e_idx,args.gpu)
        if args.dataset == "ibsr":
            score = validate_ibsr(net,args.data_dir_3d,args.train_vols,args.val_vols,args.img_suffix,args.cls_suffix,perm,args.gpu)
        if args.dataset == "mrbrains":
            val_results = validate_mrbrains(net,args.data_dir_3d,[1,2,3],[4]) # Makeshift changes! Be very careful
            dice_t1 = val_results['scores']['t1']
            dice_t1_ir = val_results['scores']['t1_ir']
            dice_t2 = val_results['scores']['t2']
            if torch.cuda.is_available():
                dice_t1 = dice_t1.cpu().numpy()
                dice_t1_ir = dice_t1_ir.cpu().numpy()
                dice_t2 = dice_t2.cpu().numpy()
            else:
                dice_t1 = dice_t1.numpy()
                dice_t1_ir = dice_t1_ir.numpy()
                dice_t2 = dice_t2.numpy()
            print("dice_t1: {}".format(dice_t1))
            print("dice_t1_ir: {}".format(dice_t1_ir))
            print("dice_t2: {}".format(dice_t2))
            score = (dice_t1 + dice_t1_ir + dice_t2 )/3
        # ############
        # # SNAPSHOT #
        # ############
        best_score = take_snapshot(best_score,score,opt,net,epoch,snapshot_path(args.snapshot_dir,args.exp_name))

        ########################
        ########################
        ## VISUALIZATION CODE ##
        ########################
        ########################


        # ##################
        # # VISUALIZE LOSS #
        # ##################
        vis.line(Y=np.array(loss_total),X=np.array(steps),win=win_loss_total,update='append')
        vis.line(Y=np.array(loss_sim),X=np.array(steps),win=win_loss_sim,update='append')
        vis.line(Y=np.array(loss_reg),X=np.array(steps),win=win_loss_reg,update='append')
        vis.line(Y=np.array(loss_seg),X=np.array(steps),win=win_loss_seg,update='append')


        ########################
        # VISUALIZE THE SCORES #
        ########################
        if win_dice_t1 is None:
            win_dice_t1 = vis.line(Y=dice_t1.reshape(1,nclasses),X=np.array([epoch]),opts=dict(title='dice_t1'))
        else:
            vis.line(Y=dice_t1.reshape(1,nclasses),X=np.array([epoch]),win=win_dice_t1,update='append')
        if win_dice_t1_ir is None:
            win_dice_t1_ir = vis.line(Y=dice_t1_ir.reshape(1,nclasses),X=np.array([epoch]),opts=dict(title='dice_t1_ir'))
        else:
            vis.line(Y=dice_t1_ir.reshape(1,nclasses),X=np.array([epoch]),win=win_dice_t1_ir,update='append')
        if win_dice_t2 is None:
            win_dice_t2 = vis.line(Y=dice_t2.reshape(1,nclasses),X=np.array([epoch]),opts=dict(title='dice_t2'))
        else:
            vis.line(Y=dice_t2.reshape(1,nclasses),X=np.array([epoch]),win=win_dice_t2,update='append')

        # #####################################################################################
        # # VISUALIZE THE TRANFORMATION PARAMETER OF CENTRAL SLICE AS HEATMAP FOR TRAIN VOL 0 #
        # #####################################################################################
        field_t1 = val_results['fields']['t1']
        field_t1_ir = val_results['fields']['t1_ir']
        field_t2 = val_results['fields']['t2']
        if torch.cuda.is_available():
            field_t1 = field_t1.cpu()
            field_t1_ir = field_t1_ir.cpu()
            field_t2  = field_t2.cpu()

        transform_x_t1 = field_t1[0,0,100,:,:,0]
        transform_y_t1 = field_t1[0,0,100,:,:,1]
        transform_x_t1_ir = field_t1_ir[0,0,100,:,:,0]
        transform_y_t1_ir = field_t1_ir[0,0,100,:,:,1]
        transform_x_t2 = field_t2[0,0,100,:,:,0]
        transform_y_t2 = field_t2[0,0,100,:,:,1]

        vis.heatmap(X=transform_x_t1,opts=dict(title="t1_0_x_{}".format(epoch)))
        vis.heatmap(X=transform_y_t1,opts=dict(title="t1_0_y_{}".format(epoch)))
        vis.heatmap(X=transform_x_t1_ir,opts=dict(title="t1_ir_0_x_{}".format(epoch)))
        vis.heatmap(X=transform_y_t1_ir,opts=dict(title="t1_ir_0_y_{}".format(epoch)))
        vis.heatmap(X=transform_x_t2,opts=dict(title="t2_0_x_{}".format(epoch)))
        vis.heatmap(X=transform_y_t2,opts=dict(title="t2_0_y_{}".format(epoch)))

        vis.save([args.exp_name])


if __name__ == "__main__":
    main()
