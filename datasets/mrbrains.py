import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose
from utils.transforms import OneHotEncode
import numpy as np
import torch

def is_background(cls):
    if torch.cuda.is_available():
        cls = cls.cpu()
    cls_np = cls.numpy()
    counts = np.unique(cls_np,return_counts=True)[1]
    if counts[0]/np.sum(counts) > 0.99:
        return True


def make_pairs(filename,val_subject,data):
    f = open(filename,'r')
    names = [name.strip() for name in f.readlines()]
    background = np.zeros((len(names)))
    pairs = []
    # Find background images
    for idx,name in enumerate(names):
        if is_background(data[1][name]):
            background[idx] = 1
    print("Background check done")
    # Only read those files where there is something other than background
    for i in range(len(names)-1):
        # Ignore validation images
        if names[i].split("_")[0] == str(val_subject):
            continue
        # Ignore images with only background class
        if background[i] == 1:
            continue
        for j in range(i+1,len(names)):
            if background[j] == 1:
                continue
            slice_idx_i = names[i].split("_")[-1]
            slice_idx_j = names[j].split("_")[-1]
            # pairs.append((names[i],names[i]))
            if slice_idx_i == slice_idx_j:
                pairs.append((names[i],names[j]))
                pairs.append((names[j],names[i]))
    print("Done making pairs")
    return pairs

def load_data(filename,datadir,co_transform,img_transform,label_transform):
    f = open(filename,'r')
    names = [name.strip() for name in f.readlines()]
    img_dict = {}
    label_dict = {}
    oh_label_dict = {}
    for name in names:
        cls_name = name.split("_")[0] + '_' + name.split("_")[-1]
        assert(os.path.exists(os.path.join(datadir,"img",name+".tif")))
        assert(os.path.exists(os.path.join(datadir,"cls",cls_name+".png")))

        img = Image.open(os.path.join(datadir,"img",name+".tif"))
        label = Image.open(os.path.join(datadir,"cls",cls_name+".png")).convert('P')
        img,label = co_transform((img,label))
        img = img_transform(img)
        label = label_transform(label)
        ohlabel = OneHotEncode()(label)

        img_dict[name] = img
        label_dict[name] = label
        oh_label_dict[name] = label
    return img_dict,label_dict,oh_label_dict

class MRBrainS(Dataset):
    data_list = 'lists/train_list_MRBrainS.txt'

    def __init__(self,root,datadir,
                co_transform=Compose([]),img_transform=Compose([]),
                label_transform=Compose([]),val_subject=4):
        self.root = root
        self.datadir = datadir
        self.val_subject = val_subject
        self.co_transform = co_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.data = load_data(os.path.join(self.root,"datasets",self.data_list),datadir,co_transform,img_transform,label_transform)
        self.list = make_pairs(os.path.join(self.root,"datasets",self.data_list),self.val_subject,self.data)

    def __getitem__(self,index):
        fname1,fname2 = self.list[index]

        img1 = self.data[0][fname1]
        img2 = self.data[0][fname2]
        label1 = self.data[1][fname1]
        label2 = self.data[1][fname2]
        ohlabel1 = self.data[2][fname1]
        ohlabel2 = self.data[2][fname2]

        return ((img1,label1,ohlabel1,fname1),(img2,label2,ohlabel2,fname2))

    def __len__(self):
        return len(self.list)
