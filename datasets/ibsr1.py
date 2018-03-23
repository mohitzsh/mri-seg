import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose
from utils.transforms import OneHotEncode

def load_list_train(filename):
    f = open(filename,'r')
    names = [name.strip() for name in f.readlines()]
    pairs = []

    for i in range(len(names)-1):
        for j in range(i+1,len(names)):
            if names[i].split('_')[1] == names[j].split('_')[1]:
                pairs.append((names[i],names[j]))
                pairs.append((names[j],names[i]))

    return pairs

def load_data(filename,datadir,co_transform,img_transform,label_transform):
    f = open(filename,'r')
    names = [name.strip() for name in f.readlines()]
    img_dict = {}
    label_dict = {}
    oh_label_dict = {}
    for name in names:
        assert(os.path.exists(os.path.join(datadir,"img",name+".tif")))
        assert(os.path.exists(os.path.join(datadir,"cls",name+".png")))

        img = Image.open(os.path.join(datadir,"img",name+".tif"))
        label = Image.open(os.path.join(datadir,"cls",name+".png")).convert('P')
        img,label = co_transform((img,label))
        img = img_transform(img)
        label = label_transform(label)
        ohlabel = OneHotEncode()(label)

        img_dict[name] = img
        label_dict[name] = label
        oh_label_dict[name] = label
    return img_dict,label_dict,oh_label_dict

class IBSRv1(Dataset):
    TRAIN_LIST = 'lists/train_list.txt'
    VAL_LIST = 'lists/val_list.txt'

    def __init__(self,root,datadir,
                co_transform=Compose([]),img_transform=Compose([]),
                label_transform=Compose([])):
        self.root = root
        self.datadir = datadir
        self.co_transform = co_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.data = load_data(os.path.join(self.root,"datasets",self.TRAIN_LIST),datadir,co_transform,img_transform,label_transform)
        self.list = load_list_train(os.path.join(self.root,"datasets",self.TRAIN_LIST))

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
