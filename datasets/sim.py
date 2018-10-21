from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from utils.transforms import OneHotEncode
from torchvision.transforms import Compose

def load_data(start,end,data_dir,img_transform,label_transform,co_transform):
    img_dir = os.path.join(data_dir,"img")
    cls_dir = os.path.join(data_dir,"cls")
    img_arr = []
    cls_arr = []
    cls_oh_arr = []
    for idx in range(start,end+1):
        img = Image.open(os.path.join(img_dir,str(idx)+'.tif'))
        cls = Image.open(os.path.join(cls_dir,str(idx)+'.png')).convert('P')
        img,cls = co_transform((img,cls))
        img = img_transform(img)
        cls = label_transform(cls)
        cls_ohe = OneHotEncode()(cls)

        img_arr.append(img)
        cls_arr.append(cls)
        cls_oh_arr.append(cls_ohe)
    return img_arr,cls_arr,cls_oh_arr

def get_pairs(start,end):
    pairs = []
    for s_idx in np.arange(end):
        pairs.append((s_idx,s_idx))
        for e_idx in np.arange(s_idx +1, end+1):
            pairs.append((s_idx,e_idx))
            pairs.append((e_idx,s_idx))

    return pairs
class Sim(Dataset):

    def __init__(self,
                data_dir,
                start_idx,
                end_idx,
                img_transform = Compose([]),
                label_transform = Compose([]),
                co_transform = Compose([])):

        self.data_dir = data_dir
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.data = load_data(start_idx,end_idx,data_dir,img_transform,label_transform,co_transform)
        self.data_list = get_pairs(start_idx,end_idx)

    def __getitem__(self,index):
        fname1, fname2 = self.data_list[index]

        idx1 = fname1 - self.start_idx
        idx2 = fname2 - self.start_idx

        img_arr,cls_arr,cls_oh_arr = self.data
        img1 = img_arr[idx1]
        img2 = img_arr[idx2]
        cls1 = cls_arr[idx1]
        cls2 = cls_arr[idx2]
        oh1 = cls_oh_arr[idx1]
        oh2 = cls_oh_arr[idx2]

        return ((img1,cls1,oh1,str(fname1)),(img2,cls2,oh2,str(fname2)))
    def __len__(self):
        return len(self.data_list)
