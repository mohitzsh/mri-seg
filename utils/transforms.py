import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class ToTensorLabel(object):
    """
        Take a Label as PIL.Image with 'P' mode and convert to Tensor
    """
    def __init__(self,tensor_type=torch.LongTensor):
        self.tensor_type = tensor_type

    def __call__(self,label):
        label = np.array(label,dtype=np.uint8)
        label = torch.from_numpy(label).type(self.tensor_type)

        return label

class OneHotEncode(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self,nclass=5):
        self.nclass = nclass

    def __call__(self,label):
        label_a = np.array(transforms.ToPILImage()(label.byte().unsqueeze(0)),np.uint8)

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.nclass):
            ohlabel[c:,:,:] = (label_a == c).astype(np.uint8)

        # # Do Some assertion
        # print("Assertion about to be made")
        # for c in range(self.nclass):
        #     for i in range(321):
        #         for j in range(321):
        #             if ohlabel[c][i][j] == 1:
        #                 assert(label_a[i][j] == c)

        return torch.from_numpy(ohlabel)

class RandomRotation(object):
    """
        Takes a PIL Image and rotate
    """
    def __init__(self,degree,resample=Image.BILINEAR):
        self.degree = degree
        self.resample = resample
    def __call__(self,img):
        degree = 2*self.degree*np.random.random_sample() + self.degree
        return img.rotate(degree,resample=self.resample)

class Rotation(object):
    """
        Takes a PIL Image and rotate
    """
    def __init__(self,degree,resample=Image.BILINEAR):
        self.degree = degree
        self.resample = resample
    def __call__(self,img):
        return img.rotate(self.degree,resample=self.resample)

"""
    Takes PIL image in 'F' mode and returns normalize image [-1,1]
"""
class ToTensorTIF(object):
    def __init__(self,norm=True):
        self.norm = norm

    def __call__(self,img):
        imgnp = np.array(img)[:,:,None]
        if np.max(imgnp) != 0:
            imgnp = imgnp/np.max(imgnp)
        if np.min(imgnp) != 0:
            imgnp = imgnp/np.abs(np.min(imgnp))

        return transforms.ToTensor()(imgnp)
