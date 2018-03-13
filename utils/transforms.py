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
    def __init__(self,nclass=4):
        self.nclass = nclass

    def __call__(self,label):
        if label.is_cuda:
            label_a = label.cpu().squeeze(0).byte().numpy()
        else:
            label_a = label.squeeze(0).byte().numpy()

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.nclass):
            ohlabel[c,:,:] = (label_a == c).astype(np.uint8)

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
    Takes PIL image in 'F' mode and returns a float32 tensor
"""
class ToTensorTIF(object):
    def __init__(self,norm=True):
        self.norm = norm

    def __call__(self,img):
        orig_size = img.size
        imgnp = np.array(img.getdata(),dtype=np.dtype('float32')).reshape((orig_size[1],orig_size[0]))
        # Add a fake dimension for channels
        imgnp = np.expand_dims(imgnp,axis=0)
        return torch.from_numpy(imgnp)
