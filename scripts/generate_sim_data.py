import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
from PIL import Image
import numbers
import argparse
from scipy.interpolate import interpn
from PIL import Image
import os
import matplotlib.pyplot as plt
def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--H",type=int,default=218)
    parser.add_argument("--W",type=int,default=182)
    parser.add_argument("--alpha",type=int,default=10)
    parser.add_argument("--sigma",type=int,default=16)
    parser.add_argument("--dataset_size",type=int,default=6000)
    parser.add_argument("--delta_c",type=int,default=5)
    parser.add_argument("--delta_alpha",type=int,default=5)
    parser.add_argument("--delta_sigma",type=int,default=5)
    parser.add_argument("--delta_r",type=int,default=5)
    parser.add_argument("--int1",type=float,default=0.8)
    parser.add_argument("--int2",type=float,default=0.3)
    parser.add_argument("--int3",type=float,default=0.5)
    parser.add_argument("--lab1",type=float,default=1)
    parser.add_argument("--lab2",type=float,default=2)
    parser.add_argument("--lab3",type=float,default=3)
    parser.add_argument("--data_dir")

    return parser.parse_args()

def elastic_transform(img,cls, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert img.ndim == 3
    assert cls.ndim == 3
    assert img.shape == cls.shape
    shape = img.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),sigma, mode="constant", cval=0) * alpha


    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    orig_points = [np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))]
    new_points = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]

    result_img = np.empty_like(img)
    result_cls = np.empty_like(cls)

    for i in range(img.shape[2]):
        result_img[:, :, i] = map_coordinates(img[:, :, i], new_points, order=spline_order, mode=mode).reshape(shape)

    x1 = np.arange(shape[0])
    y1 = np.arange(shape[1])

    delta = np.concatenate((dx[...,None],dy[...,None]),axis=2)
    orig_pos = np.concatenate((x[...,None],y[...,None]),axis=2)
    new_pos = orig_pos + delta
    result_cls = interpn((x1,y1),cls,new_pos,bounds_error=False,fill_value=0,method='nearest')

    return result_img,result_cls

def transform(shape,base_r1,base_r2,base_r3,base_cx,base_cy,args):
    delta_cx = base_cx*args.delta_c//100
    delta_cy = base_cy*args.delta_c//100
    cx = base_cx + np.random.randint(-1*delta_cx-1,delta_cx+1)
    cy = base_cy + np.random.randint(-1*delta_cy-1,delta_cy+1)

    # Generate Random radius
    delta_r1 = base_r1*args.delta_r//100
    delta_r2 = base_r2*args.delta_r//100
    delta_r3 = base_r3*args.delta_r//100

    r1 = base_r1 + np.random.randint(-1*delta_r1-1,delta_r1+1)
    r2 = base_r2 + np.random.randint(-1*delta_r2-1,delta_r2+1)
    r3 = base_r3 + np.random.randint(-1*delta_r3-1,delta_r3+1)

    # Generate Random alpha
    delta_alpha = args.alpha*args.delta_alpha/100
    alpha = args.alpha +  np.random.randint(-1*delta_alpha,delta_alpha)

    # Generate Random sigma
    delta_sigma = args.sigma*args.delta_sigma/100
    sigma = args.sigma + np.random.uniform(-1*delta_sigma,delta_sigma)

    y, x = np.ogrid[-r3: r3, -r3: r3]
    idx1 = x**2 + y**2 <= r1**2
    idx2 = np.logical_and(x**2 + y**2 > r1**2,x**2 + y**2 <= r2**2)
    idx3 = np.logical_and(x**2 + y**2 > r2**2,x**2 + y**2 <= r3**2)

    img = np.zeros(shape,dtype=np.float32)
    cls = np.zeros(shape,dtype=np.int8)

    img[cy - r3:cy+r3,cx-r3:cx+r3][idx1] = args.int1
    img[cy - r3:cy+r3,cx-r3:cx+r3][idx2] = args.int2
    img[cy - r3:cy+r3,cx-r3:cx+r3][idx3] = args.int3

    cls[cy - r3:cy+r3,cx-r3:cx+r3][idx1] = args.lab1
    cls[cy - r3:cy+r3,cx-r3:cx+r3][idx2] = args.lab2
    cls[cy - r3:cy+r3,cx-r3:cx+r3][idx3] = args.lab3

    new_img,new_cls = elastic_transform(img,cls,alpha,sigma)

    return new_img,new_cls

if __name__ == "__main__":
    args = arguments()
    base_r1 = 20
    base_r2 = 40
    base_r3 = 70

    base_cx,base_cy = args.W//2,args.H//2
    for data_idx in range(args.dataset_size):
        img,cls = transform((args.H,args.W,1),base_r1,base_r2,base_r3,base_cx,base_cy,args)
        cls = cls.astype(np.int8)
        # data folder
        if not os.path.exists(os.path.join(args.data_dir,"data")):
            os.makedirs(os.path.join(args.data_dir,"data"))
        if not os.path.exists(os.path.join(args.data_dir,"data","img")):
            os.makedirs(os.path.join(args.data_dir,"data","img"))
        if not os.path.exists(os.path.join(args.data_dir,"data","cls")):
            os.makedirs(os.path.join(args.data_dir,"data","cls"))

        if not os.path.exists(os.path.join(args.data_dir,"vis","cls")):
            os.makedirs(os.path.join(args.data_dir,"vis","cls"))

        if not os.path.exists(os.path.join(args.data_dir,"vis","img")):
            os.makedirs(os.path.join(args.data_dir,"vis","img"))

        data_img_path = os.path.join(args.data_dir,"data","img",str(data_idx))
        data_cls_path = os.path.join(args.data_dir,"data","cls",str(data_idx))
        vis_img_path = os.path.join(args.data_dir,"vis","img",str(data_idx))
        vis_cls_path = os.path.join(args.data_dir,"vis","cls",str(data_idx))

        img_pil = Image.fromarray(img[...,0])
        img_pil.save(data_img_path+'.tif')
        cls_pil = Image.fromarray(cls[...,0])
        cls_pil.save(data_cls_path+'.png')

        # visualization folder
        plt.imsave(vis_img_path+'.png',img[...,0])
        plt.imsave(vis_cls_path+'.png',cls[...,0])

        import pdb; pdb.set_trace()
        print("{} DONE".format(data_idx))
