import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

"""
    in1 : Nx1xHxW
    in2 : Nx1xHxW
    N : batch size
    n has to be an odd number

    Returns a per-batch cross-correlation
"""
def cc(I,J,n=9,eps=10e-5):
    I2 = I*I
    J2 = J*J
    IJ = I*J
    ker = torch.ones((1,1,9,9))
    if I.is_cuda:
        ker = ker.cuda()
    I_sum = F.conv2d(I,Variable(ker),padding=n//2)
    J_sum = F.conv2d(J,Variable(ker),padding=n//2)
    IJ_sum = F.conv2d(IJ,Variable(ker),padding=n//2)
    I2_sum = F.conv2d(I2,Variable(ker),padding=n//2)
    J2_sum = F.conv2d(J2,Variable(ker),padding=n//2)
    I_mean = I_sum/(1.0*n*n)
    J_mean = J_sum/(1.0*n*n)

    num = (IJ_sum - I_mean*J_sum - J_mean*I_sum + I_mean*J_mean*n*n)**2
    denom1 = (I2_sum + I_mean*I_mean*n*n - 2*I_mean*I_sum)
    denom2 = (J2_sum + J_mean*J_mean*n*n - 2*J_mean*J_sum)
    denom = denom1*denom2
    # loss = -1.0*torch.mean(num/(denom + 1e-5))
    # denom = denom1*denom2
    # # filter out negative values
    # denom[denom<0] = 0\
    loss = -1.0*torch.log(torch.mean((num+1)/(denom+1)))
    return loss
