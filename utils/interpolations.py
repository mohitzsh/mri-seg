import torch
from torch.autograd import Variable
dtype = torch.FloatTensor
dtype_long = torch.LongTensor

def sample(label,x0,x1,y0,y1,H,W):
    out = Variable(torch.Tensor(label.size()))
    for x in range(W):
        for y in range(H):
            idx0 = x0[:,y,x].data[0]
            idx1 = x1[:,y,x].data[0]
            idy0 = y0[:,y,x].data[0]
            idy1 = y1[:,y,x].data[0]
            out[:,:,y,x] = (label[:,:,idy0,idx0] + label[:,:,idy1,idx0] + label[:,:,idy0,idx1] + label[:,:,idy1,idx1])
    return out
def grid_sample_labels(label, grid):
    _,_,H,W = label.size()
    x = grid[:,:,:,0]
    y = grid[:,:,:,1]

    x = x*(W//2) + W//2 - 1
    y = y*(H//2) + H//2 - 1
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W-1)
    x1 = torch.clamp(x1, 0, W-1)
    y0 = torch.clamp(y0, 0, H-1)
    y1 = torch.clamp(y1, 0, H-1)

    out = sample(label,x0,x1,y0,y1,H,W)
    return out
