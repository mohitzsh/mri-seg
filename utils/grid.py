import torch
from torch.nn import ConstantPad2d
from torch.nn import Hardtanh
from torch.nn import Sigmoid

def forward_diff(grid):
    nd = len(grid.data.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd

    # Forward difference along x direction
    axis = 3
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    # gridx = torch.nn.ConstantPad2d((0,1,0,0),0)(grid).clone()
    diffx = grid[slice1] - grid[slice2]
    diffx = ConstantPad2d((0,1,0,0),0)(diffx)
    # Forward difference along y direction
    axis = 2
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    diffy = grid[slice1] - grid[slice2]
    diffy = ConstantPad2d((0,0,1,0),0)(diffy)

    return torch.cat((diffx,diffy),1)

def process_disp(disp):
    disp = torch.transpose(disp,1,2)
    disp = torch.transpose(disp,2,3)
    # disp = Hardtanh()(disp)
    disp = Sigmoid()(disp)
    return disp
