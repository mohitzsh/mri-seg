import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        size1 = (x1.size()[2],x1.size()[3])
        size2 = (x2.size()[2],x2.size()[3])

        max_size = (max(size1[0],size2[0]),max(size1[1],size2[1]))
        diffx1_x = abs(max_size[0] - size1[0])
        diffx1_y = abs(max_size[1] - size1[1])
        diffx2_x = abs(max_size[0] - size2[0])
        diffx2_y = abs(max_size[1] - size2[1])

        x1 = F.pad(x1,(math.ceil(diffx1_y/2),math.floor(diffx1_y/2),math.ceil(diffx1_x/2),math.floor(diffx1_x/2)),mode='reflect')
        x2 = F.pad(x2,(math.ceil(diffx2_y/2),math.floor(diffx2_y/2),math.ceil(diffx2_x/2),math.floor(diffx2_x/2)),mode='reflect')

        assert(x1.shape == x2.shape)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNetSmall(nn.Module):
    def __init__(self, nker = 8,n_channels=2, n_classes=2):
        super(UNetSmall, self).__init__()
        self.inc = inconv(n_channels, nker)
        self.down1 = down(nker, nker)
        self.down2 = down(nker, 2*nker)
        self.down3 = down(2*nker, 4*nker)
        self.down4 = down(4*nker, 4*nker)
        self.up1 = up(8*nker, 2*nker)
        self.up2 = up(4*nker, nker)
        self.up3 = up(2*nker, nker)
        self.up4 = up(2*nker, nker)
        self.outc = outconv(nker, n_classes)

    def forward(self, x):
        x1 = self.inc(x)     # x1  : 64
        x2 = self.down1(x1)  # x2  : 128
        x3 = self.down2(x2)  # x3  : 256
        x4 = self.down3(x3)  # x4  : 512
        x5 = self.down4(x4)  # x5  : 512
        x = self.up1(x5, x4) # x   : 256
        x = self.up2(x, x3)  # x   : 128
        x = self.up3(x, x2)  #
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
