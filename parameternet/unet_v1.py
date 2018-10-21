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

class trans(nn.Module):
    def __init__(self,in_ch,upscale_factor,target_shape,n_classes):
        super(trans,self).__init__()
        self.target_shape = target_shape
        self.out = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Conv2d(in_ch,n_classes,1),
            nn.UpsamplingBilinear2d(scale_factor=upscale_factor)
        )
    def forward(self,x):
        x = self.out(x)
        if (x.shape[-2:] != self.target_shape):
            # Pad x to match the target size
            y_pad = self.target_shape[0] - x.shape[-2]
            x_pad = self.target_shape[1] - x.shape[-1]

            x = F.pad(x,(0,x_pad,0,y_pad),mode='reflect')

        return x

class UNetV1(nn.Module):
    def __init__(self, nker = 8,n_channels=2, n_classes=2,target_shape=(229,193)):
        super(UNetV1, self).__init__()
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
        # self.trans4 = trans(2*nker,8,target_shape,n_classes)
        # self.trans2 = trans(nker,2,target_shape,n_classes)
        self.trans3 = trans(2*nker,4,target_shape,n_classes)
        self.trans5 = trans(4*nker,16,target_shape,n_classes)


    def forward(self, x):

        x1 = self.inc(x) # 1
        x2 = self.down1(x1) #1/2
        x3 = self.down2(x2) #1/4
        trans3 = self.trans3(x3)
        x4 = self.down3(x3) #1/8
        x5 = self.down4(x4) #1/16
        trans5 = self.trans5(x5)
        x4 = self.up1(x5, x4) #1/8
        # trans4 = self.trans4(x4)
        x3 = self.up2(x4, x3) #1/4
        x2 = self.up3(x3, x2) #1/2
        # trans2 = self.trans2(x2)
        x1 = self.up4(x2, x1) # 1
        x = self.outc(x1)
        return (trans5,trans3,x)
