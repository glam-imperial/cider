import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Resid_block(nn.Module):
    '''
    One simple residual block
    '''
    def __init__(self, inchannels, outchannels, padding=1, stride=1, downsample=None):
        super(Resid_block, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.downsample = downsample


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out


class ConvCore(nn.Module):
    def __init__(self, dropout=None, depth_scale=1, breathcough=False):
        super().__init__()
        self.d = depth_scale
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        self.conv1 = nn.Conv2d(2 if breathcough else 1,
                               int(32 * self.d),
                               kernel_size=(7, 3),
                               padding=1)
        self.resid1 = Resid_block(int(32 * self.d),
                                  int(64 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(32 * self.d),
                                                       int(64 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid2 = Resid_block(int(64 * self.d),
                                  int(128 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(64 * self.d),
                                                       int(128 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid3 = Resid_block(int(128 * self.d),
                                  int(256 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(128 * self.d),
                                                       int(256 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid4 = Resid_block(int(256 * self.d),
                                  int(512 * self.d),
                                  padding=(1, 1),
                                  stride=(2, 1),
                                  downsample=nn.Conv2d(int(256 * self.d),
                                                       int(512 * self.d),
                                                       kernel_size=1,
                                                       stride=(2, 1)))
        self.resid5 = Resid_block(int(512 * self.d),
                                  int(128 * self.d),
                                  stride=2,
                                  downsample=nn.Conv2d(int(512 * self.d),
                                                       int(128 * self.d),
                                                       kernel_size=1,
                                                       stride=2))
        self.resid6 = Resid_block(int(128 * self.d),
                                  int(64 * self.d),
                                  stride=2,
                                  downsample=nn.Conv2d(int(128 * self.d),
                                                       int(64 * self.d),
                                                       kernel_size=1,
                                                       stride=2))
        self.resid7 = Resid_block(int(64 * self.d),
                                  int(32 * self.d),
                                  stride=2,
                                  downsample=nn.Conv2d(int(64 * self.d),
                                                       int(32 * self.d),
                                                       kernel_size=1,
                                                       stride=2))
        self.resid8 = Resid_block(int(32 * self.d),
                                  int(16 * self.d),
                                  downsample=nn.Conv2d(int(32 * self.d),
                                                       int(16 * self.d),
                                                       kernel_size=1,
                                                       stride=1))
        self.resid9 = Resid_block(int(16 * self.d),
                                  int(8 * self.d),
                                  downsample=nn.Conv2d(int(16 * self.d),
                                                       int(8 * self.d),
                                                       kernel_size=1,
                                                       stride=1))

    def forward(self, x):
        out = self.conv1(x.float())
        if self.dropout != None:
            out = self.dropout(out)
        out = self.resid1(out)
        out = self.resid2(out)
        out = self.resid3(out)
        out = self.resid4(out)
        out = self.resid5(out)
        out = self.resid6(out)
        out = self.resid7(out)
        out = self.resid8(out)
        out = self.resid9(out)
        return out


class Conv_Model(ConvCore):
    def __init__(self, dropout=None, depth_scale=1, device="cuda", input_shape=(1025, 94), breathcough=False):
        super().__init__(dropout, depth_scale, breathcough=breathcough)
        self.to(device)
        out_tmp = super().forward(
            torch.randn(1, 2 if breathcough else 1, *input_shape).to(device))
        self.FC1 = nn.Linear(np.prod(out_tmp[-2:].shape), 50)
        self.FC2 = nn.Linear(50,1)
        self.relu = nn.ReLU()
        self.to(device)
        self.breathcough = breathcough

    def forward(self, x):
        out = super().forward(x)
        out = self.FC1(out.view(x.size()[0], -1))
        out = self.relu(out)
        out = self.FC2(out)
        return out


if __name__ == '__main__':
    for i in range(1, 6):
        x = torch.randn(8, 1, 1025, 94*i)
        model = Conv_Model(depth_scale=1.0, input_shape=x.shape[-2:])
        out = model(x)
    print('Done')
