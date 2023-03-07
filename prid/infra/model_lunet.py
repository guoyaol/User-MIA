import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
import torchvision
import torch.utils.model_zoo as model_zoo


class LuNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LuNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7)
        self.resblock1 = Bottleneck(128, 32, 128)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)

        self.resblock2_1 = Bottleneck(128, 32, 128)
        self.resblock2_2 = Bottleneck(128, 32, 128)
        self.resblock2_3 = Bottleneck(128, 32, 256)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)

        self.resblock3_1 = Bottleneck(256, 64, 256)
        self.resblock3_2 = Bottleneck(256, 64, 256)
        self.pool3 = nn.MaxPool2d(3, 2, padding=1)

        self.resblock4_1 = Bottleneck(256, 64, 256)
        self.resblock4_2 = Bottleneck(256, 64, 256)
        self.resblock4_3 = Bottleneck(256, 128, 512)
        self.pool4 = nn.MaxPool2d(3, 2, padding=1)

        self.resblock5_1 = Bottleneck(512, 128, 512)
        self.resblock5_2 = Bottleneck(512, 128, 512)
        self.pool5 = nn.MaxPool2d(3, 2, padding=1)

        self.resblock6 = SpecialBlock(512, 128)

        self.linear1 = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.linear2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.pool1(x)

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        x = self.resblock2_3(x)
        x = self.pool2(x)

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        x = self.pool3(x)

        x = self.resblock4_1(x)
        x = self.resblock4_2(x)
        x = self.resblock4_3(x)
        x = self.pool4(x)

        x = self.resblock5_1(x)
        x = self.resblock5_2(x)
        x = self.pool5(x)

        x = self.resblock6(x)

        x = x.view(x.shape[0], -1)

        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SpecialBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chan, out_chan, stride=1):
        super(SpecialBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, in_chan,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_chan)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_chan, out_chan,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chan)
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=3,
                          stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_chan))
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan, stride=1, stride_at_1x1=False, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)

        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=stride1x1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=stride3x3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_chan))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample == None:
            residual = x
        else:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

if __name__ == "__main__":
    input = torch.ones(32,3,128,64)
    model = LuNet()
    model.eval()
    output = model(input)
    print(output.shape)
