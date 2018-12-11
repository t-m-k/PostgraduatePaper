import torch
import torch.nn as nn
import math
from tensorboardX import SummaryWriter

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
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


class ResNetBlock(nn.Module):

    def __init__(self, block, layers,inchannels = 256):
        self.inchannels = inchannels
        super(ResNetBlock, self).__init__()
        self.layer2 = self._make_layer(block, layers, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inchannels, self.inchannels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.inchannels),
        )

        layers = []
        layers.append(block(self.inchannels, self.inchannels, stride, downsample))
        # self.inplanes = self.inchannels
        for i in range(1, blocks):
            layers.append(block(self.inchannels, self.inchannels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer2(x)

        return x



model56to28 = ResNetBlock(BasicBlock,4,inchannels = 256)
# inp56to28 = torch.randn(10,256,56,56)

model28to14 = ResNetBlock(BasicBlock,4,inchannels = 512)
# inp28to14 = torch.randn(10,512,28,28)

model14to7 = ResNetBlock(BasicBlock,4,inchannels = 768)
# inp14to7 = torch.randn(10,768,14,14)

# model7to1 = ResNetBlock(BasicBlock,4,inchannels = 1024)
# inp7to1 = torch.randn(10,1024,7,7)
#
# # with SummaryWriter(comment='model56to28') as w:
# #     w.add_graph(model56to28,(inp56to28,))
# # with SummaryWriter(comment='model28to14') as w:
# #     w.add_graph(model28to14, (inp28to14,))
# # with SummaryWriter(comment='model14to7') as w:
# #     w.add_graph(model14to7, (inp14to7,))
# # with SummaryWriter(comment='model7to1') as w:
# #     w.add_graph(model7to1, (inp7to1,))
#
# avgpooling7to1 = nn.AvgPool2d(7, stride=1)
# fc1024to101 = nn.Linear(1024, 101)

