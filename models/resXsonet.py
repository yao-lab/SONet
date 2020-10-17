import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.misc import ODEResnetBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, strides=[1, 2, 2, 2], plane=64, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = plane
        self.in_planes_1 = plane

        self.conv1 = ODEResnetBlock(3, self.in_planes_1, stride=1, solver='dopri5', adjoint=False, tol=0.1)
        self.layer1 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:1]), num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:2]), num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:3]), num_blocks[2], stride=strides[2])
        self.layer4 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:4]), num_blocks[3], stride=strides[3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.in_planes_1 * np.prod(strides[:4]) * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.linear(out.flatten(1))
        return out

    @property
    def nfe(self):
        nfe = 0
        for name, module in self.named_modules():
            if 'odefunc' in name:
                if hasattr(module, 'nfe'):
                    nfe += module.nfe
        return nfe

    @nfe.setter
    def nfe(self, value):
        for name, module in self.named_modules():
            try:
                if hasattr(module, 'nfe'):
                    module.nfe.data = torch.tensor(value).to(module.nfe).clone().detach()
            except:
                pass


def ResNet10(channel=64, num_blocks=[2, 2, 2, 2], strides=[1, 2, 2, 2], num_classes=10, **kwargs):
    return ResNet(BasicBlock, num_blocks, strides, channel, num_classes)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet10(16, [1, 1, 1, 1])
    y = net(torch.randn(1, 3, 32, 32))
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(y.size())
