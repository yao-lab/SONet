# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
from models.misc import ODEResnetBlock, ODEBlock_fc


class odeNet_arch(nn.Module):
    def __init__(self, block, num_blocks, strides, plane=32, solver='dpori5', adjoint=False, tol=1e-1, num_classes=10):
        super(odeNet_arch, self).__init__()
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol
        self.in_planes = plane
        self.in_planes_1 = plane
        self.conv1 = block(3, self.in_planes_1, stride=1, solver=solver, adjoint=adjoint, tol=tol)
        self.layer1 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:1]), num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:2]), num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:3]), num_blocks[2], stride=strides[2])
        self.layer4 = self._make_layer(block, self.in_planes_1 * np.prod(strides[:4]), num_blocks[3], stride=strides[3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = ODEBlock_fc(self.in_planes_1 * np.prod(strides[:4]) * block.expansion, solver, adjoint, tol,
                                  num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.solver, self.adjoint, self.tol))
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


def sonet(channel=64, num_blocks=[2, 2, 2, 2], strides=[1, 2, 2, 2], solver='dopri5', adjoint=False, tol=1e-1,
          num_classes=10):
    return odeNet_arch(ODEResnetBlock, num_blocks, strides, channel, solver, adjoint, tol, num_classes)


if __name__ == '__main__':
    model = sonet().cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    y = model(torch.randn(1, 3, 32, 32).cuda())
    print(model.nfe.item())
    model.nfe = 0
    print(model.nfe.item())
    print(y.size())
