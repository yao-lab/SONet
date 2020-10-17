# -*- coding: utf-8 -*-
import torch
from torch import nn
from models.misc import ODEBlock_tq, ODEfunc_tq, repeat_channel, norm


class BasicBlock(nn.Module):

    def __init__(self, in_planes, solver, adjoint, tol):
        super(BasicBlock, self).__init__()
        self.ode_conv = ODEBlock_tq(ODEfunc_tq(in_planes), solver, adjoint, tol)

    def forward(self, x):
        out = self.ode_conv(x)
        return out


class odeNet_arch(nn.Module):
    def __init__(self, block, channel, num_blocks, solver='dpori5', adjoint=False, tol=0.1, num_classes=10):
        super(odeNet_arch, self).__init__()
        self.in_planes = channel
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol
        self.num_blocks = num_blocks
        self.channel = repeat_channel(3, self.in_planes)
        self.conv = self._make_layer(block, self.in_planes)
        self.norm = norm(self.in_planes)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.linear = nn.Linear(self.in_planes * 9, num_classes)

    def _make_layer(self, block, planes):
        layers = []
        for _ in range(self.num_blocks):
            layers.append(block(planes, self.solver, self.adjoint, self.tol))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.channel(x)
        out = self.conv(out)
        out = self.norm(out)
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


def odenet(channel=54, num_blocks=10, strides=None, solver='dopri5', adjoint=False, tol=0.1, num_classes=10):
    return odeNet_arch(BasicBlock, channel=channel, num_blocks=10, solver=solver, adjoint=adjoint, tol=tol,
                       num_classes=num_classes)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    model = nn.DataParallel(odenet().cuda())
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    y = model(torch.randn(1, 3, 32, 32).cuda())
    print(model.module.nfe.item())
    model.module.nfe = 0
    print(model.module.nfe.item())
    print(y.size())
