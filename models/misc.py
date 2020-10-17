# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint


def norm(dim):
    return nn.GroupNorm(min(8, dim), dim)


def swish(x):
    return x * torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc_tq(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_tq, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.register_buffer('nfe', torch.tensor(0.))

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock_tq(nn.Module):

    def __init__(self, odefunc_tq, solver, adjoint, tol):
        super(ODEBlock_tq, self).__init__()
        self.odefunc = odefunc_tq
        self.integration_time = torch.tensor([0, 1]).float()
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        odesolver = odeint_adjoint if self.adjoint else odeint
        out = odesolver(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol, method=self.solver)
        return out[1]


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.kernel = nn.parameter.Parameter(torch.Tensor(dim, dim, 3, 3))
        torch.nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        self.norm1 = norm(dim)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.norm2 = norm(dim)
        self.register_buffer('nfe', torch.tensor(0.))

    def forward(self, t, x):
        self.nfe += 1
        width_x = x.size(3)
        x = self.norm1(x)
        y = x[:, :, :width_x, :].clone()
        z = x[:, :, width_x:, :].clone()

        out_y = F.conv2d(z, self.kernel, stride=1, padding=1)
        out_y = self.leakyrelu(out_y)
        out_z = -1 * F.conv_transpose2d(y, self.kernel, stride=1, padding=1)
        out_z = self.leakyrelu(out_z)

        out = torch.cat((out_y, out_z), 2)
        out = self.norm2(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, solver, adjoint, tol):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol

    def forward(self, y):
        self.integration_time = self.integration_time.type_as(y)
        x = torch.cat((y, y), 2)
        odesolver = odeint_adjoint if self.adjoint else odeint
        ode_out = odesolver(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol, method=self.solver)
        width = ode_out.size(4)
        out = ode_out[:, :, :, :width, :].clone()
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODEfunc_fc(nn.Module):

    def __init__(self, dim, num_classes, bias_bool=True, gamma=0.):
        super(ODEfunc_fc, self).__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(dim, num_classes))
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        if bias_bool:
            self.bias = nn.parameter.Parameter(torch.Tensor(dim + num_classes))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer('nfe', torch.tensor(0.))
        self.width_x = dim
        self.bias_bool = bias_bool
        self.gamma = gamma

    def forward(self, t, x):
        self.nfe += 1
        y = x[:, :self.width_x].clone()
        z = x[:, self.width_x:].clone()

        if self.bias_bool:
            out_y = self.leakyrelu(z)
            out_y = F.linear(out_y, self.weight, self.bias[:self.width_x]) - self.gamma * y
            out_z = self.leakyrelu(y)
            out_z = F.linear(out_z, -1 * torch.t(self.weight), self.bias[self.width_x:]) - self.gamma * z
        else:
            out_y = F.linear(z, self.weight, self.bias) - self.gamma * y
            out_z = F.linear(y, -1 * torch.t(self.weight), self.bias) - self.gamma * z
        out = torch.cat((out_y, out_z), 1)
        return out


class ODEBlock_fc(nn.Module):

    def __init__(self, in_dim, solver, adjoint, tol, num_classes=10):
        super(ODEBlock_fc, self).__init__()
        self.odefunc = ODEfunc_fc(in_dim, num_classes)
        self.integration_time = torch.tensor([0, 1]).float()
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol
        self.num_classes = num_classes

    def forward(self, y):
        self.integration_time = self.integration_time.type_as(y)
        y_width = y.size(1)
        z = 0.1 * torch.ones(y.size(0), self.num_classes).to(y.device)
        # z = torch.zeros(y.size(0), self.num_classes).to(y.device)
        x = torch.cat((y, z), 1)
        odesolver = odeint_adjoint if self.adjoint else odeint
        ode_out = odesolver(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol, method=self.solver)
        out = ode_out[:, :, y_width:].clone()
        return out[1]


class repeat_channel(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(repeat_channel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, y):
        temp = y.repeat(1, self.dim_out // self.dim_in, 1, 1)
        divided = self.dim_out % self.dim_in
        if divided != 0:
            out = torch.cat((temp, y[:, :divided, :, :]), 1)
        else:
            out = temp
        return out


class ODEResnetFunc(nn.Module):
    def __init__(self, in_planes, planes, stride=1, bias_bool=False, gamma=0.):
        super(ODEResnetFunc, self).__init__()
        self.stride = stride
        if stride == 2:
            self.kernel = nn.parameter.Parameter(torch.Tensor(planes, in_planes, 4, 4))
        else:
            self.kernel = nn.parameter.Parameter(torch.Tensor(planes, in_planes, 3, 3))
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))

        if bias_bool:
            self.bias_1 = nn.parameter.Parameter(torch.Tensor(planes))
            self.bias_2 = nn.parameter.Parameter(torch.Tensor(in_planes))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias_1, -bound, bound)
            torch.nn.init.uniform_(self.bias_2, -bound, bound)
        else:
            self.register_parameter('bias_1', None)
            self.register_parameter('bias_2', None)

        self.register_buffer('nfe', torch.tensor(0.))
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.gamma = gamma

        self.norm1_1 = nn.GroupNorm(min(8, planes), planes)
        self.norm1_2 = nn.GroupNorm(min(8, in_planes), in_planes)
        self.norm2_1 = nn.GroupNorm(min(8, planes), planes)
        self.norm2_2 = nn.GroupNorm(min(8, in_planes), in_planes)

    def forward(self, t, x):
        self.nfe += 1
        y = self.norm1_1(x[0])
        z = self.norm1_2(x[1])

        out_y = F.conv2d(z, self.kernel, stride=self.stride, padding=1, bias=self.bias_1) - self.gamma * y
        out_y = self.leakyrelu(out_y)
        out_z = -F.conv_transpose2d(y, self.kernel, stride=self.stride, padding=1, bias=self.bias_2) - self.gamma * z
        out_z = self.leakyrelu(out_z)

        out_y = self.norm2_1(out_y)
        out_z = self.norm2_2(out_z)

        return out_y, out_z


class ODEResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, solver, adjoint, tol):
        super(ODEResnetBlock, self).__init__()
        self.planes = planes
        self.stride = stride
        # self.odefunc = nn.utils.spectral_norm(ODEResnetFunc(in_planes, planes, stride), name='kernel')
        self.odefunc = ODEResnetFunc(in_planes, planes, stride)
        self.integration_time = torch.tensor([0, 1]).float()
        self.solver = solver
        self.adjoint = adjoint
        self.tol = tol

    def forward(self, y):
        temp = self.extend(y, self.planes, self.stride)
        odesolver = odeint_adjoint if self.adjoint else odeint
        ode_out = odesolver(self.odefunc, (temp, y), self.integration_time, rtol=self.tol, atol=self.tol,
                            method=self.solver)
        return ode_out[0][1]

    def extend(self, y, dim_out, stride):
        assert stride in [1, 2], f'stride {stride} must be in [1, 2]'
        dim_in = y.size()[1]
        temp = y.clone()
        if stride == 2:
            temp = F.interpolate(temp, scale_factor=0.5)

        divided = dim_out % dim_in
        if dim_out >= dim_in:
            temp = temp.repeat(1, dim_out // dim_in, 1, 1)
            if divided != 0:
                temp = torch.cat((temp, temp[:, :divided, :, :]), dim=1)
        else:
            temp = temp[:, :divided, :, :]

        return temp
