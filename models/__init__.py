from .sonet_s import sonet as sonet_s
from .odenet_tq import odenet as odenet_tq
from .resnet import ResNet10
from .wide_resnet import WideResNet
from .wide_resXsonet import WideResNet as WideResXsoNet
from .resXsonet import ResNet10 as ResXsonet10

ModelSelector = {'odenet_tq': odenet_tq,
                 'sonet_s': sonet_s,
                 'resnet10': ResNet10,
                 'wideresnet': WideResNet,
                 'wideresXsonet': WideResXsoNet,
                 'resXsonet10': ResXsonet10}
