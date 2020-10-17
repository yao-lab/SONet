import time
import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def adjust_learning_rate(lr, optimizer, lr_decay, epoch):
    """decrease the learning rate"""

    if epoch >= lr_decay[0]:
        lr = lr * 0.1
    if epoch >= lr_decay[1]:
        lr = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class AveMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class Timer(object):
    def __init__(self):
        self.start = time.time()

    def reset(self, t):
        self.start = t

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self):
        now = time.time()
        s = now - self.start
        return '%s' % (self.asMinutes(s))


def get_loader(batch_size, aug=False, data='cifar10'):
    root = f'./data/{data}/'
    if data == 'cifar10' or 'cifar100':
        if aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor()
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        training_dataset = datasets.CIFAR10(root=root, train=True, download=True,
                                            transform=transform_train) if data == 'cifar10' else \
            datasets.CIFAR100(root=root, train=True, download=True,
                              transform=transform_train)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True,
                                        transform=transform_test) if data == 'cifar10' else \
            datasets.CIFAR100(root=root, train=False, download=True,
                              transform=transform_test)
    elif data == 'mnist':
        if aug:
            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                lambda x: x.repeat(3, 1, 1)
            ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            lambda x: x.repeat(3, 1, 1)
        ])
        training_dataset = datasets.MNIST(root=root, train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform_test)

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return {'train': training_dataset, 'test': test_dataset}, \
           {'train': training_loader, 'test': test_loader}
