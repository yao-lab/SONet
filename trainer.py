import os
import re
import logging
import json
import torch
import numpy as np

np.random.seed(2333)
torch.manual_seed(2333)
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from models import ModelSelector
from attacker import Attacker
from utils import AveMeter, RunningAverageMeter, Timer, get_loader, adjust_learning_rate
from losses import trades_loss, madry_loss

logger = logging.getLogger('ODENet-Experiments')


class Trainer:
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.save_config(config)
        self.timer = Timer()

        self.lr = config.lr
        self.datasets, self.loaders = get_loader(config.batch_size, config.aug, data=config.data)
        self.epochs = config.epochs
        self.start_epoch = 0

        self.is_ode = 'resnet' not in self.config.model
        self.model = ModelSelector[config.model](**{'channel': config.channel,
                                                    'num_blocks': config.nb,
                                                    'strides': config.strides,
                                                    'solver': config.solver,
                                                    'adjoint': config.adj,
                                                    'tol': config.tol,
                                                    'num_classes': config.nc})
        if self.config.use_gpu:
            self.model = nn.DataParallel(self.model).cuda()

        self.attacker = Attacker(self.model, self.loaders['test'])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.wd)

        if config.resume:
            success = self.load(config.resume_path)
            if success:
                self.writer = SummaryWriter(log_dir=config.ckpt_dir, purge_step=self.start_epoch)
            else:
                self.writer = SummaryWriter(log_dir=config.ckpt_dir)
                logger.info(self.model)
                logger.info(
                    f'Number of Total Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        else:
            self.writer = SummaryWriter(log_dir=config.ckpt_dir)
            logger.info(self.model)
            logger.info(f'Number of Total Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def train_and_test(self):
        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"Epoch :[{epoch}/{self.epochs}]")
            self.train(epoch)
            test_acc = self.test(epoch)
            if epoch % 5 == 0:
                self.save({'net': self.model.state_dict(),
                           'test_acc': test_acc,
                           'epoch': epoch,
                           'optim': self.optimizer.state_dict()}, epoch)

        # self.writer.add_hparams(self.config.__dict__, {'hparam/test_best_acc': test_acc})
        self.writer.close()

    def train(self, epoch):
        lr = adjust_learning_rate(self.lr, self.optimizer, self.config.lr_d, epoch)
        self.model.train()
        losses = AveMeter()
        f_nfe_meter = AveMeter()
        b_nfe_meter = AveMeter()
        correct = 0
        for i, (inputs, targets) in enumerate(self.loaders['train']):
            inputs = inputs + self.config.noise * torch.randn_like(inputs)
            if self.config.use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()

            if self.is_ode:
                self.model.module.nfe = 0

            outputs = self.model(inputs)

            if self.is_ode:
                nfe_forward = self.model.module.nfe.item()

            if self.config.loss == 'ce':
                loss = self.criterion(outputs, targets)
            elif self.config.loss == 'tr':
                loss = self.criterion(outputs, targets) + trades_loss(self.model, inputs, self.optimizer,
                                                                      distance=self.config.attack_type)
            elif self.config.loss == 'ma':
                loss = madry_loss(self.model, inputs, targets, self.optimizer)

            if self.is_ode:
                self.model.module.nfe = 0
            loss.backward()
            self.optimizer.step()

            if self.is_ode:
                nfe_backward = self.model.module.nfe.item()
                self.model.module.nfe = 0

            if self.is_ode:
                f_nfe_meter.update(nfe_forward)
                b_nfe_meter.update(nfe_backward)
            losses.update(loss.item(), inputs.size()[0])
            correct += outputs.max(1)[1].eq(targets).sum().item()

        acc = 100 * correct / len(self.datasets['train'])
        logger.info(f"Train: [{i + 1}/{len(self.loaders['train'])}] | "
                    f"Time: {self.timer.timeSince()} | "
                    f"loss: {losses.avg:.4f} | "
                    f"acc: {acc:.2f}% | NFE-F: {f_nfe_meter.avg:.2f} | NFE-B: {b_nfe_meter.avg:.2f}")

        self.writer.add_scalar('train/lr', lr, epoch)
        self.writer.add_scalar('train/loss', losses.avg, epoch)
        self.writer.add_scalar('train/acc', acc, epoch)
        self.writer.add_scalar('train/nfe-f', f_nfe_meter.avg, epoch)
        self.writer.add_scalar('train/nfe-b', b_nfe_meter.avg, epoch)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                self.writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch)

    def test(self, epoch):
        self.model.eval()
        losses = AveMeter()
        correct = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.loaders['test']):
                if self.config.use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                losses.update(loss.item(), inputs.size()[0])
                correct += outputs.max(1)[1].eq(targets).sum().item()

            acc = 100 * correct / len(self.datasets['test'])

            out = f"Test: [{i + 1}/{len(self.loaders['test'])}] | " \
                  f"Time: {self.timer.timeSince()} | " \
                  f"loss: {losses.avg:.4f} | " \
                  f"acc: {acc:.2f}%"
            if epoch % self.config.save_freq == 0 and epoch != 0:
                acc_adv, grad_norm = self.attacker.attack(self.config.attack_type, self.config.num_steps,
                                                          self.config.step_size, self.config.epsilon)

                out = f"Test: [{i + 1}/{len(self.loaders['test'])}] | " \
                      f"Time: {self.timer.timeSince()} | " \
                      f"loss: {losses.avg:.4f} | " \
                      f"acc: {acc:.2f}% | " \
                      f"acc_adv_{self.config.attack_type}_{self.config.num_steps}: {acc_adv:.2f}%  " \
                      f"grad_norm: {grad_norm:.4f}"
                self.writer.add_scalar(f'test/acc_adv_{self.config.attack_type}_{self.config.num_steps}', acc_adv,
                                       epoch)
            logger.info(out)

            self.writer.add_scalar('test/loss', losses.avg, epoch)
            self.writer.add_scalar('test/acc', acc, epoch)

        return acc

    def save(self, state, epoch):
        torch.save(state, os.path.join(self.ckpt_dir, f'ckpt_{epoch}.pt'))
        logger.info('***Saving model***')

    def load(self, path):
        assert os.path.exists(path), f"resume {path} not exists!"
        ckpt_dict = {}
        for file in os.listdir(path):
            if file.startswith('ckpt_'):
                ckpt_dict[file] = int(re.findall('_([0-9]+).pt', file)[0])
        if len(ckpt_dict) == 0:
            logger.info('Do not find any checkpoint file, will train from start!')
            return False
        else:
            resume_file = sorted(ckpt_dict.items(), key=lambda x: x[1])[-1][0]
            state = torch.load(os.path.join(path, resume_file), map_location='cuda' if self.config.use_gpu else 'cpu')
            self.model.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optim'])
            self.start_epoch = state['epoch'] + 1
            logger.info('******************************************')
            logger.info(f'Successfully load ckpt from {resume_file}')
            return True

    def save_config(self, config):
        with open(os.path.join(self.ckpt_dir, 'config.json'), 'w+') as f:
            f.write(json.dumps(config.__dict__, indent=4))
        f.close()
