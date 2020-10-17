# -*- coding: utf-8 -*-
import argparse
import torch
import copy
import torch.nn.functional as F
from torch import optim


class Attacker:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def attack(self, attack_type, num_steps, step_size, epsilon, show=False, num_batchs=1e8):
        self.attack_type = attack_type
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

        self.model.eval()
        total_correct = 0
        n = 0
        grad_norm = 0
        for i, (x, y) in enumerate(self.data_loader):
            x = x.cuda()
            if self.attack_type == "linf":
                x_adv, norm = self.attacker_linf(x, y)
            else:
                x_adv, norm = self.attacker_l2(x, y)
            with torch.no_grad():
                correct = self.model(x_adv).cpu().max(1)[1].eq(y).sum().item()
            total_correct += correct
            grad_norm += norm
            n += x.size()[0]
            if show:
                print(
                    f"batch id {i}: correct {correct}/{x.size()[0]}, {total_correct / n:.4f}, grad_norm: {grad_norm / n:.4f}")
            if i >= num_batchs - 1:
                return 100 * total_correct / n, grad_norm / n
            break
        return 100 * total_correct / len(self.data_loader.dataset), grad_norm / n

    def attacker_linf(self, x_nat, target):
        x_nat = x_nat.cuda()
        target = target.cuda()
        x = copy.deepcopy(x_nat).detach() + 0. * torch.randn_like(x_nat)
        norm = 0
        for i in range(self.num_steps):
            print(f"PGD {i} step")
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.model(x), target)
            grad = torch.autograd.grad(loss, [x])[0]
            norm += grad.norm(2, dim=(1, 2, 3)).sum().item()
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_nat - self.epsilon), x_nat + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return x, norm / self.num_steps

    def attacker_l2(self, x_nat, target):
        batch_size = len(x_nat)
        x_nat = x_nat.cuda()
        target = target.cuda()
        norm = 0
        x = copy.deepcopy(x_nat).detach()
        for _ in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.model(x), target)
            grad = torch.autograd.grad(loss, [x])[0]
            norm += grad.norm(2, dim=(1, 2, 3)).sum().item()
            for idx_batch in range(batch_size):
                grad_idx = grad[idx_batch].unsqueeze(0)
                grad_idx_norm = self.l2_norm(grad_idx)
                grad_idx /= (grad_idx_norm + 1e-8)
                x[idx_batch] = x[idx_batch].detach() + self.step_size * grad_idx
                eta_x_adv = (x[idx_batch] - x_nat[idx_batch]).unsqueeze(0)
                norm_eta = self.l2_norm(eta_x_adv)
                if norm_eta > self.epsilon:
                    eta_x_adv = eta_x_adv * self.epsilon / self.l2_norm(eta_x_adv)
                x[idx_batch] = x_nat[idx_batch] + eta_x_adv
            x = torch.clamp(x, 0.0, 1.0)
        return x, norm / self.num_steps

    def l2_norm(self, x):
        flattened = x.view(x.shape[0], -1)
        return (flattened ** 2).sum(1).sqrt()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Attacker')
    parse.add_argument('--c_p',
                       default='config.json',
                       type=str, help='config path')
    parse.add_argument('--m_p',
                       default='ckpt_345.pt',
                       type=str, help='model path')
    parse.add_argument('--type', default='linf', type=str)
    parse.add_argument('--linf_s', default=0.003, type=float, help='linf step size')
    parse.add_argument('--linf_e', default=8 / 255, type=float, help='linf epsilion')
    parse.add_argument('--linf_ns', default=1000, type=int, help='linf num_steps')
    parse.add_argument('--l2_s', default=0.1, type=float, help='l2 step size')
    parse.add_argument('--l2_e', default=0.5, type=float, help='l2 epsilion')
    parse.add_argument('--l2_ns', default=20, type=int, help='l2 num_steps')
    parse.add_argument('--data', default='cifar10', choices=['cifar10', 'mnist'])
    parse.add_argument('--b', default=1, type=int, help='batch_size')
    parse.add_argument('--gpuid', default='3', type=str)

    args, _ = parse.parse_known_args()

    import json
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    from models import ModelSelector
    from utils import get_loader

    config = json.load(open(args.c_p, 'rb'))
    model = ModelSelector[config['model']](**{'channel': config['channel'],
                                              'num_blocks': config['nb'],
                                              'strides': config['strides'],
                                              'solver': config['solver'],
                                              'adjoint': config['adj'],
                                              'tol': config['tol']})
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.m_p)['net'])
    datasets, loaders = get_loader(batch_size=args.b, aug=False, data=args.data)

    attacker = Attacker(model, loaders['test'])
    if args.type == 'l2':
        acc_adv_l2 = attacker.attack('l2', num_steps=args.l2_ns, step_size=args.l2_s,
                                     epsilon=args.l2_e, show=True)
    else:
        acc_adv_linf = attacker.attack('linf', num_steps=args.linf_ns, step_size=args.linf_s,
                                       epsilon=args.linf_e, show=True)
