# -*- coding: utf-8 -*-
import os
import logging
import argparse
import datetime

parse = argparse.ArgumentParser(description='Training ODENET')

model_args = parse.add_argument_group('Model')
model_args.add_argument('--model', default='sonet_s',
                        choices=['sonet_s','odenet_tq', 'resnet10', 'resXsonet10', 'wideresXsonet', 'wideresnet'],
                        type=str)
model_args.add_argument('--channel', default=64, type=int)
model_args.add_argument('--nb', default=[2, 2, 2, 2], help='number of blocks', type=int, nargs='+')
model_args.add_argument('--strides', default=[1, 2, 2, 2], help='strides for each block', type=int, nargs='+')
model_args.add_argument('--solver', default='dopri5', choices=['explicit_adams', 'fixed_adams', 'adams',
                                                               'tsit5', 'dopri5', 'bosh3', 'euler', 'midpoint', 'rk4','adaptive_heun'],
                        type=str)
model_args.add_argument('--adj', default=False, type=eval, choices=[True, False], help='adjoint for solver')
model_args.add_argument('--tol', default=0.1, type=float, help='tol for solver')

trainer_args = parse.add_argument_group('Trainer')
trainer_args.add_argument('--data', default='cifar10', choices=['mnist', 'cifar10', 'cifar100'], type=str)
trainer_args.add_argument('--nc', default=10, type=int)
trainer_args.add_argument('--aug', default=True, type=eval, choices=[True, False])
trainer_args.add_argument('--noise', default=0., type=float)
trainer_args.add_argument('--lr', default=0.01, type=float)
trainer_args.add_argument('--wd', default=0., type=float)
trainer_args.add_argument('--lr_d', default=[150, 300], help='learning decay by 0.1', type=int, nargs='+')
trainer_args.add_argument('--epochs', default=350, type=int)
trainer_args.add_argument('--batch_size', default=100, type=int)
trainer_args.add_argument('--loss', default='ce', choices=['ce', 'tr', 'ma'], type=str)
trainer_args.add_argument('--resume', default=False, type=eval, choices=[True, False])
trainer_args.add_argument('--resume_path', default=None)

attack_args = parse.add_argument_group('Attack')
attack_args.add_argument('--attack_type', default='linf', choices=['linf', 'l2'], type=str)
attack_args.add_argument('--num_steps', default=20, type=int)
attack_args.add_argument('--step_size', default=0.003, type=float)
attack_args.add_argument('--epsilon', default=8 / 255, type=float)

other_args = parse.add_argument_group('Other')
other_args.add_argument('--ckpt_dir', default='./checkpoints/')
other_args.add_argument('--save_freq', default=5, type=int)
other_args.add_argument('--use_gpu', default=True, type=eval, choices=[True, False])
other_args.add_argument('--gpuid', default='1,2')

config, unparsed = parse.parse_known_args()

if (config.resume and config.resume_path is not None):
    config.ckpt_dir = config.resume_path
else:
    config.ckpt_dir = os.path.join(config.ckpt_dir,
                                   config.model + '/' + datetime.datetime.now().strftime(
                                       '%y-%m-%d-%H-%M'))
if not os.path.exists(config.ckpt_dir):
    os.makedirs(config.ckpt_dir)
logger = logging.getLogger("ODENet-Experiments")
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(os.path.join(config.ckpt_dir, 'log.txt'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)
