from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
from PIL import Image

import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import foolbox

from cleverhans.attacks import SPSA, ProjectedGradientDescent, CarliniWagnerL2
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
import warnings
from cw_attack import cw
from cw_li_attack import CarliniLi

warnings.filterwarnings("ignore")


def deep_fool_linf():
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    attack = foolbox.attacks.DeepFoolLinfinityAttack(fmodel, distance=foolbox.distances.Linf)

    correct = 0
    n = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy(), targets.cpu().numpy()
        adversarial = attack(inputs[0, :], targets[0], steps=100, subsample=10)
        if adversarial is None:
            adversarial = inputs.astype(np.float32)
        adversarial = inputs + np.clip(adversarial - inputs, -args.eps, args.eps)
        adversarial = np.clip(adversarial, 0., 1.)
        correct += (np.argmax(fmodel.batch_predictions(adversarial)) == targets).sum()
        n += len(targets)

        sys.stdout.write("\rWhite-box DeepFool Linf attack... Acc: %.3f%% (%d/%d)" %
                         (100. * correct / n, correct, n))
        sys.stdout.flush()

    print('White-box DeepFool Linf attack: %.3f%%' % (100. * correct / n))


def CW_attack_linf():
    correct = 0
    n = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        correct_0 = cw(model, inputs, targets, binary_search_steps=1, max_iterations=100, eps=8 / 255.)
        correct += correct_0
        n += len(targets)

        sys.stdout.write("\rWhite-box CW Linf attack... Acc: %.3f%% (%d/%d)" %
                         (100. * correct / n, correct, n))
        sys.stdout.flush()

    print('Accuracy under CW Linf attack: %.3f%%' % (100. * correct / n))


def CW_attack_l2():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    x_op = tf.placeholder(tf.float32, shape=(None, 3, 32, 32,))
    y_op = tf.placeholder(tf.float32, shape=(None, 10))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

    # Create an CW attack
    cw = CarliniWagnerL2(cleverhans_model, sess=sess)
    cw_params = {
        'binary_search_steps': 1,
        'max_iterations': 100,
        'batch_size': args.b,
        'clip_min': 0.,
        'clip_max': 1.,
        'y': y_op
    }

    adv_x_op = cw.generate(x_op, **cw_params)
    adv_preds_op = tf_model_fn(adv_x_op)

    # Evaluation against PGD attacks
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        adv = sess.run(adv_x_op, feed_dict={x_op: inputs, y_op: torch.nn.functional.one_hot(targets, 10)})
        diff = (torch.tensor(adv) - inputs).renorm(p=2, dim=0, maxnorm=0.5)
        adv = (inputs + diff).clamp(0., 1.)
        correct += model(adv).topk(1)[1][:, 0].eq(targets.cuda()).cpu().sum().item()
        total += len(inputs)

        sys.stdout.write(
            "\rWhite-box CW l2 attack... Acc: %.3f%% (%d/%d)" % (100. * correct / total, correct, total))
        sys.stdout.flush()

    print('Accuracy under CW l2 attack: %.3f%%' % (100. * correct / total))


def boundary_attack():
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    attack = foolbox.attacks.BoundaryAttack(model=fmodel)

    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy(), targets.cpu().numpy()
        adversarial = attack(inputs[0, :], targets[0], iterations=args.ns, log_every_n_steps=999999)
        if adversarial is None:
            adversarial = inputs.astype(np.float32)
        adversarial = inputs + np.clip(adversarial - inputs, -args.eps, args.eps)
        adversarial = np.clip(adversarial, 0., 1.)
        if np.argmax(fmodel.batch_predictions(adversarial)) == targets:
            correct += 1.

        sys.stdout.write("\rBlack-box Boundary attack... Acc: %.3f%% (%d/%d)" %
                         (100. * correct / (batch_idx + 1), correct, batch_idx + 1))
        sys.stdout.flush()

    print('Accuracy under Boundary attack: %.3f%%' % (100. * correct / batch_idx))


def pgd_attack():
    # Use tf for evaluation on adversarial data
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    x_op = tf.placeholder(tf.float32, shape=(None, 3, 32, 32,))
    y_op = tf.placeholder(tf.float32, shape=(None, 10))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

    # Create an PGD attack
    pgd = ProjectedGradientDescent(cleverhans_model, sess=sess)
    pgd_params = {
        'eps': args.eps,
        'eps_iter': args.ss,
        'nb_iter': args.ns,
        'clip_min': 0.,
        'clip_max': 1.,
        'y': y_op
    }

    adv_x_op = pgd.generate(x_op, **pgd_params)
    adv_preds_op = tf_model_fn(adv_x_op)

    # Evaluation against PGD attacks
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        adv_preds = sess.run(adv_preds_op, feed_dict={x_op: inputs, y_op: torch.nn.functional.one_hot(targets, 10)})
        correct += (np.argmax(adv_preds, axis=1) == targets.numpy()).sum()
        total += len(inputs)

        sys.stdout.write(
            "\rWhite-box PGD attack... Acc: %.3f%% (%d/%d)" % (100. * correct / total, correct, total))
        sys.stdout.flush()

    print('Accuracy under PGD attack: %.3f%%' % (100. * correct / total))


def spsa_attack():
    # Use tf for evaluation on adversarial data
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(model, out_dims=10)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
    cleverhans_model.nb_classes = 10

    # Create an SPSA attack
    spsa = SPSA(cleverhans_model, sess=sess)
    spsa_params = {
        'eps': args.eps,
        'nb_iter': args.ns,
        'clip_min': 0.,
        'clip_max': 1.,
        'spsa_samples': args.spsa_samples,  # in this case, the batch_size is equal to spsa_samples
        'spsa_iters': 1,
        'early_stop_loss_threshold': 0
    }

    # Evaluation against SPSA attacks
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        advs = spsa.generate_np(inputs.numpy(), y=targets.numpy().astype(np.int32), **spsa_params)
        with torch.no_grad():
            correct += (model(torch.tensor(advs).cuda()).topk(1)[1].cpu().eq(targets)).sum().item()
        total += len(inputs)

        sys.stdout.write(
            "\rBlack-box SPSA attack... Acc: %.3f%% (%d/%d)" % (100. * correct / total, correct, total))
        sys.stdout.flush()

    print('Accuracy under SPSA attack: %.3f%%' % (100. * correct / total))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Attacker')
    parse.add_argument('--c_p',
                       default='config.json',
                       type=str, help='config path')
    parse.add_argument('--m_p',
                       default='ckpt_345.pt',
                       type=str, help='model path')
    parse.add_argument('--type', default='spsa', type=str, choices=['cw', 'cw_l2', 'pgd', 'spsa', 'boundary', 'fool'])
    parse.add_argument('--ss', default=0.003, type=float, help='step size')
    parse.add_argument('--eps', default=8. / 255, type=float, help='epsilion')
    parse.add_argument('--ns', default=20, type=int, help='num_steps')
    parse.add_argument('--b', default=1, type=int, help='batch_size')
    parse.add_argument('--spsa_samples', default=32, type=int)
    parse.add_argument('--data', default='cifar10', choices=['cifar10', 'mnist'])
    parse.add_argument('--gpuid', default='1', type=str)

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
    model.eval()
    datasets, loaders = get_loader(batch_size=args.b, aug=False, data=args.data)
    test_loader = loaders['test']

    # score-based attack
    if args.type == 'spsa':
        spsa_attack()
    if args.type == 'cw':
        CW_attack_linf()
    if args.type == 'cw_l2':
        CW_attack_l2()
    if args.type == 'boundary':
        boundary_attack()
    elif args.type == 'pgd':
        pgd_attack()
    elif args.type == 'fool':
        deep_fool_linf()
