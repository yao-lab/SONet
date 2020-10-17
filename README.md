
# Adversarial Robustness of Stabilized NeuralODEs Might be from Obfuscated Gradients


## Introduction
The codes for [Adversarial Robustness of Stabilized NeuralODEs Might be from Obfuscated Gradients](https://arxiv.org/abs/2009.13145) by Yifei Huang, Yaodong Yu, Hongyang Zhang, Yi Ma, Yuan Yao.

## Prerequisites
* Python 3.7.0
* Pytorch (1.3.0)
* torchdiffeq (0.0.1)
* CUDA

## Install
```bash
$ pip install -r requirements.txt
```

## Training
For sonet: 
```bash
$ python main.py --model sonet_s --channel 64 --nb 2 2 2 2 --aug True --lr 0.01 --b 100 --wd 0 --data cifar10 --gpuid 0,1
```
For resnet10 with trades loss:
```bash
$ python main.py --model resnet10 --channel 64 --nb 1 1 1 1 --aug True lr 0.01 --b 100 --wd 0 --loss tr --data cifar10 --gpuid 0
```
For resnet10 based soblock:
```bash
$ python main.py --model resXsonet --channel 64 --nb 1 1 1 1 --aug True lr 0.01 --b 100 --wd 0 --data cifar10 --gpuid 0
```
For wideresnet based soblock:
```bash
$ python main.py --model wideresXsonet --channel 64 --nb 1 1 1 1 --aug True lr 0.01 --b 100 --wd 0 --data cifar10 --gpuid 0
```
For simple odenet:
```bash
$ python main.py --model odenet_tq --channel 64 --nb 1 1 1 1 --aug True lr 0.01 --b 100 --wd 0 --data cifar10 --gpuid 0
```


#### Arguments:
* ```model```: different models, SONet, ResNet10 or SOBlock
* ```channel```: input channel
* ```nb```: number of blocks for each layer
* ```strides```: strides for each layer
* ```aug```: whether to apply augmentation
* ```loss```: CrossEntropy, TRADES

## Robustness evaluation
After training, all of the ckpt.pt as well config.json will be saved in ./checkpoints with folder name the datatime. So, for a specific ckpt.pt, you can run the following command to evaluate its white-box PGD robustness.
```bash
$ python attacker.py --c_p {your config path} --m_p {your model path} --l2 False --linf_ns 20 --linf_s 0.003 --linf_e 0.031 --b 100 --gpuid 0
```

#### Arguments
* ```c_p```: config path
* ```m_p```: trained model path with corresponding config
* ```l2```: whether to apply l2 attack or linf
* ```linf_ns (l2_ns)```: iterations for PGD Linf (l2)
* ```linf_s (l2_s)```: step size for PGD Linf (l2)
* ```linf_e (l2_e)```: epsilion or distance for PGD Linf (l2)

For the other attacks, you can run the following command. 
```bash
$ python attacker_spsa --c_p {your config path} --m_p {your model path} --type {attack type}
```

#### Arguments
* ```type```: spsa, cwlinf, cwl2, boundary or deepfool

## Training Logs
```
Run 
tensorboard --logdir=checkpoints 
to see the training log.
```

## Results
White-box PGD attacks
```
The distance is 0.031 and 0.5 for Linf and l2 respectively.

```

| Model                            | Channel | Under which attack | Natural accuracy | Robust accuracy (Linf) <br> (epsilon = 0.031) | Robust accuracy (l2) <br> (epsilon = 0.5) |
|----------------------------------|---------|--------------------|------------------|-----------------------------------------------|--------------------------------------|
| SONet                            | 32      | PGD20      | 88.08%           | 53.67%                  | 57.39%             |
| SOBlock                          | 32      | PGD20      | 90.28%           | 58.21%                  | 60.25%             |
| ResNet10-TRADES (1/lambda = 1.0) | 32      | PGD20      | 81.52%           | 35.26%                  | 57.07%             |
| ResNet10-TRADES (1/lambda = 6.0) | 32      | PGD20      | 73.69%           | 43.46%                  | 55.73%             |
| SONet                            | 64      | PGD20      | 89.36%           | 61.62%                  | 64.08%             |
| SOBlock                          | 64      | PGD20      | **91.57%**       | **62.35%**              | **64.70%**         |
| ResNet10-TRADES (1/lambda = 1.0) | 64      | PGD20      | 82.74%           | 37.64%                  | 58.97%             |
| ResNet10-TRADES (1/lambda = 6.0) | 64      | PGD20      | 76.29%           | 45.24%                  | 57.28%             |
| SONet                            | 32      | PGD1000    | 88.08%           | 19.62%                  | 31.75%             |
| SOBlock                          | 32      | PGD1000    | 90.28%           | 52.01%                  | 52.79%             |
| ResNet10-TRADES (1/lambda = 1.0) | 32      | PGD1000    | 81.52%           | 33.60%                  | 56.70%             |
| ResNet10-TRADES (1/lambda = 6.0) | 32      | PGD1000    | 73.69%           | 43.30%                  | 55.48%             |
| SONet                            | 64      | PGD1000    | 89.36%           | 24.25%                  | 39.79%             |
| SOBlock                          | 64      | PGD1000    | **91.57%**       | **55.43%**              | 57.37%             |
| ResNet10-TRADES (1/lambda = 1.0) | 64      | PGD1000    | 82.74%           | 35.78%                  | **58.73%**         |
| ResNet10-TRADES (1/lambda = 6.0) | 64      | PGD1000    | 76.29%           | 44.70%                  | 56.87%             |