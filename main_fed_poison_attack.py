#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, ADP_FedAvg, ADP_Loss_FedAvg
from models.test import test_img, test_img_float

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dict_users = cifar_iid(dataset_train, args.num_users)
        # if args.iid:
        #     dict_users = cifar_iid(dataset_train, args.num_users)
        # else:
        #     exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list, acc_test_list = [], [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    list = [0.1]
    poison_list = [0.2]
    for attack_num in poison_list:
        for attack_strength in list:
            print('attack_num {:3f}, attack_strength {:.3f}'.format(attack_num, attack_strength))
            for iter in range(100):
                loss_locals = []
                score_locals = []
                if not args.all_clients:
                    w_locals = []
                m = max(int(0.3 * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                for idx in idxs_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss = local.ADP_noise_train(net=copy.deepcopy(net_glob).to(args.device))
                    if args.all_clients:
                        w_locals[idx] = copy.deepcopy(w)
                    else:
                        w_locals.append(copy.deepcopy(w))
                        loss_locals.append(loss)
                #poisoning attack
                m_poison = max(int(attack_num * args.num_users), 1)
                poison_users = np.random.choice(range(args.num_users), m_poison, replace=False)
                for idx in poison_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss = local.poison_attack_train(attack_strength,net=copy.deepcopy(net_glob).to(args.device))
                    if args.all_clients:
                        w_locals[idx] = copy.deepcopy(w)
                    else:
                        w_locals.append(copy.deepcopy(w))
                        loss_locals.append(loss)
                # 计算本地模型分数
                for loss in loss_locals:
                    score_locals.append(1 - loss/sum(loss_locals))

                # update global weights
                w_glob = ADP_FedAvg(w_locals,score_locals)
                #w_glob = FedAvg(w_locals)
                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)
                if(iter%5==0):
                    acc_test, loss_test = test_img_float(net_glob, dataset_test, args)
                    acc_test_list.append(acc_test)
                    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_test))
                    print("Testing accuracy: {:.2f}".format(acc_test))
            df = pd.DataFrame([acc_test_list])
            df.to_csv('./save/reputation_acc.csv', header=True, index=None,mode='a')  # mode='a' 追加

