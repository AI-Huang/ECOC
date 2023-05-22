#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-22-20 20:46
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/pytorch/examples/blob/master/mnist/main.py

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from ecoc.encode import get_codebook_tensor, read_codebook
import pytorch.train_utils as train_utils
from pytorch.model_utils import build_model, build_scheduler


def cmd_args():
    # Training parameters
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model-name', type=str, default="lenet5")

    # Pretrained model's path
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--dataset-name', type=str, default="mnist")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--do-train', action='store_true', default=True)
    parser.add_argument('--do-eval', action='store_true', default=True)

    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    # Extended parameters for ECOC
    parser.add_argument('--codebook_name', type=str, default=None,
                        help='ECOC codebook name.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--no-norm', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=np.random.randint(10000), metavar='S',
                        help='random seed (default: np.random.randint(10000))')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    if args.codebook_name is not None:
        args.output_code = "ecoc"
    else:
        args.output_code = None

    return args


def main():
    args = cmd_args()
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_code:
        output_dir = os.path.join(
            "output", f"{args.model_name}_{args.output_code}_{args.dataset_name}_{args.optimizer}_{date_time}")
    else:
        output_dir = os.path.join(
            "output", f"{args.model_name}_{args.dataset_name}_{args.optimizer}_{date_time}")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    codebook_name = args.codebook_name
    if codebook_name:
        codebook_file = f"ecoc/codebooks/{codebook_name}.csv"
        codebook = read_codebook(codebook_file)
        codebook_tensor = get_codebook_tensor(codebook).to(device)
        args.num_classes, args.len_code = len(codebook), len(codebook[0])
    else:
        args.num_classes = 10

    # Log test results, write head row
    with open(os.path.join(output_dir, 'test.csv'), "a") as f:
        f.write(",".join(["epoch", "test_loss", "test_accuracy"])+'\n')

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    if args.no_norm:
        print("No normalization.")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        print("Normalize MNIST input.")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Record config
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf8') as json_file:
        json_file.write(json.dumps(vars(args)))

    mnist_train = datasets.MNIST(os.path.expanduser(
        "~/.datasets"), train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(os.path.expanduser(
        "~/.datasets"), train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(mnist_test, **kwargs)

    # Output dimension should be the length of ECOC code, not 10 for mnist anymore.
    # LeNet5
    if args.output_code == "ecoc":
        output_dim = args.len_code
    else:
        output_dim = 10
    model = build_model(args.model_name, args.dataset_name,
                        padding=2, output_dim=output_dim).to(device)
    # Load pretrained model
    if len(args.model_path) > 0:
        if os.path.exists(args.model_path) and args.model_path.endswith(".pt"):
            if not use_cuda:
                model.load_state_dict(
                    torch.load(args.model_path,
                               map_location=torch.device('cpu'))
                )
            else:
                model.load_state_dict(torch.load(args.model_path))
            print("Load pretrained model successfully!")
        else:
            print(f"{args.model_path} NOT exists!")

    print(f"Loss function: {args.loss}.")
    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss().to(device)
        criterion_test = nn.CrossEntropyLoss(reduction='sum').to(device)
    elif args.loss == "binary_cross_entropy":
        criterion = nn.BCELoss().to(device)  # Modification for ECOC
        criterion_test = nn.BCELoss(reduction='sum').to(device)
    elif args.loss == "l1":
        criterion = nn.L1Loss().to(device)  # Modification for ECOC
        criterion_test = nn.L1Loss(reduction='sum').to(device)

    # Optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=args.momentum)
    else:
        # TODO, other optimizers such as Adam
        pass

    # Learning rate and scheduler
    scheduler = build_scheduler(optimizer, args.model_name)

    # Trainer function
    if args.output_code == "ecoc":
        train = train_utils.train_ecoc
        test = train_utils.test_ecoc
        kwargs = {"codebook_tensor": codebook_tensor}
    else:
        train = train_utils.train
        test = train_utils.test
        kwargs = {}

    # Training body
    current_accuracy = 0.0
    for epoch in tqdm(range(args.epochs)):
        if args.do_train:
            training_logs = train(
                model, train_loader, criterion, optimizer, epoch, device, args, **kwargs)
            with open(os.path.join(output_dir, 'training_logs.txt'), "a") as f:
                for line in training_logs:
                    f.write(line + '\n')
        if args.do_eval:
            test_result = test(model, test_loader,
                               criterion_test, device, args, **kwargs)
            with open(os.path.join(output_dir, 'test.csv'), "a") as f:
                f.write(",".join(
                    [str(_) for _ in [
                        epoch, test_result["test_loss"], test_result["test_accuracy"]
                    ]]
                ) + '\n')
        new_accuracy = test_result["test_accuracy"]
        if args.save_model and new_accuracy > current_accuracy:
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f"{args.model_name}_e{epoch}_testacc{str(new_accuracy)}.pt"))
        current_accuracy = new_accuracy

        if scheduler:
            scheduler.step()


if __name__ == '__main__':
    main()
