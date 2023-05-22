#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:45
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
from pytorch.model_utils import build_model, build_scheduler


def cmd_args():
    # Training parameters
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model-name', type=str, default="lenet5")
    parser.add_argument('--output-code', type=str, default='')

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
    parser.add_argument('--codebook_name', type=str, default="hunqun_deng_c10_n5",
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

    return args


def train(model, train_loader, criterion, optimizer, epoch, device, codebook_tensor, args):
    training_logs = []

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Batch ECOC encoding
        target_code = torch.zeros(len(data), args.len_code).to(device)
        for i, _ in enumerate(target):
            target_code[i] += codebook_tensor[_]
        optimizer.zero_grad()
        output = model(data)
        output = torch.sigmoid(output)  # sigmoid activation
        loss = criterion(output, target_code)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_log = f"Train Epoch: {epoch}. Batch [{batch_idx}/{len(train_loader)}].\tSample [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)].\tLoss: {loss.item():.6f}"
            print(training_log)
            training_logs.append(training_log)

    return training_logs


def test(model, test_loader, criterion, codebook_tensor, device, args):
    model.eval()
    test_loss = 0
    all_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_code = torch.zeros(len(data), args.len_code).to(device)
            for i, _ in enumerate(target):
                target_code[i] += codebook_tensor[_]  # 0s and 1s
            output = torch.sigmoid(model(data))

            # Get the index of the max log-probability
            threshold = 0.5
            output_code = torch.as_tensor(
                (output - threshold) > 0, dtype=torch.int32).to(device)
            distances = torch.zeros(
                len(output_code), len(codebook_tensor)).to(device)
            for i, _out in enumerate(output_code):
                for j, _code in enumerate(codebook_tensor):
                    distances[i][j] = (_out-_code).abs().sum()
            pred = distances.argmin(dim=1, keepdim=True).to(device)
            # TODO, weak prediction whose d is too big
            correct = pred.eq(target.view_as(pred)).sum().item()
            all_correct += correct

            # Sum up all batch loss
            test_loss += criterion(output, target_code).item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * all_correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {all_correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)\n")

    return {"test_loss": test_loss, "test_accuracy": test_accuracy}


def main():
    args = cmd_args()
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        "output", f"{args.model_name}_{args.output_code}_{args.dataset_name}_{args.optimizer}_{date_time}")
    os.makedirs(output_dir, exist_ok=True)

    codebook_name = args.codebook_name
    codebook_file = f"ecoc/codebooks/{codebook_name}.csv"
    codebook = read_codebook(codebook_file)

    # Log test results, write head row
    with open(os.path.join(output_dir, 'test.csv'), "a") as f:
        f.write(",".join(["epoch", "test_loss", "test_accuracy"])+'\n')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    codebook_tensor = get_codebook_tensor(codebook).to(device)
    args.num_classes, args.len_code = len(codebook), len(codebook[0])
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
    model = build_model(args.model_name, args.dataset_name,
                        padding=2, output_dim=args.len_code).to(device)
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
        criterion_test = nn.CrossEntropyLoss(reduction='sum').to(args.device)
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

    # Training body
    current_accuracy = 0.0
    for epoch in tqdm(range(args.epochs)):
        if args.do_train:
            training_logs = train(
                model, train_loader, criterion, optimizer, epoch, device, codebook_tensor,  args)
            with open(os.path.join(output_dir, 'training_logs.txt'), "a") as f:
                for line in training_logs:
                    f.write(line + '\n')
        if args.do_eval:
            test_result = test(model, test_loader,
                               criterion_test, codebook_tensor, device, args)
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
