#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : May-22-23 19:57
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import torch
import torch.nn.functional as F


def train(model, train_loader, criterion, optimizer, epoch, device, args, **kwargs):
    training_logs = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # criterion on [32, 10] and [32]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_log = f"Train Epoch: {epoch}. Batch [{batch_idx}/{len(train_loader)}].\tSample [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)].\tLoss: {loss.item():.6f}"
            print(training_log)
            training_logs.append(training_log)

    return training_logs


def test(model, test_loader, criterion, device, args, **kwargs):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            probs = F.softmax(output, dim=1)
            pred = probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)\n")

    return {"test_loss": test_loss, "test_accuracy": test_accuracy}


def train_ecoc(model, train_loader, criterion, optimizer, epoch, device, args, **kwargs):
    codebook_tensor = kwargs["codebook_tensor"]
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


def test_ecoc(model, test_loader, criterion, device, args, **kwargs):
    codebook_tensor = kwargs["codebook_tensor"]
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
