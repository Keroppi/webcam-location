import torch, sys, random
import torch.nn as nn
from torch.autograd import Variable
import os, sys, calendar, glob, datetime, time, functools, numpy as np, constants, PIL, hashlib, torch, subprocess, copy, torchvision, shutil, argparse
from torch.utils.data.dataset import Dataset

# SMALL TEST CASE TO CHECK IF DATAPARALLEL FAILS HERE TOO.

# Arbitrary input/output.
input = torch.randn(100, 50000000)
output = torch.randn(100, 1)

class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc = nn.Linear(50000000, 1)

    def forward(self, x):
        return self.fc(x)

class TestDataset(Dataset):
    def __getitem__(self, index):
        return (input[index, :], output[index, :])

    def __len__(self):
        return 100

dataset = TestDataset()

if torch.cuda.is_available():
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, num_workers=4, pin_memory=True)


loss_fn = torch.nn.MSELoss()
model = TestModule()
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model.cuda()
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)

def train_epoch(epoch, model, data_loader, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        if batch_idx == 0:
            vmem = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            print('V-Memory Before Train Forward: ' + str(epoch) + '\n' + str(vmem.stdout).replace('\\n', '\n'))
            sys.stdout.flush()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        if batch_idx == 0:
            vmem = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            print('V-Memory After Train Backward: ' + str(epoch) + '\n' + str(vmem.stdout).replace('\\n', '\n'))
            sys.stdout.flush()

        optimizer.step()

for epoch in range(1):
    train_epoch(epoch, model, loader, optimizer)










