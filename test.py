#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, sys, random
import torch.nn as nn
from torch.autograd import Variable
import os, sys, calendar, glob, datetime, time, functools, numpy as np, constants, PIL, hashlib, torch, subprocess, copy, torchvision, shutil, argparse
from torch.utils.data.dataset import Dataset

# This file is just for random testing. 

# Arbitrary input/output.
input = torch.randn(400, 3, 32, 128, 128)
output = torch.randn(400, 1)

class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv = [nn.Conv3d(3, [2, 3, 4, 3],
                     kernel_size=(1, 5, 5),
                     stride=1,
                     padding=0)]

        self.first_fc_layer_size = self.get_conv_output((3, 32, 128, 128))

        self.fc = [nn.Linear(self.first_fc_layer_size, 1)]
        self.network = nn.Sequential(*(self.conv + self.fc)) # Possibly change to module list!!!!

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_conv_output(self, shape):
        batch_size = 1  # Not important.
        input = Variable(torch.rand(batch_size, *shape), requires_grad=False)
        output_feat = self.forward_features(input)
        flattened_size = self.num_flat_features(output_feat)
        return flattened_size

    def forward_features(self, x):
        return self.conv[0](x)

    def forward(self, x):
        x = self.forward_features(x)

        # Flatten into vector.
        x = x.view(-1, self.first_fc_layer_size)
        return self.fc[0](x)

class TestDataset(Dataset):
    def __getitem__(self, index):
        return (input[index, :], output[index, :])

    def __len__(self):
        return 400

dataset = TestDataset()


print('Devices: {:.0f}'.format(torch.cuda.device_count()))
sys.stdout.flush()
if torch.cuda.is_available():
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, num_workers=0, pin_memory=True)


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

        print('Batch idx: ' + str(batch_idx))
        vmem = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        print('V-Memory Before: \n' + str(vmem.stdout).replace('\\n', '\n'))
        sys.stdout.flush()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        vmem = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        print('V-Memory After: \n' + str(vmem.stdout).replace('\\n', '\n'))
        sys.stdout.flush()

        optimizer.step()
        del loss

for epoch in range(1):
    train_epoch(epoch, model, loader, optimizer)










