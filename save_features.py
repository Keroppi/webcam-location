#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Train, Test
from custom_transforms import RandomResize, Resize, RandomPatch, Center, ToTensor
from custom_model import WebcamLocation
from torch.autograd import Variable

if constants.CLUSTER:
    directory = '/srv/glusterfs/vli/models/best/'
else:
    directory = '~/models/best/'
    directory = os.path.expanduser(directory)

sunrise_model = directory + 'sunrise_model_best1.pth.tar'
sunset_model = directory + 'sunset_model_best2.pth.tar'
sunrise_pkl = directory + 'sunrise_model_structure1.pkl'
sunset_pkl = directory + 'sunset_model_structure2.pkl'

sunrise_checkpt = torch.load(sunrise_model)
sunset_checkpt = torch.load(sunset_model)

with open(sunrise_pkl, 'rb') as sunrise_pkl_f:
    sunrise_model_args = pickle.load(sunrise_pkl_f)
    sunrise_model = WebcamLocation(*sunrise_model_args)

with open(sunset_pkl, 'rb') as sunset_pkl_f:
    sunset_model_args = pickle.load(sunset_pkl_f)
    sunset_model = WebcamLocation(*sunset_model_args)

sunrise_model.load_state_dict(sunrise_checkpt['state_dict'])
sunset_model.load_state_dict(sunset_checkpt['state_dict'])

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        sunrise_model = torch.nn.DataParallel(sunrise_model)
        sunset_model = torch.nn.DataParallel(sunset_model)

    sunrise_model.cuda()
    sunset_model.cuda()

sunrise_model.eval()
sunset_model.eval()

data = WebcamData()
days = data.days
if constants.CENTER: # Should I use RandomResize or Resize for Train? # VLI
    train_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), Center(), ToTensor()])
    test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), Center(), ToTensor()])
else:
    train_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])
    test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])
train_dataset = Train(data, train_transformations)
test_dataset = Test(data, test_transformations)

if torch.cuda.is_available():
    pin_memory = True
    num_workers = 0
else:
    print('WARNING - Not using GPU.')
    pin_memory = False
    num_workers = constants.NUM_LOADER_WORKERS

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

# Training Set

# sunrise

train_sunrise_input = np.zeros((len(train_loader.dataset), sunrise_model.first_fc_layer_size))
train_sunrise_output = np.zeros((len(train_loader.dataset),))
sunrise_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(train_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_features = sunrise_model.forward_features(input)

    #if batch_idx == 0:
    #    print(sunrise_features.size())
    #    print(target.size())
    #    sys.stdout.flush()

    end = min(len(train_loader.dataset), (batch_idx + 1) * constants.BATCH_SIZE)
    train_sunrise_input[batch_idx * constants.BATCH_SIZE:end, :] = sunrise_features.cpu().data.numpy()
    train_sunrise_output[batch_idx * constants.BATCH_SIZE:end] = target.data.numpy()

sunrise_predict_t1 = time.time()
print('Sunrise training prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

# sunset

train_sunset_input = np.zeros((len(train_loader.dataset), sunset_model.first_fc_layer_size))
train_sunset_output = np.zeros((len(train_loader.dataset),))
sunset_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(train_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_features = sunset_model.forward_features(input)
    end = min(len(train_loader.dataset), (batch_idx + 1) * constants.BATCH_SIZE)
    train_sunset_input[batch_idx * constants.BATCH_SIZE:end, :] = sunset_features.cpu().data.numpy()
    train_sunset_output[batch_idx * constants.BATCH_SIZE:end] = target.data.numpy()

sunset_predict_t1 = time.time()
print('Sunset training prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

# Pickle training features, output

if constants.CLUSTER:
    dir = '/srv/glusterfs/vli/features/'
else:
    dir = '/home/vli/features/'

with open(dir + str(constants.DAYS_PER_MONTH) + '_sunrise_train_input.pkl', 'wb') as f:
    pickle.dump(train_sunrise_input, f)
with open(dir + str(constants.DAYS_PER_MONTH) + '_sunrise_train_output.pkl', 'wb') as f:
    pickle.dump(train_sunrise_output, f)
with open(dir + str(constants.DAYS_PER_MONTH) + '_sunset_train_input.pkl', 'wb') as f:
    pickle.dump(train_sunset_input, f)
with open(dir + str(constants.DAYS_PER_MONTH) + '_sunset_train_output.pkl', 'wb') as f:
    pickle.dump(train_sunset_output, f)

del train_sunrise_input
del train_sunrise_output
del train_sunset_input
del train_sunset_output

# Test Set

# sunrise

# TO DO # VLI