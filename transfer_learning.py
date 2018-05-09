#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle
from sklearn.neighbors.kde import KernelDensity


sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Train
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
if constants.CENTER: # Should I use RandomResize or Resize? # VLI
    train_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), Center(), ToTensor()])
else:
    train_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])
train_dataset = Train(data, train_transformations)

if torch.cuda.is_available():
    pin_memory = True
    num_workers = 0
else:
    print('WARNING - Not using GPU.')
    pin_memory = False
    num_workers = constants.NUM_LOADER_WORKERS

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

sunrise_input = np.zeros((len(train_loader.dataset), sunrise_model.first_fc_layer_size))
sunrise_output = np.zeros((len(train_loader.dataset), 1))
sunrise_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(train_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_features = sunrise_model.forward_features(input) # Batches...



sunrise_predict_t1 = time.time()
print('Sunrise prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

sunset_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(train_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_features = sunset_model.forward_features(input)


sunset_predict_t1 = time.time()
print('Sunset prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

