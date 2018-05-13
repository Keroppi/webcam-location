#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, argparse, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Train, Test
from custom_transforms import RandomResize, Resize, RandomPatch, Center, ToTensor
from custom_model import WebcamLocation
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Save CNN Features')
parser.add_argument('--sunrise_path', default='', type=str, metavar='PATH',
                    help='path to sunrise model (default: none)')
parser.add_argument('--sunset_path', default='', type=str, metavar='PATH',
                    help='path to sunset model args (default: none)')
args = parser.parse_args()

if args.sunrise_path:
    sunrise_dir = args.sunrise_path
    sunrise_output_dir = args.sunrise_path + '/features/'
else:
    print('No sunrise path given.')
    sys.exit(1)

if args.sunset_path:
    sunset_dir = args.sunset_path
    sunset_output_dir = args.sunset_path + '/features/'
else:
    print('No sunset path given.')
    sys.exit(1)

sunrise_model = sunrise_dir + 'sunrise_model_best1.pth.tar'
sunset_model = sunset_dir + 'sunset_model_best2.pth.tar'
sunrise_pkl = sunrise_dir + 'sunrise_model_structure1.pkl'
sunset_pkl = sunset_dir + 'sunset_model_structure2.pkl'

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

train_dataset.set_mode('sunrise')
test_dataset.set_mode('sunrise')

if torch.cuda.is_available():
    pin_memory = True
else:
    print('WARNING - Not using GPU.')
    pin_memory = False

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)

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

    end = min(len(train_loader.dataset), (batch_idx + 1) * constants.BATCH_SIZE)
    train_sunrise_input[batch_idx * constants.BATCH_SIZE:end, :] = sunrise_features.cpu().data.numpy()
    train_sunrise_output[batch_idx * constants.BATCH_SIZE:end] = target.numpy()

sunrise_predict_t1 = time.time()
print('Sunrise training prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

# sunset

train_dataset.set_mode('sunset')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)
sys.stdout.flush()

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
    train_sunset_output[batch_idx * constants.BATCH_SIZE:end] = target.numpy()

sunset_predict_t1 = time.time()
print('Sunset training prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

# Pickle training features, output

with open(sunrise_output_dir + 'sunrise_train_input.pkl', 'wb') as f:
    pickle.dump(train_sunrise_input, f)
with open(sunrise_output_dir + 'sunrise_train_output.pkl', 'wb') as f:
    pickle.dump(train_sunrise_output, f)
with open(sunset_output_dir + 'sunset_train_input.pkl', 'wb') as f:
    pickle.dump(train_sunset_input, f)
with open(sunset_output_dir + 'sunset_train_output.pkl', 'wb') as f:
    pickle.dump(train_sunset_output, f)

del train_sunrise_input
del train_sunrise_output
del train_sunset_input
del train_sunset_output

# Test Set

# sunrise

test_sunrise_input = np.zeros((len(test_loader.dataset), sunrise_model.first_fc_layer_size))
test_sunrise_output = np.zeros((len(test_loader.dataset),))
sunrise_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_features = sunrise_model.forward_features(input)

    end = min(len(test_loader.dataset), (batch_idx + 1) * constants.BATCH_SIZE)
    test_sunrise_input[batch_idx * constants.BATCH_SIZE:end, :] = sunrise_features.cpu().data.numpy()
    test_sunrise_output[batch_idx * constants.BATCH_SIZE:end] = target.numpy()

sunrise_predict_t1 = time.time()
print('Sunrise testing prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

# sunset

test_dataset.set_mode('sunset')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)

test_sunset_input = np.zeros((len(test_loader.dataset), sunset_model.first_fc_layer_size))
test_sunset_output = np.zeros((len(test_loader.dataset),))
sunset_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_features = sunset_model.forward_features(input)
    end = min(len(test_loader.dataset), (batch_idx + 1) * constants.BATCH_SIZE)
    test_sunset_input[batch_idx * constants.BATCH_SIZE:end, :] = sunset_features.cpu().data.numpy()
    test_sunset_output[batch_idx * constants.BATCH_SIZE:end] = target.numpy()

sunset_predict_t1 = time.time()
print('Sunset testing prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

# Pickle testing features, output

with open(sunrise_output_dir + 'sunrise_test_input.pkl', 'wb') as f:
    pickle.dump(test_sunrise_input, f)
with open(sunrise_output_dir + 'sunrise_test_output.pkl', 'wb') as f:
    pickle.dump(test_sunrise_output, f)
with open(sunset_output_dir + 'sunset_test_input.pkl', 'wb') as f:
    pickle.dump(test_sunset_input, f)
with open(sunset_output_dir + 'sunset_test_output.pkl', 'wb') as f:
    pickle.dump(test_sunset_output, f)

del test_sunrise_input
del test_sunrise_output
del test_sunset_input
del test_sunset_output