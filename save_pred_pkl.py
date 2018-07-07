#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, argparse, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Test
from custom_transforms import Resize, RandomPatch, Center, ToTensor
from custom_model import WebcamLocation
from torch.autograd import Variable

from collections import namedtuple

print('Starting save predictions.')
sys.stdout.flush()

Location = namedtuple('Location', ['lat', 'lng', 'sunrises', 'sunsets', 'mali_solar_noons'])

parser = argparse.ArgumentParser(description='Save Sunrise / Sunset Predictions')
parser.add_argument('--sunrise_path', default='', type=str, metavar='PATH',
                    help='path to sunrise model (default: none)')
parser.add_argument('--sunset_path', default='', type=str, metavar='PATH',
                    help='path to sunset model args (default: none)')
args = parser.parse_args()

if args.sunrise_path:
    sunrise_dir = args.sunrise_path
else:
    print('No sunrise path given.')
    sys.exit(1)

if args.sunset_path:
    sunset_dir = args.sunset_path
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
if constants.CENTER:
    test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), Center(), ToTensor()])
else:
    test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])
test_dataset = Test(data, test_transformations)

#test_dataset.set_mode('sunrise')

if torch.cuda.is_available():
    pin_memory = True
else:
    print('WARNING - Not using GPU.')
    pin_memory = False

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)

# Test Set

# sunrise

locations = {}

sunrise_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        utc_sunrise = day.get_local_time(sunrise_idx[d_idx, 0].data[0]) - datetime.timedelta(seconds=days[d_idx].time_offset)

        if day.place not in locations:
            locations[day.place] = Location(day.lat, day.lng, [], [], [])
        locations[day.place].sunrises.append(utc_sunrise)
        locations[day.place].mali_solar_noons.append(day.mali_solar_noon)

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

sunrise_predict_t1 = time.time()
print('Sunrise testing prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

# sunset

#test_dataset.set_mode('sunset')
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=constants.NUM_LOADER_WORKERS, pin_memory=pin_memory)

sunset_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunset_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        utc_sunset = day.get_local_time(sunset_idx[d_idx, 0].data[0]) - datetime.timedelta(
            seconds=days[d_idx].time_offset)

        if day.place not in locations:
            locations[day.place] = Location(day.lat, day.lng, [], [], [])
        locations[day.place].sunsets.append(utc_sunset)

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

sunset_predict_t1 = time.time()
print('Sunset testing prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

# Sort by time.

sorted_locations = {}
for place in locations:
    location = locations[place]
    sorted_lists = list(zip(*sorted(zip(location.sunrises, location.sunsets, location.mali_solar_noons))))
    sorted_locations[place] = Location(locations[place].lat, locations[place].lng, sorted_lists[0], sorted_lists[1], sorted_lists[2])

with open('/srv/glusterfs/vli/pred.pkl', 'wb') as f:
    pickle.dump(sorted_locations, f)