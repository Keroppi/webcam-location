#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, argparse, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from simple_day import SimpleDay
from webcam_dataset import WebcamData
from webcam_dataset import Test
from custom_transforms import Resize, RandomPatch, Center, ToTensor
from custom_model import WebcamLocation
from torch.autograd import Variable

from collections import namedtuple

print('Starting save predictions.')
sys.stdout.flush()

#Location = namedtuple('Location', ['lat', 'lng', 'sunrises', 'sunsets', 'mali_solar_noons'])

parser = argparse.ArgumentParser(description='Save Sunrise / Sunset Predictions')
parser.add_argument('--sunrise_model', default='', type=str, metavar='PATH',
                    help='path to sunrise model (default: none)')
parser.add_argument('--sunset_model', default='', type=str, metavar='PATH',
                    help='path to sunset model args (default: none)')
args = parser.parse_args()

if args.sunrise_model:
    sunrise_dir = args.sunrise_model
else:
    print('No sunrise path given.')
    sys.exit(1)

if args.sunset_model:
    sunset_dir = args.sunset_model
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

# When testing, use uniform frames to increase likelihood of grabbing sunrise and sunset.
uniform_t0 = time.time()
for day in days:
    day.uniform_frames()
uniform_t1 = time.time()
print('Getting uniform frames time (min): {:.2f}'.format((uniform_t1 - uniform_t0) / 60))
sys.stdout.flush()

print('# Test Examples: {}'.format(len(test_loader.dataset)))
sys.stdout.flush()

# sunrise

passes = 0
locations = {}

sunrise_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    # Convert sunrise_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        day.change_frames_medium(sunrise_idx[d_idx, 0].data[0], scale_factor=4.5, pass_idx=0)

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

passes += 1
sunrise_predict_t1 = time.time()
print('Sunrise prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()


sunrise_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    # Convert sunrise_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        day.change_frames_medium(sunrise_idx[d_idx, 0].data[0], scale_factor=1.5, pass_idx=1)

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

passes += 1
sunrise_predict_t1 = time.time()
print('Sunrise prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

sunrise_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    # Convert sunrise_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        day.change_frames_fine(sunrise_idx[d_idx, 0].data[0])

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

passes += 1
sunrise_predict_t1 = time.time()
print('Sunrise prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()
sunrise_err_total = []

sunrise_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        local_sunrise = day.get_local_time(sunrise_idx[d_idx, 0].data[0]) # - datetime.timedelta(seconds=days[d_idx].time_offset)

        error_min = math.fabs(((day.sunrise - local_sunrise).total_seconds() / 60))
        sunrise_err_total.append(error_min)

        if day.place not in locations:
            #locations[day.place] = Location(day.lat, day.lng, [], [], [])
            locations[day.place] = []

        locations[day.place].append(SimpleDay(day.place, day.lat, day.lng, day.mali_solar_noon, day.time_offset, local_sunrise, None, day.sunrise_in_frames, day.sunset_in_frames, day.interval_min, day.season))
        #day.uniform_frames()  # Reset the frames to be random instead of having a bias towards where sunrise is.

        #locations[day.place].sunrises.append(utc_sunrise)
        #locations[day.place].mali_solar_noons.append(day.mali_solar_noon)

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

passes += 1
sunrise_predict_t1 = time.time()
print('Sunrise testing prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

print('Sunrise mean error (min): {}'.format(statistics.mean(sunrise_err_total)))
print('Sunrise median error (min): {}'.format(statistics.median(sunrise_err_total)))

for day in days:
    day.uniform_frames() # Reset the frames to be random instead of having a bias towards where sunrise is.

# sunset

sunset_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    # Convert sunset_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunset_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        day.change_frames_medium(sunset_idx[d_idx, 0].data[0], mode='sunset', scale_factor=4.5, pass_idx=0) # VLI mode is only for print statements

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

sunset_predict_t1 = time.time()
print('Sunset prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

sunset_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    # Convert sunset_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunset_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        day.change_frames_medium(sunset_idx[d_idx, 0].data[0], mode='sunset', scale_factor=1.5, pass_idx=1) # VLI mode is only for print statements

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

sunset_predict_t1 = time.time()
print('Sunset prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

sunset_predict_t0 = time.time()
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    # Convert sunset_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunset_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        day.change_frames_fine(sunset_idx[d_idx, 0].data[0], mode='sunset') # VLI mode is only for print statements

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

sunset_predict_t1 = time.time()
print('Sunset prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

location_idx = {}
sunset_err_total = []

sunset_predict_t0 = time.time()
for batch_idx, (input, target) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunset_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        local_sunset = day.get_local_time(sunset_idx[d_idx, 0].data[0]) #- datetime.timedelta(seconds=days[d_idx].time_offset)

        error_min = math.fabs(((day.sunset - local_sunset).total_seconds() / 60))
        sunset_err_total.append(error_min)

        #if day.place not in locations:
            #locations[day.place] = Location(day.lat, day.lng, [], [], [])
        #locations[day.place].sunsets.append(utc_sunset)

        if day.place not in location_idx:
            location_idx[day.place] = 0

        locations[day.place][location_idx[day.place]].sunset = local_sunset
        location_idx[day.place] += 1

    if batch_idx % constants.LOG_INTERVAL == 0:
        print('Batch Index: {}'.format(batch_idx))
        sys.stdout.flush()

sunset_predict_t1 = time.time()
print('Sunset testing prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

print('Sunset mean error (min): {}'.format(statistics.mean(sunset_err_total)))
print('Sunset median error (min): {}'.format(statistics.median(sunset_err_total)))

# Sort by time.

sorted_locations = {}
for place in locations:
    location = locations[place]
    #sorted_lists = list(zip(*sorted(zip(location.sunrises, location.sunsets, location.mali_solar_noons))))
    #sorted_locations[place] = Location(locations[place].lat, locations[place].lng, sorted_lists[0], sorted_lists[1], sorted_lists[2])

    locations[place].sort(key=lambda x: x.sunrise, reverse=False)

if not os.path.isdir(sunrise_dir + '/predictions/'):
    os.mkdir(sunrise_dir + '/predictions/')

with open(sunrise_dir + '/predictions/{}_pred.pkl'.format(passes), 'wb') as f:
    #pickle.dump(sorted_locations, f)
    pickle.dump(locations, f)
