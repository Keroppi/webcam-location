import torch, torchvision, os, datetime, time, math, pandas as pd
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Test
from custom_transforms import Resize, RandomPatch, ToTensor
from torch.autograd import Variable

if constants.CLUSTER:
    directory = '/srv/glusterfs/vli/models/best/'
else:
    directory = '~/models/best/'
    directory = os.path.expanduser(directory)

sunrise_model = directory + 'sunrise_model_best.pth.tar'
sunset_model = directory + 'sunset_model_best.pth.tar'

sunrise_checkpt = torch.load(sunrise_model)
sunset_checkpt = torch.load(sunset_model)

sunrise_model = sunrise_checkpt['model']
sunset_model = sunset_checkpt['model']

sunrise_model.eval()
sunset_model.eval()

data = WebcamData()
test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])
test_dataset = Test(data, test_transformations)

if torch.cuda.is_available():
    pin_memory = True
    num_workers = 1
else:
    print('WARNING - Not using GPU.')
    pin_memory = False
    num_workers = constants.NUM_LOADER_WORKERS

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

sunrises = []
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    # Convert sunrise_idx into a local time.

    # Assume that data is in the same order as the batch, since it's not shuffled.
    batch_days = data[batch_idx * constants.BATCH_SIZE:(batch_idx + 1) * constants.BATCH_SIZE]

    for d_idx, day in enumerate(batch_days):
        local_sunrise = day.get_local_time(sunrise_idx[d_idx, 0])
        #utc_sunrise = local_sunrise - datetime.timedelta(seconds=day.time_offset)
        sunrises.append(local_sunrise)

sunsets = []
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    # Convert sunset_idx into a local time.

    # Assume that data is in the same order as the batch, since it's not shuffled.
    batch_days = data[batch_idx * constants.BATCH_SIZE:(batch_idx + 1) * constants.BATCH_SIZE]

    for d_idx, day in enumerate(batch_days):
        local_sunset = day.get_local_time(sunrise_idx[d_idx, 0])
        #utc_sunset = local_sunset - datetime.timedelta(seconds=day.time_offset)
        sunsets.append(local_sunset)

# Compute solar noon and day length.
solar_noons = []
day_lengths = []
for sunrise, sunset in zip(sunrises, sunsets):
    solar_noon = (sunset - sunrise) / 2 + sunrise
    solar_noons.append(solar_noon)
    day_lengths.append((sunset - sunrise).total_seconds())


# Compute longitude.
longitudes = []
for d_idx, solar_noon in enumerate(solar_noons):
    utc_diff = data[d_idx].mali_solar_noon - solar_noon # Sun rises in the east and sets in the west.

    hours_time_zone_diff = data[d_idx].time_offset / 60 / 60
    hours_utc_diff = utc_diff.total_seconds() / 60 / 60
    longitudes.append((hours_utc_diff + hours_time_zone_diff) * 15)

# Compute latitude.
latitudes = []
for d_idx, day_length in enumerate(day_lengths):
    day_length_hours = day_length / 3600

    ts = pd.Series(pd.to_datetime([str(data[d_idx].date.date())]))
    day_of_year = int(ts.dt.dayofyear)

    declination = math.radians(23.45 * math.sin(360 * (283 + day_of_year) / 365)) # Brock model, day_of_year from 1 to 365, inclusive
    lat = math.degrees(math.atan(-math.cos(math.radians(15 * day_length_hours / 2)) / math.tan(declination)))

    latitudes.append(lat) # Only one day to predict latitude - could average across many days.

# Go through day objects - create a map where we can average all lat/longs of same places.

# Haversine formula for computing distance.
# https://www.movable-type.co.uk/scripts/latlong.html

# Make sure get_local_time works.