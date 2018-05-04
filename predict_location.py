#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Test
from custom_transforms import Resize, RandomPatch, ToTensor
from custom_model import WebcamLocation
from torch.autograd import Variable

if constants.CLUSTER:
    directory = '/srv/glusterfs/vli/models/best/'
else:
    directory = '~/models/best/'
    directory = os.path.expanduser(directory)

constants.BATCH_SIZE = 250

sunrise_model = directory + 'sunrise_model_best1.pth.tar'
sunset_model = directory + 'sunset_model_best2.pth.tar'
sunrise_pkl = directory + 'sunrise_model_structure1.pkl'
sunset_pkl = directory + 'sunset_model_structure2.pkl'

sunrise_checkpt = torch.load(sunrise_model)
sunset_checkpt = torch.load(sunset_model)

#sunrise_model = sunrise_checkpt['model']
#sunset_model = sunset_checkpt['model']

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
test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), ToTensor()])
test_dataset = Test(data, test_transformations)

if torch.cuda.is_available():
    pin_memory = True
    num_workers = 0
else:
    print('WARNING - Not using GPU.')
    pin_memory = False
    num_workers = constants.NUM_LOADER_WORKERS

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

sunrise_predict_t0 = time.time()
sunrises = []
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunrise_idx = sunrise_model(input)

    # Convert sunrise_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        local_sunrise = day.get_local_time(sunrise_idx[d_idx, 0].data[0])
        #utc_sunrise = local_sunrise - datetime.timedelta(seconds=day.time_offset)
        sunrises.append(local_sunrise)
sunrise_predict_t1 = time.time()
print('Sunrise prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
sys.stdout.flush()

sunset_predict_t0 = time.time()
sunsets = []
for batch_idx, (input, _) in enumerate(test_loader):
    input = Variable(input, volatile=True)

    if torch.cuda.is_available():
        input = input.cuda()

    sunset_idx = sunset_model(input)

    # Convert sunset_idx into a local time.
    batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunset_idx.size()[0]]

    for d_idx, day in enumerate(batch_days):
        local_sunset = day.get_local_time(sunset_idx[d_idx, 0].data[0])
        #utc_sunset = local_sunset - datetime.timedelta(seconds=day.time_offset)
        sunsets.append(local_sunset)
sunset_predict_t1 = time.time()
print('Sunset prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
sys.stdout.flush()

# Compute solar noon and day length.
solar_noons = []
day_lengths = []
for d_idx, (sunrise, sunset) in enumerate(zip(sunrises, sunsets)):
    # Threshold sunrise to be at earliest midnight.
    if sunrise.date() < days[d_idx].date:
        sunrise = datetime.datetime.combine(sunrise, datetime.time.min)
    # Threshold sunset to be at latest 2 AM the next day.
    if sunset > datetime.datetime.combine(days[d_idx].date + datetime.timedelta(days=1), datetime.time(2, 0, 0)):
        sunset = datetime.datetime.combine(days[d_idx].date + datetime.timedelta(days=1), datetime.time(2, 0, 0))

    solar_noon = (sunset - sunrise) / 2 + sunrise

    # Latest solar noon in the world is in western China at 15:10, so truncate any time past ~15:14
    if solar_noon.hour > 15 or (solar_noon.hour == 15 and solar_noon.minute >= 14):
        solar_noon = solar_noon.replace(hour=15, minute=14, second=0, microsecond=0)
    # Earliest solar noon in the world is in Greenland around 10:04 AM, so truncate any time before ~10 AM.
    if solar_noon.hour < 10:
        solar_noon = solar_noon.replace(hour=10, minute=0, second=0, microsecond=0)

    '''
    if random.randint(1, 100) < 5: # VLI
        print('Sunrise / sunset / solar noon')
        print(sunrise)
        print(sunset)
        print(solar_noon)
        print('')
        sys.stdout.flush()
    '''

    solar_noons.append(solar_noon)
    day_lengths.append((sunset - sunrise).total_seconds())

# RANSAC for day lengths.

# Create tuple with (date, day length) for each place - get date from solar noons?
# Sort by date field - convert to an int from [0, 365) - March 1, 2017 onward.
# Fit a sin/cos with N solar noons (minutes change per day).

# y = A * sin(2 * pi / 365 * (input_day_integer + x_offset_in_days)) + y_offset - A, offset, y_offset are unknowns
# randomly take one point from each month?
# take the mean of points to get y_offset
# take the max or min value, subtract from estimated y_offset, take abs value to guess amplitude
# try all values of x_offset_in_days - [0, 365)? - could be very slow, maybe parallelize the locations...
# repeat many times...?
# reestimate with inliers
# https://ch.mathworks.com/matlabcentral/answers/178528-fitting-a-sinusoidal-curve-to-a-set-of-data-points
# Threshold ~20 minutes for inliers?
# May not work for locations near the equator... almost constant model.

# RANSAC for solar noons.
# Maybe not possible - not really linear...
# Very simple outlier detection - convert to UTC using offset from each day (as done in longitude section).
# Fit a line.
# Reject points which are more than... 60 minutes? away.
# Maybe not that useful.
# Could just threshold? Anything past ... 3:10 PM is probably wrong.
# https://astronomy.stackexchange.com/questions/18737/what-time-and-where-on-earth-is-the-latest-solar-noon
# Anything before... 10 AM?
# https://www.timeanddate.com/sun/greenland/daneborg - 10:04 AM solar noon
# http://www.dailymail.co.uk/sciencetech/article-2572317/Are-YOU-living-sync-Amazing-map-reveals-manmade-timezones-countries-false-sense-sun-rises.html

# Compute longitude.
longitudes = []
for d_idx, solar_noon in enumerate(solar_noons):
    utc_diff = days[d_idx].mali_solar_noon - solar_noon
    hours_time_zone_diff = days[d_idx].time_offset / 60 / 60
    hours_utc_diff = utc_diff.total_seconds() / 60 / 60
    lng = (hours_utc_diff + hours_time_zone_diff) * 15

    # Convert to UTC time first - doesn't make a difference compared to above.
    #utc_solar_noon = solar_noon - datetime.timedelta(seconds=days[d_idx].time_offset)
    #utc_diff = days[d_idx].mali_solar_noon - utc_solar_noon
    #hours_utc_diff = utc_diff.total_seconds() / 60 / 60
    #lng = hours_utc_diff * 15

    # What to do if outside [-180, 180] range?
    if lng < -180:
        lng += 360
        print('WARNING - lng below -180')
        sys.stdout.flush()
    elif lng > 180:
        lng -= 360
        print('WARNING - lng over 180')
        sys.stdout.flush()

    '''
    if random.randint(1, 100) < 5: # VLI
        print('Lng')
        print(lng)
        print('')
        sys.stdout.flush()
    '''

    longitudes.append(lng)

# Compute latitude.
latitudes = []
for d_idx, day_length in enumerate(day_lengths):
    day_length_hours = day_length / 3600

    ts = pd.Series(pd.to_datetime([str(days[d_idx].date)]))
    day_of_year = int(ts.dt.dayofyear) # Brock model, day_of_year from 1 to 365, inclusive

    declination = math.radians(23.45) * math.sin(math.radians(360 * (283 + day_of_year) / 365))
    lat = math.degrees(math.atan(-math.cos(math.radians(15 * day_length_hours / 2)) / math.tan(declination)))

    '''
    if random.randint(1, 100) < 5: # VLI
        print('Lat')
        print(lat)
        print('')
        sys.stdout.flush()
    '''

    latitudes.append(lat) # Only one day to predict latitude - could average across many days.

# Get mean and median of all lat/longs of same places.
#places = {}
lats = {}
lngs = {}
for i in range(data.types['test']):
    '''
    if places.get(days[i].place) is None:
        places[days[i].place] = [0, 0, 0] # lat, lng, number of data points to average, average

    places[days[i].place][0] += latitudes[i]
    places[days[i].place][1] += longitudes[i]
    places[days[i].place][2] += 1
    '''

    # Collect lat/lng based on location.
    if lats.get(days[i].place) is None:
        lats[days[i].place] = []
    lats[days[i].place].append(latitudes[i])

    if lngs.get(days[i].place) is None:
        lngs[days[i].place] = []
    lngs[days[i].place].append(longitudes[i])

'''
places_lat_lng = {}
for key in places:
    places_lat_lng[key] = (places[key][0] / places[key][2], places[key][1] / places[key][2])
'''

mean_locations = {}
median_locations = {}
for key in lats:
    mean_locations[key] = (statistics.mean(lats[key]), statistics.mean(lngs[key]))
    median_locations[key] = (statistics.median(lats[key]), statistics.median(lngs[key]))

# Kernel density estimation to guess location.
density_locations = {}
for key in lats:
    np_lats = np.array(lats[key])
    np_lngs = np.array(lngs[key])
    possible_points = np.vstack((np_lats, np_lngs))

    # Gaussian Kernel Density Estimation
    kernel = scipy.stats.gaussian_kde(possible_points)

    # Find MLE
    # Note, this uses around 5.2 GB memory.
    latitude_search = np.linspace(-90, 90, num=36001) # 0.005 step size
    longitude_search = np.linspace(-180, 180, num=36001) # 0.01 step size
    search_space = np.vstack((latitude_search, longitude_search))
    density = kernel(search_space)
    ind = np.unravel_index(np.argmax(density, axis=None), density.shape)
    density_locations[key] = (ind[0] * 0.005 - 90, ind[1] * 0.01 - 180)


def compute_distance(lat1, lng1, lat2, lng2): # kilometers
    # Haversine formula for computing distance.
    # https://www.movable-type.co.uk/scripts/latlong.html

    radius_of_earth = 6371  # km
    actual_lat_rad = math.radians(lat1)
    pred_lat_rad = math.radians(lat2)
    diff_lat_rad = actual_lat_rad - pred_lat_rad
    diff_lng_rad = math.radians(lng1 - lng2)
    temp = math.sin(diff_lat_rad / 2) ** 2 + \
           math.cos(pred_lat_rad) * math.cos(actual_lat_rad) * math.sin(diff_lng_rad / 2) ** 2
    temp1 = 2 * math.atan2(math.sqrt(temp), math.sqrt(1 - temp))
    distance = radius_of_earth * temp1

    return distance


finished_places = []
mean_distances = []
median_distances = []
density_distances = []
for i in range(data.types['test']):
    place = days[i].place

    # Go through each place.
    if place in finished_places:
        continue
    else:
        finished_places.append(place)

    actual_lat = days[i].lat
    actual_lng = days[i].lng

    mean_pred_lat = mean_locations[place][0]
    mean_pred_lng = mean_locations[place][1]
    median_pred_lat = median_locations[place][0] #places_lat_lng[place][0]
    median_pred_lng = median_locations[place][1] #places_lat_lng[place][1]
    density_pred_lat = density_locations[place][0]
    density_pred_lng = density_locations[place][1]

    mean_distance = compute_distance(actual_lat, actual_lng, mean_pred_lat, mean_pred_lng)
    mean_distances.append(mean_distance)
    median_distance = compute_distance(actual_lat, actual_lng, median_pred_lat, median_pred_lng)
    median_distances.append(median_distance)
    density_distance = compute_distance(actual_lat, actual_lng, density_pred_lat, density_pred_lng)
    density_distances.append(density_distance)

    if random.randint(1, 100) < 20: # VLI
        print('Distance')
        print(place)
        print('# Days Used: ' + str(len(lats[place])))
        print('Using mean: ' + str(mean_distance))
        print('Using median: ' + str(median_distance))
        print('Using density: ' + str(density_distance))
        print(str(actual_lat) + ', ' + str(actual_lng))
        print(str(mean_pred_lat) + ', ' + str(mean_pred_lng))
        print(str(median_pred_lat) + ', ' + str(median_pred_lng))
        print(str(density_pred_lat) + ', ' + str(density_pred_lng))
        print('')
        sys.stdout.flush()

    #print(distance)

#average_dist /= len(finished_places)
print('Means Avg. Distance Error: {:.6f}'.format(statistics.mean(mean_distances)))
print('Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(median_distances)))
print('Density Avg. Distance Error: {:.6f}'.format(statistics.mean(density_distances)))
print('Means Max Distance Error: {:.6f}'.format(max(mean_distances)))
print('Means Min Distance Error: {:.6f}'.format(min(mean_distances)))
print('Medians Max Distance Error: {:.6f}'.format(max(median_distances)))
print('Medians Min Distance Error: {:.6f}'.format(min(median_distances)))
print('Density Max Distance Error: {:.6f}'.format(max(density_distances)))
print('Density Min Distance Error: {:.6f}'.format(min(density_distances)))
sys.stdout.flush()