#!/srv/glusterfs/vli/.pyenv/shims/python

import matplotlib
matplotlib.use('agg')

import torch, torchvision, os, argparse, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle
from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import minimize

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from webcam_dataset import WebcamData
from webcam_dataset import Test
from custom_transforms import Resize, RandomPatch, Center, ToTensor
from custom_model import WebcamLocation
from torch.autograd import Variable

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

print('Starting predict location.')
sys.stdout.flush()

temp = torch.randn(1).cuda()

parser = argparse.ArgumentParser(description='Predict Location')
parser.add_argument('--sunrise_model', default='', type=str, metavar='PATH',
                    help='path to sunrise model (default: none)')
parser.add_argument('--sunset_model', default='', type=str, metavar='PATH',
                    help='path to sunset model (default: none)')
parser.add_argument('--sunrise_pred', default='', type=str, metavar='PATH',
                    help='pickled numpy predictions for sunrise test data (default: none)')
parser.add_argument('--sunset_pred', default='', type=str, metavar='PATH',
                    help='pickled numpy predictions for sunset test data (default: none)')
args = parser.parse_args()

if args.sunrise_model and args.sunset_model:
    sunrise_directory = args.sunrise_model
    sunset_directory = args.sunset_model
    from_model = True
elif args.sunrise_pred and args.sunset_pred:
    sunrise_directory = args.sunrise_pred
    sunset_directory = args.sunset_pred
    from_model = False
else:
    print('Wrong arguments.')
    sys.stdout.flush()
    sys.exit(1)

data = WebcamData()
days = data.days

if from_model: # Use the trained model to generate predictions.
    sunrise_model = sunrise_directory + 'sunrise_model_best1.pth.tar'
    sunset_model = sunset_directory + 'sunset_model_best2.pth.tar'
    sunrise_pkl = sunrise_directory + 'sunrise_model_structure1.pkl'
    sunset_pkl = sunset_directory + 'sunset_model_structure2.pkl'

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

    if constants.CENTER:
        test_transformations = torchvision.transforms.Compose([Resize(), RandomPatch(constants.PATCH_SIZE), Center(), ToTensor()])
    else:
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

    print('# Test Examples: {}'.format(len(test_loader.dataset)))
    sys.stdout.flush()

    sunrise_predict_t0 = time.time()
    #sunrises = []
    for batch_idx, (input, _) in enumerate(test_loader):
        input = Variable(input, volatile=True)

        if torch.cuda.is_available():
            input = input.cuda()

        sunrise_idx = sunrise_model(input)

        # Convert sunrise_idx into a local time.
        batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

        for d_idx, day in enumerate(batch_days):
            day.change_frames(sunrise_idx[d_idx, 0].data[0])

        if batch_idx % constants.LOG_INTERVAL == 0:
            print('Batch Index: {}'.format(batch_idx))
            sys.stdout.flush()

    sunrise_predict_t1 = time.time()
    print('Sunrise prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
    sys.stdout.flush()

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
            day.random_frames() # Reset the frames to be random instead of having a bias towards where sunrise is.


        if batch_idx % constants.LOG_INTERVAL == 0:
            print('Batch Index: {}'.format(batch_idx))
            sys.stdout.flush()

    sunrise_predict_t1 = time.time()
    print('Sunrise prediction time (min): {:.2f}'.format((sunrise_predict_t1 - sunrise_predict_t0) / 60))
    sys.stdout.flush()

    sunset_predict_t0 = time.time()
    #sunsets = []
    for batch_idx, (input, _) in enumerate(test_loader):
        input = Variable(input, volatile=True)

        if torch.cuda.is_available():
            input = input.cuda()

        sunset_idx = sunset_model(input)

        # Convert sunset_idx into a local time.
        batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunset_idx.size()[0]]

        for d_idx, day in enumerate(batch_days):
            day.change_frames(sunset_idx[d_idx, 0].data[0], mode='sunset') # VLI mode is only for print statements

        if batch_idx % constants.LOG_INTERVAL == 0:
            print('Batch Index: {}'.format(batch_idx))
            sys.stdout.flush()

    sunset_predict_t1 = time.time()
    print('Sunset prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
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

        if batch_idx % constants.LOG_INTERVAL == 0:
            print('Batch Index: {}'.format(batch_idx))
            sys.stdout.flush()

    sunset_predict_t1 = time.time()
    print('Sunset prediction time (min): {:.2f}'.format((sunset_predict_t1 - sunset_predict_t0) / 60))
    sys.stdout.flush()

else: # Predictions already stored in a pickled numpy obj.
    sunrise_pred_pkl = sunrise_directory + 'sunrise_pred.pkl'
    sunset_pred_pkl = sunset_directory + 'sunset_pred.pkl'

    with open(sunrise_pred_pkl, 'rb') as sunrise_pkl_f:
        sunrise_pred = pickle.load(sunrise_pkl_f)

    with open(sunset_pred_pkl, 'rb') as sunset_pkl_f:
        sunset_pred = pickle.load(sunset_pkl_f)

    num_test_days = data.types['test']

    sunrises = []
    for d_idx, day in enumerate(days[:num_test_days]):
        local_sunrise = day.get_local_time(sunrise_pred[d_idx])
        sunrises.append(local_sunrise)

    sunsets = []
    for d_idx, day in enumerate(days[:num_test_days]):
        local_sunset = day.get_local_time(sunset_pred[d_idx])
        sunsets.append(local_sunset)

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

    if day_lengths[-1] < 0:
        print('WARNING - Negative day length!')
        sys.stdout.flush()

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
        print(days[d_idx].place)
        sys.stdout.flush()
    elif lng > 180:
        lng -= 360
        print('WARNING - lng over 180')
        print(days[d_idx].place)
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
cbm_latitudes = []
for d_idx, day_length in enumerate(day_lengths):
    day_length_hours = day_length / 3600

    ts = pd.Series(pd.to_datetime([str(days[d_idx].date)]))
    day_of_year = int(ts.dt.dayofyear) # day_of_year from 1 to 365, inclusive

    # Brock Model
    declination = math.radians(23.45) * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    lat = math.degrees(math.atan(-math.cos(math.radians(15 * day_length_hours / 2)) / math.tan(declination)))

    # CBM Model
    theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
    phi = math.asin(0.39795 * math.cos(theta))
    cbm_lat = 180 / math.pi * math.atan(math.cos(phi) / math.sin(phi) * math.cos(-math.pi / 24 * (day_length_hours - 24)))

    '''
    if random.randint(1, 100) < 5: # VLI
        print('Lat')
        print(lat)
        print('')
        sys.stdout.flush()
    '''

    latitudes.append(lat) # Only one day to predict latitude - could average across many days.
    cbm_latitudes.append(cbm_lat)  # Only one day to predict latitude - could average across many days.

# Get mean and median of all lat/longs of same places.
#places = {}
cbm_lats = {}
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

    if cbm_lats.get(days[i].place) is None:
        cbm_lats[days[i].place] = []
    cbm_lats[days[i].place].append(cbm_latitudes[i])

    if lngs.get(days[i].place) is None:
        lngs[days[i].place] = []
    lngs[days[i].place].append(longitudes[i])

# Collect intervals of each place.
intervals = {}
for i in range(data.types['test']):
    if intervals.get(days[i].place) is None:
        intervals[days[i].place] = []
    intervals[days[i].place].append(days[i].interval_min)

# Average intervals for each place.
for key in intervals:
    intervals[key] = statistics.mean(intervals[key])

# Collect breakdown of sunrise / sunset visible.
sun_visibles = {}
for i in range(data.types['test']):
    if sun_visibles.get(days[i].place) is None:
        sun_visibles[days[i].place] = [0, 0, 0, 0]

    if days[i].sunrise_in_frames and days[i].sunset_in_frames:
        sun_visibles[days[i].place][0] += 1
    elif days[i].sunrise_in_frames and not days[i].sunset_in_frames:
        sun_visibles[days[i].place][1] += 1
    elif not days[i].sunrise_in_frames and days[i].sunset_in_frames:
        sun_visibles[days[i].place][2] += 1
    else:
        sun_visibles[days[i].place][3] += 1

# Turn into percentages for each place.
for key in sun_visibles:
    normalized = [x / sum(sun_visibles[key]) for x in sun_visibles[key]]
    sun_visibles[key] = normalized

'''
places_lat_lng = {}
for key in places:
    places_lat_lng[key] = (places[key][0] / places[key][2], places[key][1] / places[key][2])
'''

mean_locations = {}
median_locations = {}

cbm_mean_locations = {}
cbm_median_locations = {}

for key in lats:
    mean_locations[key] = (statistics.mean(lats[key]), statistics.mean(lngs[key]))
    median_locations[key] = (statistics.median(lats[key]), statistics.median(lngs[key]))

    cbm_mean_locations[key] = (statistics.mean(cbm_lats[key]), statistics.mean(lngs[key]))
    cbm_median_locations[key] = (statistics.median(cbm_lats[key]), statistics.median(lngs[key]))

def kde_func_to_minimize(x, kernel):
    x = x.reshape(1, 2)
    density = kernel.score_samples(x)
    return -density[0] # Trying to minimize

# Kernel density estimation to guess location.
def kde(lats, lngs, median_locations):
    kernel_t0 = time.time()
    density_locations = {}
    for key in lats:
        if len(lats[key]) == 1:
            density_locations[key] = (lats[key][0], lngs[key][0])
            continue
        elif len(lats[key]) == 2: # Points are colinear, results in singular matrix
            density_locations[key] = (statistics.mean(lats[key]), statistics.mean(lngs[key]))
            continue

        np_lats = np.array([math.radians(x) for x in lats[key]])
        np_lngs = np.array([math.radians(x) for x in lngs[key]])
        possible_points = np.vstack((np_lats, np_lngs))

        kernel = KernelDensity(kernel='gaussian', bandwidth=constants.BANDWIDTH, metric='haversine').fit(possible_points.T)

        min_lat = min(np_lats)
        max_lat = max(np_lats)
        min_lng = min(np_lngs)
        max_lng = max(np_lngs)

        bnds = ((min_lat, max_lat), (min_lng, max_lng))
        res = minimize(kde_func_to_minimize, np.asarray(median_locations[key]), args=(kernel,), method='BFGS')

        if res.success:
            density_locations[key] = (res.x[0], res.x[1])
        else:
            #print('WARNING - scipy minimize function failed on location ' + key)
            #sys.stdout.flush()

            # Grid search for maximum density.

            # density_locations[key] = median_locations[key] # Use median if it fails.
            best_score = -float('inf')
            best_longitude = -181
            best_latitude = -91

            latitude_search = np.linspace(min_lat, max_lat,
                                          num=4001)  # Worst case pi/4000 radians (0.045 degrees) step size.
            longitude_search = np.linspace(min_lng, max_lng, num=8001)  # Worst case pi/4000 radians step size.

            for i in range(latitude_search.shape[0]):
                curr_lat = np.array([latitude_search[i]] * longitude_search.shape[0])
                search_space = np.vstack((curr_lat, longitude_search))

                density = kernel.score_samples(search_space.T)

                ind = np.argmax(density, axis=None)

                if best_score < density[ind]:
                    best_score = density[ind]
                    best_longitude = math.degrees(longitude_search[ind])
                    best_latitude = math.degrees(latitude_search[i])

                del curr_lat
                del search_space
                del density

            del latitude_search
            del longitude_search

            density_locations[key] = (best_latitude, best_longitude)

    kernel_t1 = time.time()
    print('Calculating density time (m): ' + str((kernel_t1 - kernel_t0) / 60))
    sys.stdout.flush()

    return density_locations

density_locations = kde(lats, lngs, median_locations)
cbm_density_locations = kde(cbm_lats, lngs, cbm_median_locations)

'''
kernel_t0 = time.time()
for key in lats:
    if len(lats[key]) == 1:
        density_locations[key] = (lats[key][0], lngs[key][0])
        continue
    elif len(lats[key]) == 2: # Points are colinear, results in singular matrix
        density_locations[key] = (statistics.mean(lats[key]), statistics.mean(lngs[key]))
        continue

    np_lats = np.array([math.radians(x) for x in lats[key]])
    np_lngs = np.array([math.radians(x) for x in lngs[key]])
    possible_points = np.vstack((np_lats, np_lngs))

    #finite = np.where(np.isfinite(possible_points) == False)[0].shape == (0,) # Check all values are not inf or NaN

    #if finite:

    kernel = KernelDensity(kernel='gaussian', bandwidth=constants.BANDWIDTH, metric='haversine').fit(possible_points.T)
    #kernel = scipy.stats.gaussian_kde(possible_points, bw_method=None)

    best_score = -float('inf')
    best_longitude = -181
    best_latitude = -91

    min_lat = min(np_lats)
    max_lat = max(np_lats)
    min_lng = min(np_lngs)
    max_lng = max(np_lngs)

    latitude_search = np.linspace(min_lat, max_lat, num=1001) # Worst case pi/1000 radians (0.18 degrees) step size.
    longitude_search = np.linspace(min_lng, max_lng, num=2001) # Worst case pi/1000 radians step size.
    #lat_v, lng_v = np.meshgrid(latitude_search, longitude_search)
    for i in range(latitude_search.shape[0]):
        curr_lat = np.array([latitude_search[i]] * longitude_search.shape[0])
        search_space = np.vstack((curr_lat, longitude_search))

        density = kernel.score_samples(search_space.T)
        #density = kernel(search_space)

        ind = np.argmax(density, axis=None)

        if best_score < density[ind]:
            best_score = density[ind]
            best_longitude = math.degrees(longitude_search[ind])
            best_latitude = math.degrees(latitude_search[i])

        del curr_lat
        del search_space
        del density

    del latitude_search
    del longitude_search

    density_locations[key] = (best_latitude, best_longitude)
    #else:
    #    density_locations[key] = None
    #    print('WARNING - NaN or inf found at ' + key)
    #    print(possible_points)
    #    lats_valid = [math.isnan(lat) or math.isinf(lat) for lat in latitudes]
    #    lngs_valid = [math.isnan(lng) or math.isinf(lng) for lng in longitudes]
    #    print(False in lats_valid)
    #    print(False in lngs_valid)
    #    sys.stdout.flush()
kernel_t1 = time.time()
print('Calculating density time (h): ' + str((kernel_t1 - kernel_t0) / 3600))
sys.stdout.flush()
'''

actual_locations = {}
for i in range(data.types['test']):
    if actual_locations.get(days[i].place) is None:
        actual_locations[days[i].place] = (days[i].lat, days[i].lng)
    else:
        continue

def plot_map(lats, lngs, mean_locations, median_locations, density_locations, mode='sun'):
    # Plot locations on a map.
    for place in lats:
        if len(lats[place]) < 50: # Need at least 50 points.
            continue

        min_lat = max(min(lats[place]) - 1, -90)
        max_lat = min(max(lats[place]) + 1, 90)
        min_lng = max(min(lngs[place]) - 1, -180)
        max_lng = min(max(lngs[place]) + 1, 180)

        colors = []

        #actual_lng = float('inf')
        #actual_lat = float('inf')
        actual_lat = actual_locations[place][0]
        actual_lng = actual_locations[place][1]

        for i in range(data.types['test']):
            if days[i].place != place:
                continue

            #actual_lng = days[i].lng
            #actual_lat = days[i].lat

            if mode == 'sun':
                if days[i].sunrise_in_frames and days[i].sunset_in_frames:
                    colors.append('g')
                elif not days[i].sunrise_in_frames and days[i].sunset_in_frames:
                    colors.append('r')
                elif not days[i].sunset_in_frames and days[i].sunrise_in_frames:
                    colors.append(mcolors.CSS4_COLORS['crimson'])
                else: # not days[i].sunrise_in_frames and not days[i].sunset_in_frames:
                    colors.append('k')
            elif mode == 'season':
                if days[i].season == 'winter':
                    colors.append('b')
                elif days[i].season == 'spring':
                    colors.append('y')
                elif days[i].season == 'summer':
                    colors.append('r')
                else:
                    colors.append(mcolors.CSS4_COLORS['tan'])

        plt.figure(figsize=(24,12))
        map = Basemap(projection='cyl', # This projection is equidistant.
                      llcrnrlat=min_lat, urcrnrlat=max_lat,
                      llcrnrlon=min_lng, urcrnrlon=max_lng,
                      resolution='h')
        #map.drawcoastlines()
        map.fillcontinents(color='coral',lake_color='aqua')
        map.drawmapboundary(fill_color='aqua')

        #x,y = map(lngs[place], lats[place])
        #x_actual,y_actual = map([actual_lng], [actual_lat])
        #x_mean,y_mean = map([mean_locations[place][1]], [mean_locations[place][0]])
        #x_median,y_median = map([median_locations[place][1]], [median_locations[place][0]])
        #x_density,y_density = map([density_locations[place][1]], [density_locations[place][0]])

        actual_and_pred_lngs = [actual_lng] + [mean_locations[place][1]] + [median_locations[place][1]] + [density_locations[place][1]]
        actual_and_pred_lats = [actual_lat] + [mean_locations[place][0]] + [median_locations[place][0]] + [density_locations[place][0]]
        actual_and_pred_colors = ['w', 'm', 'c', mcolors.CSS4_COLORS['fuchsia']]

        guesses = map.scatter(lngs[place], lats[place], s=40, c=colors, latlon=True, zorder=10)
        actual_and_pred = map.scatter(actual_and_pred_lngs, actual_and_pred_lats, s=40, c=actual_and_pred_colors, latlon=True, zorder=10, marker='^')

        #plt.legend(handles=[guesses, actual, mean_guess, median_guess, density_guess])

        if mode == 'sun':
            guess_colors = ['g', 'r', mcolors.CSS4_COLORS['crimson'], 'k']
            legend_labels = ['sunrise and sunset in frames', 'sunrise not in frames', 'sunset not in frames', 'sunrise and sunset not in frames', 'actual location', 'mean', 'median', 'gaussian kde']

            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                         [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]
        elif mode == 'season':
            guess_colors = ['b', 'y', 'r', mcolors.CSS4_COLORS['tan']]
            legend_labels = ['winter', 'spring', 'summer', 'fall'
                             'actual location', 'mean', 'median', 'gaussian kde']
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                         [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]

        plt.legend(handlelist, legend_labels)

        plt.title(place)
        plt.savefig('/srv/glusterfs/vli/maps/' + mode + '/' + place + '.png')
        plt.close()

plot_map(lats, lngs, mean_locations, median_locations, density_locations, 'sun')
plot_map(cbm_lats, lngs, cbm_mean_locations, cbm_median_locations, cbm_density_locations, 'season')

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

#finished_places = []
days_used = []
mean_distances = []
median_distances = []
density_distances = []
mean_longitude_err = []
mean_latitude_err = []
median_longitude_err = []
median_latitude_err = []
density_longitude_err = []
density_latitude_err = []

cbm_mean_distances = []
cbm_median_distances = []
cbm_density_distances = []
cbm_mean_latitude_err = []
cbm_median_latitude_err = []
cbm_density_latitude_err = []

#for i in range(data.types['test']):
for place in lats:
    #place = days[i].place

    # Go through each place.
    #if place in finished_places:
    #    continue
    #else:
    #    finished_places.append(place)

    days_used.append(len(lats[place]))

    actual_lat = actual_locations[place][0] #days[i].lat
    actual_lng = actual_locations[place][1] #days[i].lng

    mean_pred_lat = mean_locations[place][0]
    mean_pred_lng = mean_locations[place][1]
    median_pred_lat = median_locations[place][0]
    median_pred_lng = median_locations[place][1]
    density_pred_lat = density_locations[place][0]
    density_pred_lng = density_locations[place][1]

    cbm_mean_pred_lat = cbm_mean_locations[place][0]
    cbm_mean_pred_lng = cbm_mean_locations[place][1]
    cbm_median_pred_lat = cbm_median_locations[place][0]
    cbm_median_pred_lng = cbm_median_locations[place][1]
    cbm_density_pred_lat = cbm_density_locations[place][0]
    cbm_density_pred_lng = cbm_density_locations[place][1]

    mean_distance = compute_distance(actual_lat, actual_lng, mean_pred_lat, mean_pred_lng)
    mean_distances.append(mean_distance)
    median_distance = compute_distance(actual_lat, actual_lng, median_pred_lat, median_pred_lng)
    median_distances.append(median_distance)
    density_distance = compute_distance(actual_lat, actual_lng, density_pred_lat, density_pred_lng)
    density_distances.append(density_distance)

    cbm_mean_distance = compute_distance(actual_lat, actual_lng, cbm_mean_pred_lat, cbm_mean_pred_lng)
    cbm_mean_distances.append(cbm_mean_distance)
    cbm_median_distance = compute_distance(actual_lat, actual_lng, cbm_median_pred_lat, cbm_median_pred_lng)
    cbm_median_distances.append(cbm_median_distance)
    cbm_density_distance = compute_distance(actual_lat, actual_lng, cbm_density_pred_lat, cbm_density_pred_lng)
    cbm_density_distances.append(cbm_density_distance)

    mean_latitude_err.append(compute_distance(actual_lat, actual_lng, mean_pred_lat, actual_lng))
    mean_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, mean_pred_lng))
    median_latitude_err.append(compute_distance(actual_lat, actual_lng, median_pred_lat, actual_lng))
    median_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, median_pred_lng))
    density_latitude_err.append(compute_distance(actual_lat, actual_lng, density_pred_lat, actual_lng))
    density_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, density_pred_lng))

    cbm_mean_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_mean_pred_lat, actual_lng))
    cbm_median_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_median_pred_lat, actual_lng))
    cbm_density_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_density_pred_lat, actual_lng))

    if random.randint(1, 100) < 101: # VLI
        if median_distance < 25 or density_distance < 25:
            print('Under 25km!')

        print(place)
        print('# Days Used: ' + str(days_used[-1]))
        print('Brock Distance')
        print('Using mean: ' + str(mean_distance))
        print('Using median: ' + str(median_distance))
        print('Using density: ' + str(density_distance))
        print('CBM Distance')
        print('Using mean: ' + str(cbm_mean_distance))
        print('Using median: ' + str(cbm_median_distance))
        print('Using density: ' + str(cbm_density_distance))
        print('Actual lat, lng: ' + str(actual_lat) + ', ' + str(actual_lng))
        print('Mean lat, lng: ' + str(mean_pred_lat) + ', ' + str(mean_pred_lng))
        print('Median lat, lng: ' + str(median_pred_lat) + ', ' + str(median_pred_lng))
        print('Density lat, lng: ' + str(density_pred_lat) + ', ' + str(density_pred_lng))
        print('Avg. Interval (min): ' + str(intervals[place]))
        print('Sunrise / Sunset Visible Breakdown of Days: ' + str(sun_visibles[place]))
        print('')
        sys.stdout.flush()

# Plot Error vs Days Used
def scatter(days_used, distances, fmt, label, color=None, linestyle=None, marker=None, cbm=False):
    plt.figure(figsize=(24,12))
    if fmt is not None:
        days_err, = plt.plot(days_used, distances, fmt, markersize=3, label=label)
    else:
        days_err, = plt.plot(days_used, distances, color=color, linestyle=linestyle, marker=marker, markersize=3, label='gaussian kde')

    plt.legend(handles=[days_err])
    plt.xlabel('# Days Used')
    plt.ylabel('Avg. Error (km)')
    plt.title('Avg. Error (km) Using ' + label[0].upper() + label[1:] + ' vs. # Days Used')

    if cbm:
        prefix = 'cbm_'
    else:
        prefix = ''

    plt.savefig('/srv/glusterfs/vli/maps/' + prefix + label + '_days_used.png')
    plt.close()

scatter(days_used, mean_distances, 'mo', 'mean')
scatter(days_used, median_distances, 'co', 'median')
scatter(days_used, density_distances, None, 'gaussian kde', color=mcolors.CSS4_COLORS['fuchsia'], linestyle='None', marker='o')

scatter(days_used, cbm_mean_distances, 'mo', 'mean', cbm=True)
scatter(days_used, cbm_median_distances, 'co', 'median', cbm=True)
scatter(days_used, cbm_density_distances, None, 'gaussian kde', color=mcolors.CSS4_COLORS['fuchsia'], linestyle='None', marker='o', cbm=True)

def bar(x, y, xlabel, ylabel, x_labels, title, filename):
    plt.figure(figsize=(24, 12))
    x = np.arange(len(x))
    #y = bucket_distances
    width = 0.35
    plt.bar(x, y, width, color='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    plt.title(title)
    plt.savefig('/srv/glusterfs/vli/maps/' + filename)
    plt.close()

# Plot average distance error vs. time interval OVER ALL DAYS.
buckets = list(range(0, round(24 * 60 / constants.IMAGES_PER_DAY) + 5, 5)) # 5 minute intervals
bucket_labels = [str(x) + '-' + str(x + 5) for x in buckets]
bucket_labels[-1] = bucket_labels[-1] + '+'
bucket_distances = [[] for x in range(len(buckets))]
cbm_bucket_distances = [[] for x in range(len(buckets))]
for i in range(data.types['test']):
    for bIdx, bucket in enumerate(buckets):
        if days[i].interval_min < bucket + 5:
            break

    distance_err = compute_distance(days[i].lat, days[i].lng, latitudes[i], longitudes[i])
    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, cbm_latitudes[i], longitudes[i])

    bucket_distances[bIdx].append(distance_err)
    cbm_bucket_distances[bIdx].append(cbm_distance_err)

for bdIdx, distance_errs in enumerate(bucket_distances):
    if len(distance_errs) > 0:
        bucket_distances[bdIdx] = statistics.mean(distance_errs)
    else:
        bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_bucket_distances):
    if len(distance_errs) > 0:
        cbm_bucket_distances[bdIdx] = statistics.mean(distance_errs)
    else:
        cbm_bucket_distances[bdIdx] = 0

bar(buckets, bucket_distances, 'Avg. Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Avg. Error (km) Over All Days vs. Photo Interval (min)', 'interval.png')
bar(buckets, cbm_bucket_distances, 'Avg. Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Avg. Error (km) Over All Days vs. Photo Interval (min)', 'cbm_interval.png')

# Plot average distance error vs. sunrise, sunset available over ALL DAYS.
sun_type_labels = ['Both', 'Sunrise Only', 'Sunset Only', 'Neither']
sun_type_distances = [[] for x in range(len(sun_type_labels))]
cbm_sun_type_distances = [[] for x in range(len(sun_type_labels))]
for i in range(data.types['test']):
    distance_err = compute_distance(days[i].lat, days[i].lng, latitudes[i], longitudes[i])
    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, cbm_latitudes[i], longitudes[i])

    if days[i].sunrise_in_frames and days[i].sunset_in_frames:
        sun_type_distances[0].append(distance_err)
        cbm_sun_type_distances[0].append(cbm_distance_err)
    elif days[i].sunrise_in_frames:
        sun_type_distances[1].append(distance_err)
        cbm_sun_type_distances[1].append(cbm_distance_err)
    elif days[i].sunset_in_frames:
        sun_type_distances[2].append(distance_err)
        cbm_sun_type_distances[2].append(cbm_distance_err)
    else:
        sun_type_distances[3].append(distance_err)
        cbm_sun_type_distances[3].append(cbm_distance_err)

for sIdx, distance_errs in enumerate(sun_type_distances):
    if len(distance_errs) > 0:
        sun_type_distances[sIdx] = statistics.mean(distance_errs)
    else:
        sun_type_distances[sIdx] = 0

for sIdx, distance_errs in enumerate(cbm_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_sun_type_distances[sIdx] = statistics.mean(distance_errs)
    else:
        cbm_sun_type_distances[sIdx] = 0

bar(sun_type_labels, sun_type_distances, 'Avg. Distance Error (km)', 'Sunrise and sunset in frame?', sun_type_labels, 'Avg. Error (km) Over All Days vs. Sunrise / Sunset In Frame', 'sun_in_frame.png')
bar(sun_type_labels, cbm_sun_type_distances, 'Avg. Distance Error (km)', 'Sunrise and sunset in frame?', sun_type_labels, 'Avg. Error (km) Over All Days vs. Sunrise / Sunset In Frame', 'cbm_sun_in_frame.png')

# Plot average distance error vs. season over ALL DAYS.
season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
season_distances = [[] for x in range(len(season_labels))]
cbm_season_distances = [[] for x in range(len(season_labels))]
for i in range(data.types['test']):
    distance_err = compute_distance(days[i].lat, days[i].lng, latitudes[i], longitudes[i])
    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, cbm_latitudes[i], longitudes[i])

    if days[i].season == 'winter':
        season_distances[0].append(distance_err)
        cbm_season_distances[0].append(cbm_distance_err)
    elif days[i].season == 'spring':
        season_distances[1].append(distance_err)
        cbm_season_distances[1].append(cbm_distance_err)
    elif days[i].season == 'summer':
        season_distances[2].append(distance_err)
        cbm_season_distances[2].append(cbm_distance_err)
    else:
        season_distances[3].append(distance_err)
        cbm_season_distances[3].append(cbm_distance_err)

for sIdx, distance_errs in enumerate(season_distances):
    if len(distance_errs) > 0:
        season_distances[sIdx] = statistics.mean(distance_errs)
    else:
        season_distances[sIdx] = 0

for sIdx, distance_errs in enumerate(cbm_season_distances):
    if len(distance_errs) > 0:
        cbm_season_distances[sIdx] = statistics.mean(distance_errs)
    else:
        cbm_season_distances[sIdx] = 0

bar(season_labels, season_distances, 'Avg. Distance Error (km)', 'Season', season_labels, 'Avg. Error (km) Over All Days vs. Season', 'season.png')
bar(season_labels, cbm_season_distances, 'Avg. Distance Error (km)', 'Season', season_labels, 'Avg. Error (km) Over All Days vs. Season', 'cbm_season.png')

# Plot average distance error vs. intervals over ALL PLACES.
# Only using CBM model for now.
cbm_median_bucket_distances = [[] for x in range(len(buckets))]
cbm_density_bucket_distances = [[] for x in range(len(buckets))]
for key in intervals:
    for bIdx, bucket in enumerate(buckets):
        if intervals[key] < bucket + 5:
            break

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_bucket_distances[bIdx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_bucket_distances[bIdx].append(cbm_density_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_bucket_distances):
    if len(distance_errs) > 0:
        cbm_median_bucket_distances[bdIdx] = statistics.mean(distance_errs)
    else:
        cbm_median_bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_bucket_distances):
    if len(distance_errs) > 0:
        cbm_density_bucket_distances[bdIdx] = statistics.mean(distance_errs)
    else:
        cbm_density_bucket_distances[bdIdx] = 0

bar(buckets, cbm_median_bucket_distances, 'Avg. Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Avg. Error (km) Over All Locations Using Median vs. Photo Interval (min)', 'cbm_interval_median_places.png')
bar(buckets, cbm_density_bucket_distances, 'Avg. Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Avg. Error (km) Over All Locations Using Gaussian KDE vs. Photo Interval (min)', 'cbm_interval_density_places.png')

# Plot average distance error vs. percentage of days with sunrise and sunset visible over ALL PLACES.
# Only using CBM model for now.
buckets = list(range(0, 100, 10)) # 10% buckets
bucket_labels = [str(x) + '-' + str(x + 10) for x in buckets]
cbm_median_sun_type_distances = [[] for x in range(len(buckets))]
cbm_density_sun_type_distances = [[] for x in range(len(buckets))]
for key in intervals:
    for bIdx, bucket in enumerate(buckets):
        if sun_visibles[key][0] * 100 < bucket + 10:
            break

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_sun_type_distances[bIdx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_sun_type_distances[bIdx].append(cbm_density_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_median_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
    else:
        cbm_median_sun_type_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_density_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
    else:
        cbm_density_sun_type_distances[bdIdx] = 0

bar(buckets, cbm_median_sun_type_distances, 'Avg. Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using Median vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_median_places.png')
bar(buckets, cbm_density_sun_type_distances, 'Avg. Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using Gaussian KDE vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_density_places.png')

# Average distance error vs. latitude over ALL PLACES.

buckets = list(range(-90, 90, 10)) # 10 degree buckets
bucket_labels = [str(x) + '-' + str(x + 10) for x in buckets]
cbm_median_lat_distances = [[] for x in range(len(buckets))]
cbm_density_lat_distances = [[] for x in range(len(buckets))]



print('Brock Means Avg. Distance Error: {:.6f}'.format(statistics.mean(mean_distances)))
print('Brock Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(median_distances)))
print('Brock Density Avg. Distance Error: {:.6f}'.format(statistics.mean(density_distances)))
print('Brock Means Median Distance Error: {:.6f}'.format(statistics.median(mean_distances)))
print('Brock Medians Median Distance Error: {:.6f}'.format(statistics.median(median_distances)))
print('Brock Density Median Distance Error: {:.6f}'.format(statistics.median(density_distances)))
print('Brock Means Max Distance Error: {:.6f}'.format(max(mean_distances)))
print('Brock Means Min Distance Error: {:.6f}'.format(min(mean_distances)))
print('Brock Medians Max Distance Error: {:.6f}'.format(max(median_distances)))
print('Brock Medians Min Distance Error: {:.6f}'.format(min(median_distances)))
print('Brock Density Max Distance Error: {:.6f}'.format(max(density_distances)))
print('Brock Density Min Distance Error: {:.6f}'.format(min(density_distances)))
print('')
print('CBM Means Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_mean_distances)))
print('CBM Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_median_distances)))
print('CBM Density Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_density_distances)))
print('CBM Means Median Distance Error: {:.6f}'.format(statistics.median(cbm_mean_distances)))
print('CBM Medians Median Distance Error: {:.6f}'.format(statistics.median(cbm_median_distances)))
print('CBM Density Median Distance Error: {:.6f}'.format(statistics.median(cbm_density_distances)))
print('CBM Means Max Distance Error: {:.6f}'.format(max(cbm_mean_distances)))
print('CBM Means Min Distance Error: {:.6f}'.format(min(cbm_mean_distances)))
print('CBM Medians Max Distance Error: {:.6f}'.format(max(cbm_median_distances)))
print('CBM Medians Min Distance Error: {:.6f}'.format(min(cbm_median_distances)))
print('CBM Density Max Distance Error: {:.6f}'.format(max(cbm_density_distances)))
print('CBM Density Min Distance Error: {:.6f}'.format(min(cbm_density_distances)))
print('')
print('Means Avg. Longitude Error: {:.6f}'.format(statistics.mean(mean_longitude_err)))
print('Medians Avg. Longitude Error: {:.6f}'.format(statistics.mean(median_longitude_err)))
print('Density Avg. Longitude Error: {:.6f}'.format(statistics.mean(density_longitude_err)))
print('Means Median Longitude Error: {:.6f}'.format(statistics.median(mean_longitude_err)))
print('Medians Median Longitude Error: {:.6f}'.format(statistics.median(median_longitude_err)))
print('Density Median Longitude Error: {:.6f}'.format(statistics.median(density_longitude_err)))
print('Means Max Longitude Error: {:.6f}'.format(max(mean_longitude_err)))
print('Means Min Longitude Error: {:.6f}'.format(min(mean_longitude_err)))
print('Medians Max Longitude Error: {:.6f}'.format(max(median_longitude_err)))
print('Medians Min Longitude Error: {:.6f}'.format(min(median_longitude_err)))
print('Density Max Longitude Error: {:.6f}'.format(max(density_longitude_err)))
print('Density Min Longitude Error: {:.6f}'.format(min(density_longitude_err)))
print('')
print('Brock Means Avg. Latitude Error: {:.6f}'.format(statistics.mean(mean_latitude_err)))
print('Brock Medians Avg. Latitude Error: {:.6f}'.format(statistics.mean(median_latitude_err)))
print('Brock Density Avg. Latitude Error: {:.6f}'.format(statistics.mean(density_latitude_err)))
print('Brock Means Median Latitude Error: {:.6f}'.format(statistics.median(mean_latitude_err)))
print('Brock Medians Median Latitude Error: {:.6f}'.format(statistics.median(median_latitude_err)))
print('Brock Density Median Latitude Error: {:.6f}'.format(statistics.median(density_latitude_err)))
print('Brock Means Max Latitude Error: {:.6f}'.format(max(mean_latitude_err)))
print('Brock Means Min Latitude Error: {:.6f}'.format(min(mean_latitude_err)))
print('Brock Medians Max Latitude Error: {:.6f}'.format(max(median_latitude_err)))
print('Brock Medians Min Latitude Error: {:.6f}'.format(min(median_latitude_err)))
print('Brock Density Max Latitude Error: {:.6f}'.format(max(density_latitude_err)))
print('Brock Density Min Latitude Error: {:.6f}'.format(min(density_latitude_err)))
print('')
print('CBM Means Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_mean_latitude_err)))
print('CBM Medians Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_median_latitude_err)))
print('CBM Density Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_density_latitude_err)))
print('CBM Means Median Latitude Error: {:.6f}'.format(statistics.median(cbm_mean_latitude_err)))
print('CBM Medians Median Latitude Error: {:.6f}'.format(statistics.median(cbm_median_latitude_err)))
print('CBM Density Median Latitude Error: {:.6f}'.format(statistics.median(cbm_density_latitude_err)))
print('CBM Means Max Latitude Error: {:.6f}'.format(max(cbm_mean_latitude_err)))
print('CBM Means Min Latitude Error: {:.6f}'.format(min(cbm_mean_latitude_err)))
print('CBM Medians Max Latitude Error: {:.6f}'.format(max(cbm_median_latitude_err)))
print('CBM Medians Min Latitude Error: {:.6f}'.format(min(cbm_median_latitude_err)))
print('CBM Density Max Latitude Error: {:.6f}'.format(max(cbm_density_latitude_err)))
print('CBM Density Min Latitude Error: {:.6f}'.format(min(cbm_density_latitude_err)))
print('')
sys.stdout.flush()