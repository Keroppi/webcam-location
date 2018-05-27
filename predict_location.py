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

temp = torch.randn(100).cuda()

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

    sunrise_predict_t0 = time.time()
    #sunrises = []
    for batch_idx, (input, _) in enumerate(test_loader):
        input = Variable(input, volatile=True)

        if torch.cuda.is_available():
            input = input.cuda()

        sunrise_idx = sunrise_model(input)

        # Convert sunrise_idx into a local time.
        batch_days = days[batch_idx * constants.BATCH_SIZE:batch_idx * constants.BATCH_SIZE + sunrise_idx.size()[0]]

        '''
        for d_idx, day in enumerate(batch_days):
            local_sunrise = day.get_local_time(sunrise_idx[d_idx, 0].data[0])
            #utc_sunrise = local_sunrise - datetime.timedelta(seconds=day.time_offset)
            sunrises.append(local_sunrise)
        '''

        for d_idx, day in enumerate(batch_days):
            day.change_frames(sunrise_idx[d_idx, 0].data[0])

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

        '''
        for d_idx, day in enumerate(batch_days):
            local_sunset = day.get_local_time(sunset_idx[d_idx, 0].data[0])
            #utc_sunset = local_sunset - datetime.timedelta(seconds=day.time_offset)
            sunsets.append(local_sunset)
        '''

        for d_idx, day in enumerate(batch_days):
            day.change_frames(sunset_idx[d_idx, 0].data[0])

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

def kde_func_to_minimize(x, kernel):
    x = x.reshape(1, 2)
    density = kernel.score_samples(x)
    return -density[0] # Trying to minimize

# Kernel density estimation to guess location.

density_locations = {}

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
        print('WARNING - scipy minimize function failed on location ' + key)
        sys.stdout.flush()

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

# Plot locations on a map.
for place in lats:
    if len(lats[place]) < 100: # Need at least 100 points.
        continue

    min_lat = max(min(lats[place]) - 1, -90)
    max_lat = min(max(lats[place]) + 1, 90)
    min_lng = max(min(lngs[place]) - 1, -180)
    max_lng = min(max(lngs[place]) + 1, 180)

    colors = []

    actual_lng = float('inf')
    actual_lat = float('inf')

    for i in range(data.types['test']):
        if days[i].place != place:
            continue

        actual_lng = days[i].lng
        actual_lat = days[i].lat

        if days[i].sunrise_in_frames and days[i].sunset_in_frames:
            colors.append('g')
        elif not days[i].sunrise_in_frames and days[i].sunset_in_frames:
            colors.append('r')
        elif not days[i].sunset_in_frames and days[i].sunrise_in_frames:
            colors.append(mcolors.CSS4_COLORS['crimson'])
        else: # not days[i].sunrise_in_frames and not days[i].sunset_in_frames:
            colors.append('k')

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
    guess_colors = ['g', 'r', mcolors.CSS4_COLORS['crimson'], 'k']
    legend_labels = ['sunrise and sunset in frames', 'sunrise not in frames', 'sunset not in frames', 'sunrise and sunset not in frames', 'actual location', 'mean', 'median', 'gaussian kde']

    handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                 [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]
    plt.legend(handlelist, legend_labels)

    plt.title(place)
    plt.savefig('/srv/glusterfs/vli/maps/' + place + '.png')
    plt.close()


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

days_used = []
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

    days_used.append(len(lats[place]))

    actual_lat = days[i].lat
    actual_lng = days[i].lng

    mean_pred_lat = mean_locations[place][0]
    mean_pred_lng = mean_locations[place][1]
    median_pred_lat = median_locations[place][0] #places_lat_lng[place][0]
    median_pred_lng = median_locations[place][1] #places_lat_lng[place][1]

    #if density_locations[place] is not None:
    density_pred_lat = density_locations[place][0]
    density_pred_lng = density_locations[place][1]

    mean_distance = compute_distance(actual_lat, actual_lng, mean_pred_lat, mean_pred_lng)
    mean_distances.append(mean_distance)
    median_distance = compute_distance(actual_lat, actual_lng, median_pred_lat, median_pred_lng)
    median_distances.append(median_distance)

    #if density_locations[place] is not None:
    density_distance = compute_distance(actual_lat, actual_lng, density_pred_lat, density_pred_lng)
    density_distances.append(density_distance)

    if random.randint(1, 100) < 101: # VLI
        print('Distance')
        print(place)
        print('# Days Used: ' + str(days_used[-1]))
        print('Using mean: ' + str(mean_distance))
        print('Using median: ' + str(median_distance))
        print('Using density: ' + str(density_distance))
        print(np.vstack((np.array(lats[place]), np.array(lngs[place]))))
        print(str(actual_lat) + ', ' + str(actual_lng))
        print(str(mean_pred_lat) + ', ' + str(mean_pred_lng))
        print(str(median_pred_lat) + ', ' + str(median_pred_lng))
        print(str(density_pred_lat) + ', ' + str(density_pred_lng))

        print('')
        sys.stdout.flush()

    #print(distance)


# Plot Error vs Days Used
plt.figure(figsize=(24,12))
mean_days_err, = plt.plot(days_used, mean_distances, 'mo', markersize=3, label='mean')
plt.legend(handles=[mean_days_err])
plt.xlabel('# Days Used')
plt.ylabel('Avg. Error (km)')
plt.title('Avg. Error (km) Using Mean vs. # Days Used')
plt.savefig('/srv/glusterfs/vli/maps/mean_days_used.png')
plt.close()

plt.figure(figsize=(24,12))
median_days_err, = plt.plot(days_used, median_distances, 'co', markersize=3, label='median')
plt.legend(handles=[median_days_err])
plt.xlabel('# Days Used')
plt.ylabel('Avg. Error (km)')
#plt.xlim(xmin=0)
#plt.ylim(ymin=0)
plt.title('Avg. Error (km) Using Median vs. # Days Used')
plt.savefig('/srv/glusterfs/vli/maps/median_days_used.png')
plt.close()

plt.figure(figsize=(24,12))
density_days_err, = plt.plot(days_used, density_distances, color=mcolors.CSS4_COLORS['fuchsia'], linestyle='None', marker='o', markersize=3, label='gaussian kde')
plt.legend(handles=[density_days_err])
plt.xlabel('# Days Used')
plt.ylabel('Avg. Error (km)')
#plt.xlim(xmin=0)
#plt.ylim(ymin=0)
plt.title('Avg. Error (km) Using Gaussian KDE vs. # Days Used')
plt.savefig('/srv/glusterfs/vli/maps/kde_days_used.png')
plt.close()

#average_dist /= len(finished_places)
print('Means Avg. Distance Error: {:.6f}'.format(statistics.mean(mean_distances)))
print('Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(median_distances)))
print('Density Avg. Distance Error: {:.6f}'.format(statistics.mean(density_distances)))
print('Means Median Distance Error: {:.6f}'.format(statistics.median(mean_distances)))
print('Medians Median Distance Error: {:.6f}'.format(statistics.median(median_distances)))
print('Density Median Distance Error: {:.6f}'.format(statistics.median(density_distances)))
print('Means Max Distance Error: {:.6f}'.format(max(mean_distances)))
print('Means Min Distance Error: {:.6f}'.format(min(mean_distances)))
print('Medians Max Distance Error: {:.6f}'.format(max(median_distances)))
print('Medians Min Distance Error: {:.6f}'.format(min(median_distances)))
print('Density Max Distance Error: {:.6f}'.format(max(density_distances)))
print('Density Min Distance Error: {:.6f}'.format(min(density_distances)))
sys.stdout.flush()