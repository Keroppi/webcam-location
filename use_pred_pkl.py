#!/srv/glusterfs/vli/.pyenv/shims/python

import matplotlib
matplotlib.use('agg')

import os, argparse, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, pickle, collections
from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from simple_day import SimpleDay
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

print('Starting predict location.')
sys.stdout.flush()

parser = argparse.ArgumentParser(description='Predict Location')
parser.add_argument('--pickle_file', default='', type=str, metavar='PATH',
                    help='path to prediction pickle file (default: none)')
args = parser.parse_args()

if args.pickle_file:
    with open(args.pickle_file, 'rb') as f:
        predictions = pickle.load(f)
else:
    print('Wrong arguments.')
    sys.stdout.flush()
    sys.exit(1)

sunrises = []
sunsets = []
days = []
for place in predictions:
    days.append(predictions[place])
    sunrises.append(predictions[place].sunrise)
    sunsets.append(predictions[place].sunset)

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
    if random.randint(1, 100) < 5:
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
    if random.randint(1, 100) < 5:
        print('Lng')
        print(lng)
        print('')
        sys.stdout.flush()
    '''

    longitudes.append(lng)

# CBM Model
def cbm(day_of_year, day_length_hours):
    theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
    phi = math.asin(0.39795 * math.cos(theta))
    cbm_lat = 180 / math.pi * math.atan(math.cos(phi) / math.sin(phi) * math.cos(-math.pi / 24 * (day_length_hours - 24)))

    return cbm_lat

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
    cbm_lat = cbm(day_of_year, day_length_hours)

    # Check if they're in different (north / south) hemispheres.
    if (cbm_lat > 0 and days[d_idx].lat < 0) or (cbm_lat < 0 and days[d_idx].lat > 0):
        theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
        phi = math.asin(0.39795 * math.cos(theta))
        denominator = math.sin(phi)
        numerator = math.cos(phi) * math.cos(-math.pi / 24 * (day_length_hours - 24))
        denominator1 = math.tan(phi)
        numerator1 = math.cos(-math.pi / 24 * (day_length_hours - 24))

        #EPSILON = 0.05
        #if math.fabs(denominator) < EPSILON or math.fabs(denominator1) < EPSILON:
        #    cbm_lat *= -1

        print('WARNING - Different hemispheres')
        print(days[d_idx].place)
        print('Denominator (sin) value: ' + str(denominator))
        print('Numerator (sin) value: ' + str(numerator))
        print('Denominator (tan) value: ' + str(denominator1))
        print('Numerator (tan) value: ' + str(numerator1))

    '''
    if random.randint(1, 100) < 5:
        print('Lat')
        print(lat)
        print('')
        sys.stdout.flush()
    '''

    latitudes.append(lat) # Only one day to predict latitude - could average across many days.
    cbm_latitudes.append(cbm_lat)  # Only one day to predict latitude - could average across many days.

    if cbm_lat > 90 or cbm_lat < -90:
        print('WARNING - CBM latitude out of range')
        print(cbm_lat)
        print((day_of_year, day_length_hours))
    if lat > 90 or lat < -90:
        print('WARNING - Brock latitude out of range')
        print(lat)
        print((day_of_year, day_length_hours))

    sys.stdout.flush()

'''
# Store which day of the year and day length for each place.
day_lens = {}
days_of_year = {}
for i in range(len(days)):
    if days_of_year.get(days[i].place) is None:
        days_of_year[days[i].place] = []

    if day_lens.get(days[i].place) is None:
        day_lens[days[i].place] = []

    ts = pd.Series(pd.to_datetime([str(days[i].date)]))
    day_of_year = int(ts.dt.dayofyear)  # day_of_year from 1 to 365, inclusive

    days_of_year[days[i].place].append(days_of_year)
    day_lens[days[i].place].append(day_lengths[i])
'''

# Get mean and median of all lat/longs of same places.
#places = {}
cbm_lats = {}
lats = {}
lngs = {}
for i in range(len(days)):
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


def compute_haversine_distance(lat1, lng1, lat2, lng2):  # kilometers
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


def compute_distance(lat1, lng1, lat2, lng2):  # km
    if lng1 == -180:  #
        lng1 = 180  #

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lng1_rad = math.radians(lng1)  #
    lng2_rad = math.radians(lng2)  #

    a = 6378.137  # km
    b = 6356.752314245  # km
    f = 1 / 298.257223563

    L = lng2_rad - lng1_rad  # math.radians(lng2 - lng1) #

    tanU1 = (1 - f) * math.tan(lat1_rad)
    cosU1 = 1 / math.sqrt((1 + tanU1 * tanU1))
    sinU1 = tanU1 * cosU1
    tanU2 = (1 - f) * math.tan(lat2_rad)
    cosU2 = 1 / math.sqrt((1 + tanU2 * tanU2))
    sinU2 = tanU2 * cosU2

    lamb = L
    iterations = 0
    antimeridian = math.fabs(L) > math.pi

    while True:
        sin_lambda = math.sin(lamb)
        cos_lambda = math.cos(lamb)

        sin_sigma = math.sqrt(math.pow(cosU2 * sin_lambda, 2) + math.pow(cosU1 * sinU2 - sinU1 * cosU2 * cos_lambda, 2))

        if sin_sigma == 0:  #
            iterations += 1001  # coincident points #

        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)  #

        sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - math.pow(sin_alpha, 2)

        if cos_sq_alpha != 0:  #
            cos_2_sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha  #
        else:  # Equatorial line #
            cos_2_sigma_m = 0  #

        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        new_lamb = L + (1 - C) * f * sin_alpha * (
            sigma + C * sin_sigma * (cos_2_sigma_m + C * cos_sigma * (-1 + 2 * math.pow(cos_2_sigma_m, 2))))

        if antimeridian:  #
            iteration_check = math.fabs(new_lamb) - math.pi  #
        else:  #
            iteration_check = math.fabs(new_lamb)  #

        if iteration_check > math.pi:  #
            iterations += 1001  #

        if iterations > 1000:
            print('WARNING - Vincenty distance did not converge.')
            sys.stdout.flush()

            return compute_haversine_distance(lat1, lng1, lat2, lng2)  # Use haversine distance if it doesn't converge.

        if math.fabs(new_lamb - lamb) < 1e-12:
            u_sq = cos_sq_alpha * (a * a - b * b) / (b * b)
            A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
            B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
            delta_sigma = B * sin_sigma * (cos_2_sigma_m + 1 / 4 * B * (
                cos_sigma * (-1 + 2 * math.pow(cos_2_sigma_m, 2)) - B / 6 * cos_2_sigma_m * (
                    -3 + 4 * math.pow(sin_sigma, 2)) * (-3 + 4 * math.pow(cos_2_sigma_m, 2))))
            s = b * A * (sigma - delta_sigma)

            return s

        iterations += 1
        lamb = new_lamb

def ransac(lats, lngs):
    ransacs = {}

    for place in lats:
        guesses = list(zip(lats[place], lngs[place]))

        inliers = [[] for x in range(len(guesses))]
        for g_idx1, guess1 in enumerate(guesses):
            inliers[g_idx1].append(guess1)

            for g_idx2, guess2 in enumerate(guesses[:g_idx1] + guesses[g_idx1 + 1:]):
                if compute_distance(guess1[0], guess1[1], guess2[0], guess2[1]) < constants.INLIER_THRESHOLD:
                    inliers[g_idx1].append(guess2)

        max_inliers = -1
        max_idx = -1
        for i_idx, inlier in enumerate(inliers):
            if len(inlier) > max_inliers:
                max_idx = i_idx
                max_inliers = len(inlier)

        ransacs[place] = (statistics.mean([x[0] for x in inliers[max_idx]]), statistics.mean([x[1] for x in inliers[max_idx]]))

    return ransacs

ransac_locations = ransac(cbm_lats, lngs)

# Collect intervals of each place.
intervals = {}
for i in range(len(days)):
    if intervals.get(days[i].place) is None:
        intervals[days[i].place] = []
    intervals[days[i].place].append(days[i].interval_min)

# Average intervals for each place.
for key in intervals:
    intervals[key] = statistics.median(intervals[key])

# Collect breakdown of sunrise / sunset visible.
sun_visibles = {}
for i in range(len(days)):
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
        res = minimize(kde_func_to_minimize, np.asarray([math.radians(x) for x in median_locations[key]]), args=(kernel,), method='L-BFGS-B', bounds=bnds, options={'maxiter':150})

        if res.success:
            #density_locations[key] = (math.degrees(res.x[0]), math.degrees(res.x[1]))
            best_latitude = math.degrees(res.x[0])
            best_longitude = math.degrees(res.x[1])
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

        if best_longitude > 180 or best_longitude < -180:
            print('WARNING - KDE returns out of bound longitude.')
            print(key)
            print(best_longitude)
        if best_latitude > 90 or best_latitude < -90:
            print('WARNING - KDE returns out of bound latitude.')
            print(key)
            print(best_latitude)
        sys.stdout.flush()

        density_locations[key] = (best_latitude, best_longitude)

    kernel_t1 = time.time()
    print('Calculating density time (m): ' + str((kernel_t1 - kernel_t0) / 60))
    sys.stdout.flush()

    return density_locations

density_locations = kde(lats, lngs, median_locations)
cbm_density_locations = kde(cbm_lats, lngs, cbm_median_locations)

actual_locations = {}
for i in range(len(days)):
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

        for i in range(len(days)):
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

        actual_and_pred_lngs = [actual_lng] + [mean_locations[place][1]] + [median_locations[place][1]] + [density_locations[place][1]] + [ransac_locations[place][1]]
        actual_and_pred_lats = [actual_lat] + [mean_locations[place][0]] + [median_locations[place][0]] + [density_locations[place][0]] + [ransac_locations[place][0]]
        actual_and_pred_colors = ['w', 'm', 'c', mcolors.CSS4_COLORS['fuchsia'], 'xkcd:chartreuse']

        guesses = map.scatter(lngs[place], lats[place], s=40, c=colors, latlon=True, zorder=10)
        actual_and_pred = map.scatter(actual_and_pred_lngs, actual_and_pred_lats, s=40, c=actual_and_pred_colors, latlon=True, zorder=10, marker='^')

        #plt.legend(handles=[guesses, actual, mean_guess, median_guess, density_guess])

        if mode == 'sun':
            guess_colors = ['g', 'r', mcolors.CSS4_COLORS['crimson'], 'k']
            legend_labels = ['sunrise and sunset in frames', 'sunrise not in frames', 'sunset not in frames', 'sunrise and sunset not in frames', 'actual location', 'mean', 'median', 'gaussian kde', 'RANSAC']

            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                         [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]
        elif mode == 'season':
            guess_colors = ['b', 'y', 'r', mcolors.CSS4_COLORS['tan']]
            legend_labels = ['winter', 'spring', 'summer', 'fall',
                             'actual location', 'mean', 'median', 'gaussian kde', 'RANSAC']
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                         [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]

        plt.legend(handlelist, legend_labels)

        plt.title(place)
        plt.savefig('/srv/glusterfs/vli/maps/' + mode + '/' + place + '.png')
        plt.close()

plot_map(lats, lngs, mean_locations, median_locations, density_locations, 'sun')
plot_map(cbm_lats, lngs, cbm_mean_locations, cbm_median_locations, cbm_density_locations, 'season')

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

ransac_distances = []
ransac_longitude_err = []
ransac_latitude_err = []

#for i in range(len(days)):
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
    ransac_pred_lat = ransac_locations[place][0]
    ransac_pred_lng = ransac_locations[place][1]

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

    ransac_distance = compute_distance(actual_lat, actual_lng, ransac_pred_lat, ransac_pred_lng)
    ransac_distances.append(ransac_distance)
    ransac_latitude_err.append(compute_distance(actual_lat, actual_lng, ransac_pred_lat, actual_lng))
    ransac_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, ransac_pred_lng))

    if random.randint(1, 100) < 101:
        if median_distance < 25 or density_distance < 25 or ransac_distance < 25 or \
           cbm_median_distance < 25 or cbm_density_distance < 25:
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
        print('Using RANSAC: ' + str(ransac_distance))
        print('Actual lat, lng: ' + str(actual_lat) + ', ' + str(actual_lng))
        print('Brock Distance')
        print('Mean lat, lng: ' + str(mean_pred_lat) + ', ' + str(mean_pred_lng))
        print('Median lat, lng: ' + str(median_pred_lat) + ', ' + str(median_pred_lng))
        print('Density lat, lng: ' + str(density_pred_lat) + ', ' + str(density_pred_lng))
        print('CBM Distance')
        print('Mean lat, lng: ' + str(cbm_mean_pred_lat) + ', ' + str(cbm_mean_pred_lng))
        print('Median lat, lng: ' + str(cbm_median_pred_lat) + ', ' + str(cbm_median_pred_lng))
        print('Density lat, lng: ' + str(cbm_density_pred_lat) + ', ' + str(cbm_density_pred_lng))
        print('RANSAC lat, lng: ' + str(ransac_pred_lat) + ', ' + str(ransac_pred_lng))
        print('Median Interval (min): ' + str(intervals[place]))
        print('Sunrise / Sunset Visible Breakdown of Days: ' + str(sun_visibles[place]))
        print('')
        sys.stdout.flush()

# Plot Error vs Days Used
def scatter(days_used, distances, fmt, label, color=None, linestyle=None, marker=None, cbm=False):
    plt.figure(figsize=(24,12))

    days_used_medians = {}
    for d_idx, days in enumerate(days_used):

        if days not in days_used_medians:
            days_used_medians[days] = []

        days_used_medians[days].append(distances[d_idx])

    for days in days_used_medians:
        days_used_medians[days] = statistics.median(days_used_medians[days])

    if fmt is not None:
        days_err, = plt.plot(days_used, distances, fmt, markersize=3, label=label)
    else:
        days_err, = plt.plot(days_used, distances, color=color, linestyle=linestyle, marker=marker, markersize=3, label=label)

    days_used_medians = collections.OrderedDict(sorted(days_used_medians.items()))

    means_err, = plt.plot(list(days_used_medians.keys()), list(days_used_medians.values()), color='k', linestyle='-',
                          marker='^', markersize=9, label='Trend Using Median')

    plt.legend(handles=[days_err, means_err])
    plt.xlabel('# Days Used')
    plt.ylabel('Error (km)')
    plt.title('Error (km) Using ' + label[0].upper() + label[1:] + ' vs. # Days Used')

    if cbm:
        prefix = 'cbm_'
    else:
        prefix = ''

    plt.savefig('/srv/glusterfs/vli/maps/' + prefix + label + '_days_used.png')
    plt.close()

#scatter(days_used, mean_distances, 'mo', 'mean')
scatter(days_used, median_distances, 'co', 'median')
scatter(days_used, density_distances, None, 'gaussian kde', color=mcolors.CSS4_COLORS['fuchsia'], linestyle='None', marker='o')

#scatter(days_used, cbm_mean_distances, 'mo', 'mean', cbm=True)
scatter(days_used, cbm_median_distances, 'co', 'median', cbm=True)
scatter(days_used, cbm_density_distances, None, 'gaussian kde', color=mcolors.CSS4_COLORS['fuchsia'], linestyle='None', marker='o', cbm=True)
scatter(days_used, ransac_distances, None, 'RANSAC', color='xkcd:chartreuse', linestyle='None', marker='o', cbm=True)

def median_rmse(data):
    median = statistics.median(data)
    mse = mean_squared_error(data, [median] * len(data))
    return math.sqrt(mse)

def bar(x, y, ylabel, xlabel, x_labels, title, filename, yerr=None):
    plt.figure(figsize=(24, 12))
    x = np.arange(len(x))
    #y = bucket_distances
    width = 0.35
    plt.bar(x, y, width, color='r', yerr=[(0,) * len(x), tuple(yerr)])
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
bucket_rmses = [0 for x in range(len(buckets))] #
cbm_bucket_rmses = [0 for x in range(len(buckets))] #
bucket_num_data_pts = [0] * len(buckets) #
cbm_bucket_num_data_pts = [0] * len(buckets) #
for i in range(len(days)):
    for bIdx, bucket in enumerate(buckets):
        if days[i].interval_min < bucket + 5:
            break

    distance_err = compute_distance(days[i].lat, days[i].lng, latitudes[i], longitudes[i])
    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, cbm_latitudes[i], longitudes[i])

    bucket_distances[bIdx].append(distance_err)
    cbm_bucket_distances[bIdx].append(cbm_distance_err)

for bdIdx, distance_errs in enumerate(bucket_distances):
    if len(distance_errs) > 0:
        bucket_distances[bdIdx] = statistics.median(distance_errs)
        bucket_rmses[bdIdx] = median_rmse(distance_errs) #
        bucket_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_bucket_distances):
    if len(distance_errs) > 0:
        cbm_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_bucket_rmses[bdIdx] = median_rmse(distance_errs) #
        cbm_bucket_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        cbm_bucket_distances[bdIdx] = 0

#bar(buckets, bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Days vs. Photo Interval (min)', 'interval.png', bucket_rmses)
bar(buckets, cbm_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Days vs. Photo Interval (min)', 'cbm_interval.png', cbm_bucket_rmses) #
print('INTERVAL OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_bucket_num_data_pts)) #

# Plot average distance error vs. sunrise, sunset available over ALL DAYS.
sun_type_labels = ['Both', 'Sunrise Only', 'Sunset Only', 'Neither']
sun_type_distances = [[] for x in range(len(sun_type_labels))]
cbm_sun_type_distances = [[] for x in range(len(sun_type_labels))]
sun_type_stdevs = [0 for x in range(len(sun_type_labels))] #
cbm_sun_type_stdevs = [0 for x in range(len(sun_type_labels))] #
sun_type_num_data_pts = [0] * len(sun_type_labels) #
cbm_sun_type_num_data_pts = [0] * len(sun_type_labels) #
for i in range(len(days)):
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
        if len(distance_errs) >= 2:
            sun_type_stdevs[sIdx] = statistics.stdev(distance_errs) #
        else:
            sun_type_stdevs[sIdx] = 0 #
        sun_type_num_data_pts[sIdx] += len(distance_errs) #
    else:
        sun_type_distances[sIdx] = 0

for sIdx, distance_errs in enumerate(cbm_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_sun_type_distances[sIdx] = statistics.mean(distance_errs)
        if len(distance_errs) >= 2:
            cbm_sun_type_stdevs[sIdx] = statistics.stdev(distance_errs) #
        else:
            cbm_sun_type_stdevs[sIdx] = 0 #
        cbm_sun_type_num_data_pts[sIdx] += len(distance_errs) #
    else:
        cbm_sun_type_distances[sIdx] = 0

#bar(sun_type_labels, sun_type_distances, 'Median Distance Error (km)', 'Sunrise and sunset in frame?', sun_type_labels, 'Median Error (km) Over All Days vs. Sunrise / Sunset In Frame', 'sun_in_frame.png', sun_type_stdevs)
bar(sun_type_labels, cbm_sun_type_distances, 'Avg. Distance Error (km)', 'Sunrise and sunset in frame?', sun_type_labels, 'Avg. Error (km) Over All Days vs. Sunrise / Sunset In Frame', 'cbm_sun_in_frame.png', cbm_sun_type_stdevs)
print('SUN OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_sun_type_num_data_pts)) #

# Plot average distance error vs. season over ALL DAYS.
season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
season_distances = [[] for x in range(len(season_labels))]
cbm_season_distances = [[] for x in range(len(season_labels))]
season_rmses = [0 for x in range(len(season_labels))] #
cbm_season_rmses = [0 for x in range(len(season_labels))] #
season_num_data_pts = [0] * len(season_labels) #
cbm_season_num_data_pts = [0] * len(season_labels) #
for i in range(len(days)):
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
        season_distances[sIdx] = statistics.median(distance_errs)
        season_rmses[sIdx] = median_rmse(distance_errs) #
        season_num_data_pts[sIdx] += len(distance_errs) #
    else:
        season_distances[sIdx] = 0

for sIdx, distance_errs in enumerate(cbm_season_distances):
    if len(distance_errs) > 0:
        cbm_season_distances[sIdx] = statistics.median(distance_errs)
        cbm_season_rmses[sIdx] = median_rmse(distance_errs) #
        cbm_season_num_data_pts[sIdx] += len(distance_errs) #
    else:
        cbm_season_distances[sIdx] = 0

#bar(season_labels, season_distances, 'Median Distance Error (km)', 'Season', season_labels, 'Median Error (km) Over All Days vs. Season', 'season.png', season_rmses) #
bar(season_labels, cbm_season_distances, 'Median Distance Error (km)', 'Season', season_labels, 'Median Error (km) Over All Days vs. Season', 'cbm_season.png', cbm_season_rmses) #
print('SEASON OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_season_num_data_pts)) #

# Plot average distance error vs. intervals over ALL PLACES.
# Only using CBM model for now.
cbm_median_bucket_distances = [[] for x in range(len(buckets))]
cbm_density_bucket_distances = [[] for x in range(len(buckets))]
ransac_bucket_distances = [[] for x in range(len(buckets))]

cbm_median_bucket_rmses = [0 for x in range(len(buckets))] #
cbm_median_bucket_num_data_pts = [0] * len(buckets) #
cbm_density_bucket_rmses = [0 for x in range(len(buckets))] #
cbm_density_bucket_num_data_pts = [0] * len(buckets) #
ransac_bucket_rmses = [0 for x in range(len(buckets))] #
ransac_bucket_num_data_pts = [0] * len(buckets) #
for key in intervals:
    for bIdx, bucket in enumerate(buckets):
        if intervals[key] < bucket + 5:
            break

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_bucket_distances[bIdx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_bucket_distances[bIdx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], ransac_locations[key][0], ransac_locations[key][1])
    ransac_bucket_distances[bIdx].append(ransac_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_bucket_distances):
    if len(distance_errs) > 0:
        cbm_median_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_median_bucket_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_median_bucket_num_data_pts[bdIdx] += 1 #
    else:
        cbm_median_bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_bucket_distances):
    if len(distance_errs) > 0:
        cbm_density_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_density_bucket_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_density_bucket_num_data_pts[bdIdx] += 1 #
    else:
        cbm_density_bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(ransac_bucket_distances):
    if len(distance_errs) > 0:
        ransac_bucket_distances[bdIdx] = statistics.median(distance_errs)
        ransac_bucket_rmses[bdIdx] = median_rmse(distance_errs)  #
        ransac_bucket_num_data_pts[bdIdx] += 1 #
    else:
        ransac_bucket_distances[bdIdx] = 0

bar(buckets, cbm_median_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using Median vs. Photo Interval (min)', 'cbm_interval_median_places.png', cbm_median_bucket_rmses)
bar(buckets, cbm_density_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using Gaussian KDE vs. Photo Interval (min)', 'cbm_interval_density_places.png', cbm_density_bucket_rmses)
bar(buckets, ransac_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using RANSAC vs. Photo Interval (min)', 'cbm_interval_ransac_places.png', ransac_bucket_rmses)
print('INTERVAL OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_bucket_num_data_pts)) #
print('INTERVAL OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_bucket_num_data_pts)) #
print('INTERVAL OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_bucket_num_data_pts)) #

# Plot average distance error vs. percentage of days with sunrise and sunset visible over ALL PLACES.
# Only using CBM model for now.
buckets = list(range(0, 100, 10)) # 10% buckets
bucket_labels = [str(x) + '-' + str(x + 10) for x in buckets]
cbm_median_sun_type_distances = [[] for x in range(len(buckets))]
cbm_density_sun_type_distances = [[] for x in range(len(buckets))]
ransac_sun_type_distances = [[] for x in range(len(buckets))]

cbm_median_sun_type_rmses = [0 for x in range(len(buckets))] #
cbm_median_sun_type_num_data_pts = [0] * len(buckets) #
cbm_density_sun_type_rmses = [0 for x in range(len(buckets))] #
cbm_density_sun_type_num_data_pts = [0] * len(buckets) #
ransac_sun_type_rmses = [0 for x in range(len(buckets))] #
ransac_sun_type_num_data_pts = [0] * len(buckets) #

for key in intervals:
    for bIdx, bucket in enumerate(buckets):
        if sun_visibles[key][0] * 100 < bucket + 10:
            break

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_sun_type_distances[bIdx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_sun_type_distances[bIdx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], ransac_locations[key][0], ransac_locations[key][1])
    ransac_sun_type_distances[bIdx].append(ransac_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_median_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
        cbm_median_sun_type_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_median_sun_type_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        cbm_median_sun_type_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_density_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
        cbm_density_sun_type_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_density_sun_type_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        cbm_density_sun_type_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(ransac_sun_type_distances):
    if len(distance_errs) > 0:
        ransac_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
        ransac_sun_type_rmses[bdIdx] = median_rmse(distance_errs)  #
        ransac_sun_type_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        ransac_sun_type_distances[bdIdx] = 0

bar(buckets, cbm_median_sun_type_distances, 'Median Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using Median vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_median_places.png', cbm_median_sun_type_rmses)
bar(buckets, cbm_density_sun_type_distances, 'Median Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using Gaussian KDE vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_density_places.png', cbm_density_sun_type_rmses)
bar(buckets, ransac_sun_type_distances, 'Median Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using RANSAC vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_ransac_places.png', ransac_sun_type_rmses)
print('SUN OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_sun_type_num_data_pts)) #
print('SUN OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_sun_type_num_data_pts)) #
print('SUN OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_sun_type_num_data_pts)) #

# Average distance error vs. latitude over ALL PLACES.
buckets = list(range(-90, 90, 10)) # 10 degree buckets
bucket_labels = [str(x) + '-' + str(x + 10) for x in buckets]

cbm_median_lat_distances = [[] for x in range(len(buckets))]
cbm_density_lat_distances = [[] for x in range(len(buckets))]
ransac_lat_distances = [[] for x in range(len(buckets))]

cbm_median_lat_rmses = [0 for x in range(len(buckets))] #
cbm_median_lat_num_data_pts = [0] * len(buckets) #
cbm_density_lat_rmses = [0 for x in range(len(buckets))] #
cbm_density_lat_num_data_pts = [0] * len(buckets) #
ransac_lat_rmses = [0 for x in range(len(buckets))] #
ransac_lat_num_data_pts = [0] * len(buckets) #

for key in lats:
    median_idx = len(buckets) - 1
    density_idx = len(buckets) - 1
    ransac_idx = len(buckets) - 1

    for bIdx, bucket in enumerate(buckets):
        if cbm_median_locations[key][0] < bucket + 10:
            median_idx = min(bIdx, median_idx)
        if cbm_density_locations[key][0] < bucket + 10:
            density_idx = min(bIdx, density_idx)
        if ransac_locations[key][0] < bucket + 10:
            ransac_idx = min(bIdx, ransac_idx)

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_lat_distances[median_idx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_lat_distances[density_idx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], ransac_locations[key][0], ransac_locations[key][1])
    ransac_lat_distances[ransac_idx].append(ransac_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_lat_distances):
    if len(distance_errs) > 0:
        cbm_median_lat_distances[bdIdx] = statistics.median(distance_errs)
        cbm_median_lat_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_median_lat_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        cbm_median_lat_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_lat_distances):
    if len(distance_errs) > 0:
        cbm_density_lat_distances[bdIdx] = statistics.median(distance_errs)
        cbm_density_lat_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_density_lat_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        cbm_density_lat_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(ransac_lat_distances):
    if len(distance_errs) > 0:
        ransac_lat_distances[bdIdx] = statistics.median(distance_errs)
        ransac_lat_rmses[bdIdx] = median_rmse(distance_errs)  #
        ransac_lat_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        ransac_lat_distances[bdIdx] = 0

bar(buckets, cbm_median_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using Median vs. Latitude', 'cbm_lat_median_places.png', cbm_median_lat_rmses)
bar(buckets, cbm_density_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using Gaussian KDE vs. Latitude', 'cbm_lat_density_places.png', cbm_density_lat_rmses)
bar(buckets, ransac_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using RANSAC vs. Latitude', 'cbm_lat_ransac_places.png', ransac_lat_rmses)
print('LAT OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_lat_num_data_pts)) #
print('LAT OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_lat_num_data_pts)) #
print('LAT OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_lat_num_data_pts)) #

# Average distance error vs. longitude over ALL PLACES.
buckets = list(range(-180, 180, 20)) # 20 degree buckets
bucket_labels = [str(x) + '-' + str(x + 20) for x in buckets]

cbm_median_lng_distances = [[] for x in range(len(buckets))]
cbm_density_lng_distances = [[] for x in range(len(buckets))]
ransac_lng_distances = [[] for x in range(len(buckets))]

cbm_median_lng_rmses = [0 for x in range(len(buckets))] #
cbm_median_lng_num_data_pts = [0] * len(buckets) #
cbm_density_lng_rmses = [0 for x in range(len(buckets))] #
cbm_density_lng_num_data_pts = [0] * len(buckets) #
ransac_lng_rmses = [0 for x in range(len(buckets))] #
ransac_lng_num_data_pts = [0] * len(buckets) #

for key in lngs:
    median_idx = len(buckets) - 1
    density_idx = len(buckets) - 1
    ransac_idx = len(buckets) - 1

    for bIdx, bucket in enumerate(buckets):
        if cbm_median_locations[key][1] <= bucket + 20:
            median_idx = min(bIdx, median_idx)
        if cbm_density_locations[key][1] <= bucket + 20:
            density_idx = min(bIdx, density_idx)
        if ransac_locations[key][1] <= bucket + 20:
            ransac_idx = min(bIdx, ransac_idx)

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_lng_distances[median_idx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_lng_distances[density_idx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], ransac_locations[key][0], ransac_locations[key][1])
    ransac_lng_distances[ransac_idx].append(ransac_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_lng_distances):
    if len(distance_errs) > 0:
        cbm_median_lng_distances[bdIdx] = statistics.median(distance_errs)
        cbm_median_lng_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_median_lng_num_data_pts[bdIdx] += len(distance_errs)  #
    else:
        cbm_median_lng_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_lng_distances):
    if len(distance_errs) > 0:
        cbm_density_lng_distances[bdIdx] = statistics.median(distance_errs)
        cbm_density_lng_rmses[bdIdx] = median_rmse(distance_errs)  #
        cbm_density_lng_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        cbm_density_lng_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(ransac_lng_distances):
    if len(distance_errs) > 0:
        ransac_lng_distances[bdIdx] = statistics.median(distance_errs)
        ransac_lng_rmses[bdIdx] = median_rmse(distance_errs)  #
        ransac_lng_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        ransac_lng_distances[bdIdx] = 0

bar(buckets, cbm_median_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using Median vs. Longitude', 'cbm_lng_median_places.png', cbm_median_lng_rmses)
bar(buckets, cbm_density_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using Gaussian KDE vs. Longitude', 'cbm_lng_density_places.png', cbm_density_lng_rmses)
bar(buckets, ransac_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using RANSAC vs. Longitude', 'cbm_lng_ransac_places.png', ransac_lng_rmses)
print('LNG OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_lng_num_data_pts)) #
print('LNG OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_lng_num_data_pts)) #
print('LNG OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_lng_num_data_pts)) #

green = 0
sunrise_only = 0
sunset_only = 0
black = 0

avg_interval = 0

for i in range(len(days)):
    if days[i].sunrise_in_frames and days[i].sunset_in_frames:
        green += 1
    elif days[i].sunrise_in_frames and not days[i].sunset_in_frames:
        sunrise_only += 1
    elif not days[i].sunrise_in_frames and days[i].sunset_in_frames:
        sunset_only += 1
    else:
        black += 1

    avg_interval += days[i].interval_min

print('Green / Sunrise Only / Sunset Only / Black: {}, {}, {}, {}'.format(green, sunrise_only, sunset_only, black))
print('')
print('Average Interval (min): ' + str(avg_interval / len(days)))
sys.stdout.flush()

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
print('RANSAC Avg. Distance Error: {:.6f}'.format(statistics.mean(ransac_distances)))
print('CBM Means Median Distance Error: {:.6f}'.format(statistics.median(cbm_mean_distances)))
print('CBM Medians Median Distance Error: {:.6f}'.format(statistics.median(cbm_median_distances)))
print('CBM Density Median Distance Error: {:.6f}'.format(statistics.median(cbm_density_distances)))
print('RANSAC Median Distance Error: {:.6f}'.format(statistics.median(ransac_distances)))
print('CBM Means Max Distance Error: {:.6f}'.format(max(cbm_mean_distances)))
print('CBM Means Min Distance Error: {:.6f}'.format(min(cbm_mean_distances)))
print('CBM Medians Max Distance Error: {:.6f}'.format(max(cbm_median_distances)))
print('CBM Medians Min Distance Error: {:.6f}'.format(min(cbm_median_distances)))
print('CBM Density Max Distance Error: {:.6f}'.format(max(cbm_density_distances)))
print('CBM Density Min Distance Error: {:.6f}'.format(min(cbm_density_distances)))
print('RANSAC Max Distance Error: {:.6f}'.format(max(ransac_distances)))
print('RANSAC Min Distance Error: {:.6f}'.format(min(ransac_distances)))
print('')
print('Means Avg. Longitude Error: {:.6f}'.format(statistics.mean(mean_longitude_err)))
print('Medians Avg. Longitude Error: {:.6f}'.format(statistics.mean(median_longitude_err)))
print('Density Avg. Longitude Error: {:.6f}'.format(statistics.mean(density_longitude_err)))
print('RANSAC Avg. Longitude Error: {:.6f}'.format(statistics.mean(ransac_longitude_err)))
print('Means Median Longitude Error: {:.6f}'.format(statistics.median(mean_longitude_err)))
print('Medians Median Longitude Error: {:.6f}'.format(statistics.median(median_longitude_err)))
print('Density Median Longitude Error: {:.6f}'.format(statistics.median(density_longitude_err)))
print('RANSAC Median Longitude Error: {:.6f}'.format(statistics.median(ransac_longitude_err)))
print('Means Max Longitude Error: {:.6f}'.format(max(mean_longitude_err)))
print('Means Min Longitude Error: {:.6f}'.format(min(mean_longitude_err)))
print('Medians Max Longitude Error: {:.6f}'.format(max(median_longitude_err)))
print('Medians Min Longitude Error: {:.6f}'.format(min(median_longitude_err)))
print('Density Max Longitude Error: {:.6f}'.format(max(density_longitude_err)))
print('Density Min Longitude Error: {:.6f}'.format(min(density_longitude_err)))
print('RANSAC Max Longitude Error: {:.6f}'.format(max(ransac_longitude_err)))
print('RANSAC Min Longitude Error: {:.6f}'.format(min(ransac_longitude_err)))
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
print('RANSAC Avg. Latitude Error: {:.6f}'.format(statistics.mean(ransac_latitude_err)))
print('CBM Means Median Latitude Error: {:.6f}'.format(statistics.median(cbm_mean_latitude_err)))
print('CBM Medians Median Latitude Error: {:.6f}'.format(statistics.median(cbm_median_latitude_err)))
print('CBM Density Median Latitude Error: {:.6f}'.format(statistics.median(cbm_density_latitude_err)))
print('RANSAC Median Latitude Error: {:.6f}'.format(statistics.median(ransac_latitude_err)))
print('CBM Means Max Latitude Error: {:.6f}'.format(max(cbm_mean_latitude_err)))
print('CBM Means Min Latitude Error: {:.6f}'.format(min(cbm_mean_latitude_err)))
print('CBM Medians Max Latitude Error: {:.6f}'.format(max(cbm_median_latitude_err)))
print('CBM Medians Min Latitude Error: {:.6f}'.format(min(cbm_median_latitude_err)))
print('CBM Density Max Latitude Error: {:.6f}'.format(max(cbm_density_latitude_err)))
print('CBM Density Min Latitude Error: {:.6f}'.format(min(cbm_density_latitude_err)))
print('RANSAC Max Longitude Error: {:.6f}'.format(max(ransac_latitude_err)))
print('RANSAC Min Longitude Error: {:.6f}'.format(min(ransac_latitude_err)))
print('')
sys.stdout.flush()