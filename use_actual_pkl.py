#!/srv/glusterfs/vli/.pyenv/shims/python

import matplotlib
matplotlib.use('agg')

import os, argparse, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, pickle, collections, copy, scipy
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
#from sklearn.mixture import BayesianGaussianMixture
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from simple_day import SimpleDay
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import picos as pic

print('Starting predict location.')
print('Bandwidth: {}'.format(constants.BANDWIDTH))
print('RANSAC Inlier: {}'.format(constants.INLIER_THRESHOLD))
print('Particle Inlier: {}'.format(constants.AZIMUTHAL_INLIER_THRESHOLD))
print('Big-M: {}'.format(constants.BIGM))
print('GMM Mahalanobis: {}'.format(constants.MAHALANOBIS_INLIER_THRESHOLD))
print('Particle Mahalanobis: {}'.format(constants.AZIMUTHAL_MAHALANOBIS_INLIER_THRESHOLD))
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
    days += predictions[place]

    for d_idx, day in enumerate(predictions[place]):
        sunrises.append(predictions[place][d_idx].sunrise)
        sunsets.append(predictions[place][d_idx].sunset)

# Compute solar noon and day length.
solar_noons = []
day_lengths = []
for d_idx, (sunrise, sunset) in enumerate(zip(sunrises, sunsets)):
    # Threshold sunrise to be at earliest midnight.
    if sunrise.date() < days[d_idx].sunrise.date():
        sunrise = datetime.datetime.combine(sunrise, datetime.time.min)
        print('WARNING - Sunrise truncated to 12:00 same day.')
    # Threshold sunset to be at latest 2 AM the next day.
    if sunset > datetime.datetime.combine(days[d_idx].sunrise.date() + datetime.timedelta(days=1), datetime.time(2, 0, 0)):
        sunset = datetime.datetime.combine(days[d_idx].sunrise.date() + datetime.timedelta(days=1), datetime.time(2, 0, 0))
        print('WARNING - Sunset truncated to 2:00 next day.')

    solar_noon = (sunset - sunrise) / 2 + sunrise

    # Latest solar noon in the world is in western China at 15:10, so truncate any time past ~15:20
    if solar_noon.hour > 15 or (solar_noon.hour == 15 and solar_noon.minute >= 20):
        solar_noon = solar_noon.replace(hour=15, minute=20, second=0, microsecond=0)
        print('WARNING - Solar noon truncated to 15:20.')
    # Earliest solar noon in the world is in Greenland around 9:32 AM, so truncate any time before ~9:28 AM.
    # https://www.timeanddate.com/sun/@81.5053,-12.1311?month=11&year=2017
    if solar_noon.hour < 9 or (solar_noon.hour == 9 and solar_noon.minute <= 28):
        solar_noon = solar_noon.replace(hour=9, minute=28, second=0, microsecond=0)
        print('WARNING - Solar noon truncated to 9:28.')

    sys.stdout.flush()

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
        #print('WARNING - lng below -180')
        #print(days[d_idx].place)
        #sys.stdout.flush()
    elif lng > 180:
        lng -= 360
        #print('WARNING - lng over 180')
        #print(days[d_idx].place)
        #sys.stdout.flush()

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

    p = 0.26667 # Constant for data from timeanddate.com (sunrise and sunset is when top of sun disk at horizon)
                # https://www.timeanddate.com/sun/help
                # https://www.ikhebeenvraag.be/mediastorage/FSDocument/171/Forsythe+-+A+model+comparison+for+daylength+as+a+function+of+latitude+and+day+of+year+-+1995.pdf
    p_value = math.sin(p * math.pi / 180)
    cbm_lat = 180 / math.pi * math.atan(math.cos(phi) / math.sin(phi) * (math.cos(-math.pi / 24 * (day_length_hours - 24)) - p_value))

    return cbm_lat

# Compute latitude.
#latitudes_weird = []
latitudes = []
cbm_latitudes = []
for d_idx, day_length in enumerate(day_lengths):
    day_length_hours = day_length / 3600

    ts = pd.Series(pd.to_datetime([str(days[d_idx].sunrise.date())]))
    day_of_year = int(ts.dt.dayofyear) # day_of_year from 1 to 365, inclusive

    # Brock Model
    declination = math.radians(23.45) * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    lat = math.degrees(math.atan(-math.cos(math.radians(15 * day_length_hours / 2)) / math.tan(declination)))

    # CBM Model
    cbm_lat = cbm(day_of_year, day_length_hours)

    '''
    # Check if they're in different (north / south) hemispheres.
    if (cbm_lat > 0 and days[d_idx].lat < 0) or (cbm_lat < 0 and days[d_idx].lat > 0):
        if math.fabs(day_length_hours - 12) < 1:
            cbm_lat *= -1
            # cbm_latitudes.append(-cbm_lat)

    # Check if they're in different (north / south) hemispheres.
    if (lat > 0 and days[d_idx].lat < 0) or (lat < 0 and days[d_idx].lat > 0):
        if math.fabs(day_length_hours - 12) < 1:
            lat *= -1
            #latitudes.append(-lat)
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


# Store which day of the year and day length for each place.
day_lens = {}
#days_of_year = {}
for i in range(len(days)):
    #if days_of_year.get(days[i].place) is None:
    #    days_of_year[days[i].place] = []

    if day_lens.get(days[i].place) is None:
        day_lens[days[i].place] = []

    #ts = pd.Series(pd.to_datetime([str(days[i].date)]))
    #day_of_year = int(ts.dt.dayofyear)  # day_of_year from 1 to 365, inclusive

    #days_of_year[days[i].place].append(days_of_year)
    day_lens[days[i].place].append(day_lengths[i])


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

    #if day_lengths[i] / 3600 > 11 and day_lengths[i] / 3600 < 13:
    #    lats[days[i].place].append(-latitudes[i])
    #    cbm_lats[days[i].place].append(-cbm_latitudes[i])
    #    lngs[days[i].place].append(longitudes[i])



def compute_haversine_distance(lat1, lng1, lat2, lng2): # km
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


def compute_distance(lat1, lng1, lat2, lng2): # km
    if lng1 == -180:
        lng1 = 180

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lng1_rad = math.radians(lng1)
    lng2_rad = math.radians(lng2)

    a = 6378.137 # km
    b = 6356.752314245 # km
    f = 1 / 298.257223563

    L = lng2_rad - lng1_rad

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

        if sin_sigma == 0:
            iterations += 1001  # coincident points #

        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)  #

        sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - math.pow(sin_alpha, 2)

        if cos_sq_alpha != 0:
            cos_2_sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha  #
        else:  # Equatorial line #
            cos_2_sigma_m = 0

        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        new_lamb = L + (1 - C) * f * sin_alpha * (
            sigma + C * sin_sigma * (cos_2_sigma_m + C * cos_sigma * (-1 + 2 * math.pow(cos_2_sigma_m, 2))))

        if antimeridian:
            iteration_check = math.fabs(new_lamb) - math.pi
        else:
            iteration_check = math.fabs(new_lamb)

        if iteration_check > math.pi:
            iterations += 1001

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

actual_locations = {}
for i in range(len(days)):
    if actual_locations.get(days[i].place) is None:
        actual_locations[days[i].place] = (days[i].lat, days[i].lng)
    else:
        continue

mean_locations = {}
median_locations = {}

cbm_mean_locations = {}
cbm_median_locations = {}

for key in lats:
    mean_locations[key] = (statistics.mean(lats[key]), statistics.mean(lngs[key]))
    median_locations[key] = (statistics.median(lats[key]), statistics.median(lngs[key]))

    cbm_mean_locations[key] = (statistics.mean(cbm_lats[key]), statistics.mean(lngs[key]))
    cbm_median_locations[key] = (statistics.median(cbm_lats[key]), statistics.median(lngs[key]))

cbm_mean_distances = []
cbm_median_distances = []

for place in lats:
    actual_lat = actual_locations[place][0] #days[i].lat
    actual_lng = actual_locations[place][1] #days[i].lng

    cbm_mean_pred_lat = cbm_mean_locations[place][0]
    cbm_mean_pred_lng = cbm_mean_locations[place][1]
    cbm_median_pred_lat = cbm_median_locations[place][0]
    cbm_median_pred_lng = cbm_median_locations[place][1]

    cbm_mean_distance = compute_distance(actual_lat, actual_lng, cbm_mean_pred_lat, cbm_mean_pred_lng)
    cbm_mean_distances.append(cbm_mean_distance)
    cbm_median_distance = compute_distance(actual_lat, actual_lng, cbm_median_pred_lat, cbm_median_pred_lng)
    cbm_median_distances.append(cbm_median_distance)

print('CBM Means Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_mean_distances)))
print('CBM Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_median_distances)))
print('CBM Means Median Distance Error: {:.6f}'.format(statistics.median(cbm_mean_distances)))
print('CBM Medians Median Distance Error: {:.6f}'.format(statistics.median(cbm_median_distances)))
print('CBM Means Max Distance Error: {:.6f}'.format(max(cbm_mean_distances)))
print('CBM Means Min Distance Error: {:.6f}'.format(min(cbm_mean_distances)))
print('CBM Medians Max Distance Error: {:.6f}'.format(max(cbm_median_distances)))
print('CBM Medians Min Distance Error: {:.6f}'.format(min(cbm_median_distances)))

sys.stdout.flush()