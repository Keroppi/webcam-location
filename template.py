#!/srv/glusterfs/vli/.pyenv/shims/python

import os, argparse, datetime, time, math, sys, random, statistics, pickle, pandas as pd

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from simple_day import SimpleDay

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

sunrises = {}
sunsets = {}

for place in predictions: # predictions is a dictionary with each place as a key
    if place not in sunrises:
        sunrises[place] = []
        sunsets[place] = []

    for d_idx, day in enumerate(predictions[place]): # each place contains a list of SimpleDay objects which are sorted by time
        sunrises[place].append(predictions[place][d_idx].sunrise)
        sunsets[place].append(predictions[place][d_idx].sunset)

# Compute solar noon and day length.
solar_noons = {}
day_lengths = {}
for place in sunrises:
    if place not in solar_noons:
        solar_noons[place] = []
        day_lengths[place] = []

    for sunrise, sunset in zip(sunrises[place], sunsets[place]):
        solar_noon = (sunset - sunrise) / 2 + sunrise
        day_length = (sunset - sunrise).total_seconds()

        solar_noons[place].append(solar_noon)
        day_lengths[place].append(day_length)

# Compute longitude.
longitudes = {}
for place in solar_noons:
    if place not in longitudes:
        longitudes[place] = []

    for idx, solar_noon in enumerate(solar_noons[place]):
        utc_diff = predictions[place][idx].mali_solar_noon - solar_noon
        hours_time_zone_diff = predictions[place][idx].time_offset / 60 / 60
        hours_utc_diff = utc_diff.total_seconds() / 60 / 60
        lng = (hours_utc_diff + hours_time_zone_diff) * 15

        # What to do if outside [-180, 180] range?
        if lng < -180:
            lng += 360
        elif lng > 180:
            lng -= 360

        longitudes[place].append(lng)

# CBM Model for Latitude
def cbm(day_of_year, day_length_hours):
    theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
    phi = math.asin(0.39795 * math.cos(theta))
    cbm_lat = 180 / math.pi * math.atan(math.cos(phi) / math.sin(phi) * math.cos(-math.pi / 24 * (day_length_hours - 24)))

    return cbm_lat

# Compute latitude.
latitudes = {}
for place in day_lengths:
    if place not in latitudes:
        latitudes[place] = []

    for idx, day_length in enumerate(day_lengths[place]):
        day_length_hours = day_length / 3600 # seconds to hours

        ts = pd.Series(pd.to_datetime([str(predictions[place][idx].date)]))
        day_of_year = int(ts.dt.dayofyear) # day_of_year from 1 to 365, inclusive

        # CBM Model
        lat = cbm(day_of_year, day_length_hours)

        latitudes[place].append(lat)

def compute_haversine_distance(lat1, lng1, lat2, lng2):  # Haversine distance, less accurate, returns km
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

def compute_distance(lat1, lng1, lat2, lng2):  # Vincenty distance, returns km
    if lng1 == -180:
        lng1 = 180

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lng1_rad = math.radians(lng1)
    lng2_rad = math.radians(lng2)

    a = 6378.137  # km
    b = 6356.752314245  # km
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
            iterations += 1001  # coincident points

        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - math.pow(sin_alpha, 2)

        if cos_sq_alpha != 0:
            cos_2_sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos_sq_alpha
        else:  # Equatorial line
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

for place in longitudes:
    # Use simple mean as a guess.
    pred_lat = statistics.mean(latitudes[place])
    pred_lng = statistics.mean(longitudes[place])

    actual_lat = predictions[place][0].lat
    actual_lng = predictions[place][0].lng

    error = compute_distance(pred_lat, pred_lng, actual_lat, actual_lng) # km
    print(place + ': ' + str(error) + ' km')
