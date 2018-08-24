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
print('Discard Days From Equinox: {}'.format(constants.EQUINOX_DISCARD_DAYS))
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

def days_from_equinox(date):
    spring_2017_diff = math.fabs((date - constants.VERNAL_EQUINOX_2017).total_seconds())
    autumn_2017_diff = math.fabs((date - constants.AUTUMNAL_EQUINOX_2017).total_seconds())
    spring_2018_diff = math.fabs((date - constants.VERNAL_EQUINOX_2018).total_seconds())
    autumn_2018_diff = math.fabs((date - constants.AUTUMNAL_EQUINOX_2018).total_seconds())

    days_away = min([spring_2017_diff, autumn_2017_diff, spring_2018_diff, autumn_2018_diff]) / 60 / 60 / 24

    return days_away


def days_from_solstice(date):
    winter_2016_diff = math.fabs((date - constants.WINTER_SOLSTICE_2016).total_seconds())
    summer_2017_diff = math.fabs((date - constants.SUMMER_SOLSTICE_2017).total_seconds())
    winter_2017_diff = math.fabs((date - constants.WINTER_SOLSTICE_2017).total_seconds())
    summer_2018_diff = math.fabs((date - constants.SUMMER_SOLSTICE_2018).total_seconds())

    days_away = min([winter_2016_diff, summer_2017_diff, winter_2017_diff, summer_2018_diff]) / 60 / 60 / 24

    return days_away


sunrises = []
sunsets = []
equinox_offsets = []
solstice_offsets = []
days = []
for place in predictions:
    #days += predictions[place]

    for d_idx, day in enumerate(predictions[place]):
        equinox_days = days_from_equinox(predictions[place][d_idx].sunrise - datetime.timedelta(seconds=predictions[place][d_idx].time_offset))
        solstice_days = days_from_solstice(predictions[place][d_idx].sunrise - datetime.timedelta(seconds=predictions[place][d_idx].time_offset))

        # VLI
        #if equinox_days < constants.EQUINOX_DISCARD_DAYS: # ? weeks
        #    continue

        days += [day]

        sunrises.append(predictions[place][d_idx].sunrise)
        sunsets.append(predictions[place][d_idx].sunset)

        equinox_offsets.append(equinox_days)
        solstice_offsets.append(solstice_days)

print('Number of days: {}'.format(len(days)))

# Compute solar noon and day length.
solar_noons = []
day_lengths = []
for d_idx, (sunrise, sunset) in enumerate(zip(sunrises, sunsets)):
    # Threshold sunrise to be at earliest midnight.
    if sunrise.date() < days[d_idx].sunrise.date():
        sunrise = datetime.datetime.combine(sunrise, datetime.time.min)
    # Threshold sunset to be at latest 2 AM the next day.
    if sunset > datetime.datetime.combine(days[d_idx].sunrise.date() + datetime.timedelta(days=1), datetime.time(2, 0, 0)):
        sunset = datetime.datetime.combine(days[d_idx].sunrise.date() + datetime.timedelta(days=1), datetime.time(2, 0, 0))

    solar_noon = (sunset - sunrise) / 2 + sunrise

    # Latest solar noon in the world is in western China at 15:10, so truncate any time past ~15:20
    if solar_noon.hour > 15 or (solar_noon.hour == 15 and solar_noon.minute >= 20):
        solar_noon = solar_noon.replace(hour=15, minute=20, second=0, microsecond=0)
    # Earliest solar noon in the world is in Greenland around 9:32 AM, so truncate any time before ~9:28 AM.
    # https://www.timeanddate.com/sun/@81.5053,-12.1311?month=11&year=2017
    if solar_noon.hour < 9 or (solar_noon.hour == 9 and solar_noon.minute <= 28):
        solar_noon = solar_noon.replace(hour=9, minute=28, second=0, microsecond=0)

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

    # Note:
    # hours_time_zone_diff is lowest 0.5 hours, times 15 which is 7.5 degrees
    # utc_diff is in minute accuracy (multiples of 60), so hours_utc_diff is at best accuracy of 1/60, times 15 is 1/4 = 0.25 degrees!
    # 1/8 degrees are because my solar_noon is an average which can get half minutes

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

def inverse_cbm(day_of_year, latitude):
    theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
    phi = math.asin(0.39795 * math.cos(theta))

    p = 0.26667  # Constant for data from timeanddate.com (sunrise and sunset is when top of sun disk at horizon)
    p_value = math.sin(p * math.pi / 180)
    day_length = 24 - 24 / math.pi * math.acos((p_value + math.sin(latitude * math.pi / 180) * math.sin(phi)) / (math.cos(latitude * math.pi / 180) * math.cos(phi)))

    return day_length

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
days_of_year = {}
phis = {} # absolute value of solar declination in CBM model
for i in range(len(days)):
    if days_of_year.get(days[i].place) is None:
        days_of_year[days[i].place] = []

    if day_lens.get(days[i].place) is None:
        day_lens[days[i].place] = []

    if phis.get(days[i].place) is None:
        phis[days[i].place] = []

    ts = pd.Series(pd.to_datetime([str(days[i].sunrise)]))
    day_of_year = int(ts.dt.dayofyear)  # day_of_year from 1 to 365, inclusive

    days_of_year[days[i].place].append(day_of_year)
    day_lens[days[i].place].append(day_lengths[i])

    theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
    phi = math.asin(0.39795 * math.cos(theta))
    phis[days[i].place].append(math.fabs(phi))


equinox_offs = {}
solstice_offs = {}
for i in range(len(days)):
    if equinox_offs.get(days[i].place) is None:
        equinox_offs[days[i].place] = []

    equinox_offs[days[i].place].append(equinox_offsets[i])

    if solstice_offs.get(days[i].place) is None:
        solstice_offs[days[i].place] = []

    solstice_offs[days[i].place].append(solstice_offsets[i])

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

def azimuthal_equidistant(lat, lng):
    # 0 lat, 0 lng as center
    lat = math.radians(lat)
    lng = math.radians(lng)

    c = math.acos(math.cos(lat) * math.cos(lng))

    if c == 0:
        return (0, 0)

    k_prime = c / math.sin(c)
    x = k_prime * math.cos(lat) * math.sin(lng)
    y = k_prime * math.sin(lat)

    return (x, y)

def azimuthal_equidistant_inverse(x, y):
    c = math.sqrt(x * x + y * y)
    lat = math.degrees(math.asin(y * math.sin(c) / c))
    lng = math.degrees(math.atan2(x * math.sin(c), (c * math.cos(c))))

    return (lat, lng)


# 'day' weighting scales based on days from the equinox
# 'declination' weighting scales based on the solar declination angle (0 to 23.44 degrees)
# https://en.wikipedia.org/wiki/Position_of_the_Sun#Declination_of_the_Sun_as_seen_from_Earth
def equinox_weighting(lats, mode='day'):
    output = {}

    for place in equinox_offs:
        curr_lats = np.array(lats[place])

        if mode == 'day':
            curr_weights = np.array(equinox_offs[place])
        elif mode == 'declination':
            curr_weights = np.array(phis[place])

        numerator = np.dot(curr_lats, curr_weights)
        total_weight = np.sum(curr_weights)
        final_guess = numerator / total_weight
        output[place] = final_guess

    return output

'''
def solstice_weighting(lngs, mode='day'):
    output = {}

    for place in solstice_offs:
        curr_lngs = np.array(lngs[place])

        if mode == 'day':
            curr_weights = np.array(solstice_offs[place])
        elif mode == 'declination':
            curr_weights = np.array(phis[place])

        numerator = np.dot(curr_lngs, curr_weights)
        total_weight = np.sum(curr_weights)
        final_guess = numerator / total_weight
        output[place] = final_guess

    return output
'''

def equinox_solstice_weightings(lats, lngs, mode='day'):
    new_lats = equinox_weighting(lats, mode)
    #new_lngs = solstice_weighting(lngs, mode)

    locations = {}
    for place in lats:
        locations[place] = (new_lats[place], None)

    return locations

cbm_equinox_day_locations = equinox_solstice_weightings(cbm_lats, lngs, 'day')
cbm_equinox_declin_locations = equinox_solstice_weightings(cbm_lats, lngs, 'declination')

def gaussian_mixture(lats, lngs):
    gmm_t0 = time.time()
    #cov_avg = np.zeros((2, 2))
    locations = {}

    for place in lats:
        if len(lats[place]) == 1:
            locations[place] = (lats[place][0], lngs[place][0])
            continue

        points = []
        for lat, lng in zip(lats[place], lngs[place]):
            x, y = azimuthal_equidistant(lat, lng)
            points.append([x, y])

        points = np.array(points)
        #cov = np.cov(points)
        #print(place)
        #print(cov)

        mean_covariance = np.array([0.01018481, 0.13424137]) # if using BayesianGMM for covariance_prior_

        gmms = []
        gmm1 = GaussianMixture(n_components=1, covariance_type='diag').fit(points)
        gmm2 = GaussianMixture(n_components=2, covariance_type='diag').fit(points)
        gmm3 = GaussianMixture(n_components=3, covariance_type='diag').fit(points)
        gmms = [gmm1, gmm2, gmm3]

        bics = []
        bics.append(gmm1.bic(points))
        bics.append(gmm2.bic(points))
        bics.append(gmm3.bic(points))

        aics = []
        aics.append(gmm1.aic(points))
        aics.append(gmm2.aic(points))
        aics.append(gmm3.aic(points))

        # Pick # of clusters based on BIC.
        bic, k_idx1 = min((val, idx) for (idx, val) in enumerate(bics))

        # Pick # of clusters based on AIC.
        aic, k_idx2 = min((val, idx) for (idx, val) in enumerate(aics))

        # Choose least clusters between AIC and BIC.
        k_idx = min(k_idx1, k_idx2)
        gmm = gmms[k_idx]

        #print(place + ' - # clusters - ' + str(k_idx + 1))

        if k_idx > 0: # If there is more than one cluster, pick cluster with most points.
            classes = gmm.predict(points)
            class_counts = collections.Counter(classes)

            cluster_idx = -1
            max_count = -1
            for cluster in class_counts:
                if max_count < class_counts[cluster]:
                    max_count = class_counts[cluster]
                    cluster_idx = cluster
        else:
            cluster_idx = 0

        center = gmm.means_[cluster_idx, :]
        cov = np.diag(gmm.covariances_[cluster_idx, :])

        #cov_avg += cov

        #if k_idx == 0:
        #    print('COV MATRIX BELOW')
        #    print(cov) # Set covariance_prior_ using BayesianGMM?
        #    print(cov[0, 0] / cov[1, 1])

        cov_inv = np.linalg.inv(cov)

        # Calculate Mahalanobis distance from mean to all points... reject points that are too far?
        inliers = []
        for row in range(points.shape[0]):
            m_dist = scipy.spatial.distance.mahalanobis(center, points[row, :], cov_inv)

            if m_dist < constants.MAHALANOBIS_INLIER_THRESHOLD:
                inliers.append(points[row, :])
            #else:
            #    print('Mahalanobis distance: {}'.format(m_dist))

        #print(len(inliers))
        if len(inliers) > 0:
            inliers = np.array(inliers)

            x_star = statistics.mean(inliers[:, 0])
            y_star = statistics.mean(inliers[:, 1])
        else:
            print('WARNING - No inliers found with GMM!')
            x_star = center[0]
            y_star = center[1]


        lat, lng = azimuthal_equidistant_inverse(x_star, y_star)
        locations[place] = (lat, lng)

    #cov_avg /= len(lats)
    #print('COV AVG')
    #print(cov_avg)

    gmm_t1 = time.time()
    print('Calculating GMM time (m): ' + str((gmm_t1 - gmm_t0) / 60))
    sys.stdout.flush()

    return locations

cbm_gmm_locations = gaussian_mixture(cbm_lats, lngs)

def particle_filter(lats, lngs, mahalanobis=False):
    particle_t0 = time.time()

    bigM = constants.BIGM

    particle_locations = {}

    for place in lats:
        if len(lats[place]) == 1:
            particle_locations[place] = (lats[place][0], lngs[place][0])
            continue
        elif len(lats[place]) == 2:
            particle_locations[place] = (statistics.mean(lats[place]), statistics.mean(lngs[place]))
            continue

        transformed_lats = []
        transformed_lngs = []

        # Create the 'prob' variable to contain the problem data
        prob = pic.Problem()
        x_star = prob.add_variable('x*', 2)  # (x, y) coordinate we're trying to find
        z = [] # data points
        b = [0] * len(lngs[place])

        for idx, (lat, lng) in enumerate(zip(lats[place], lngs[place])):
            x, y = azimuthal_equidistant(lat, lng)

            transformed_lats.append(x)
            transformed_lngs.append(y)

            z.append([x, y])
            b[idx] = prob.add_variable('b[{0}]'.format(idx), 1, vtype='binary')  # 0 if inlier, 1 if outlier

        if mahalanobis:
            z = np.array(z)
            scaler = StandardScaler()

            scaler.fit(z)
            z = scaler.transform(z)
            z = np.ndarray.tolist(z)


        z = pic.new_param('z', tuple(z))

        # print(z[0])
        # print(z[1])
        # print(z[0] - z[1])
        # print((z[0] - z[1]).size)

        if not mahalanobis:
            epsilon = constants.AZIMUTHAL_INLIER_THRESHOLD
        else:
            epsilon = constants.AZIMUTHAL_MAHALANOBIS_INLIER_THRESHOLD

        prob.add_list_of_constraints(
            [abs(z[i] - x_star) <= epsilon + b[i] * bigM for i in range(0, len(lngs[place]))], 'i',
            '1...N')

        prob.set_objective('min', pic.sum(b, 'i', '1..N'))

        #print(prob)
        sol = prob.solve(solver='mosek', verbose=0)
        #print(x_star)  # optimal value of x
        #print(azimuthal_equidistant_inverse(x_star[0].value[0], x_star[1].value[0]))
        #particle_lat, particle_lng = azimuthal_equidistant_inverse(x_star[0].value[0], x_star[1].value[0])

        # Find inliers and compute average point.
        inlier_set = [b[i].value[0] < 0.5 for i in range(len(b))]
        inlier_transformed_lats = [transformed_lats[idx] for idx, valid in enumerate(inlier_set) if valid]
        inlier_transformed_lngs = [transformed_lngs[idx] for idx, valid in enumerate(inlier_set) if valid]
        #inlier_transformed_lats += [x_star[0].value[0]]
        #inlier_transformed_lngs += [x_star[1].value[0]]
        transformed_particle_lat = statistics.mean(inlier_transformed_lats)
        transformed_particle_lng = statistics.mean(inlier_transformed_lngs)

        #if mahalanobis: # Revert scaling.
        #    guess = np.array([[transformed_particle_lat, transformed_particle_lng]])
        #    guess = scaler.inverse_transform(guess)
        #    transformed_particle_lat = guess[:, 0]
        #    transformed_particle_lng = guess[:, 1]

        particle_lat, particle_lng = azimuthal_equidistant_inverse(transformed_particle_lat, transformed_particle_lng)
        #print((particle_lat, particle_lng))

        particle_locations[place] = (particle_lat, particle_lng)

    particle_t1 = time.time()
    print('Calculating Particle Filter time (m): ' + str((particle_t1 - particle_t0) / 60))
    sys.stdout.flush()

    return particle_locations

# VLI
brock_particle_locations = particle_filter(lats, lngs)
cbm_particle_locations = particle_filter(cbm_lats, lngs)
cbm_particle_mahalanobis_locations = particle_filter(cbm_lats, lngs, True)

#brock_particle_locations = cbm_gmm_locations
#cbm_particle_locations = cbm_gmm_locations
#cbm_particle_mahalanobis_locations = cbm_gmm_locations

def ransac(lats, lngs, actual=False):
    ransac_t0 = time.time()
    ransacs = {}
    inlier_dict = {}

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

        if not actual:
            inlier_dict[place] = max_inliers
        else:
            inlier_dict[place] = len(inliers[-1]) - 1 # just use the actual location inliers and subtract the actual location itself

        ransacs[place] = (statistics.mean([x[0] for x in inliers[max_idx]]), statistics.mean([x[1] for x in inliers[max_idx]]))

    ransac_t1 = time.time()
    print('Calculating RANSAC time (m): ' + str((ransac_t1 - ransac_t0) / 60))
    sys.stdout.flush()

    return (ransacs, inlier_dict)

brock_ransac_locations, _ = ransac(lats, lngs)
cbm_ransac_locations, inliers1 = ransac(cbm_lats, lngs)

print('Mean number of inliers: {}'.format(statistics.mean(list(inliers1.values()))))
print('Median number of inliers: {}'.format(statistics.median(list(inliers1.values()))))
print('Stdev number of inliers: {}'.format(statistics.stdev(list(inliers1.values()))))
sys.stdout.flush()

# Do RANSAC again but with the actual
lats_with_actuals = copy.deepcopy(cbm_lats)
lngs_with_actuals = copy.deepcopy(lngs)

for place in actual_locations:
    lats_with_actuals[place].append(actual_locations[place][0])
    lngs_with_actuals[place].append(actual_locations[place][1])

_, inliers2 = ransac(lats_with_actuals, lngs_with_actuals, actual=True)

without_actual = 0
tied = 0
actual_better = 0

for place in actual_locations:
    if inliers1[place] < inliers2[place]:
        actual_better += 1
    elif inliers1[place] == inliers2[place]:
        tied += 1
    else:
        without_actual += 1

print('Less inliers with actual location: {}'.format(without_actual))
print('Tied: {}'.format(tied))
print('More inliers with actual location: {}'.format(actual_better))
sys.stdout.flush()

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
    normalized = [x * 100 / sum(sun_visibles[key]) for x in sun_visibles[key]]
    sun_visibles[key] = normalized

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
def kde(lats, lngs, init_guess_locations):
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
        res = minimize(kde_func_to_minimize, np.asarray([math.radians(x) for x in init_guess_locations[key]]), args=(kernel,), method='L-BFGS-B', bounds=bnds, options={'maxiter':500})

        if res.success:
            #density_locations[key] = (math.degrees(res.x[0]), math.degrees(res.x[1]))
            best_latitude = math.degrees(res.x[0])
            best_longitude = math.degrees(res.x[1])
        else:
            print('WARNING - scipy minimize function failed on location ' + key)
            sys.stdout.flush()

            # Grid search for maximum density.

            # density_locations[key] = median_locations[key] # Use median if it fails.
            best_score = -float('inf')
            best_longitude = -181
            best_latitude = -91

            latitude_search = np.linspace(min_lat, max_lat,
                                          num=16001)  # Worst case pi/16000 radians (0.01125 degrees) step size.
            longitude_search = np.linspace(min_lng, max_lng, num=32001)  # Worst case pi/16000 radians step size.

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

#density_locations = kde(lats, lngs, median_locations)
#cbm_density_locations = kde(cbm_lats, lngs, cbm_median_locations)
density_locations = kde(lats, lngs, brock_ransac_locations)
cbm_density_locations = kde(cbm_lats, lngs, cbm_ransac_locations)

# Use Gaussian KDE for longitude.
for place in cbm_equinox_day_locations:
    cbm_equinox_day_locations[place] = (cbm_equinox_day_locations[place][0], cbm_density_locations[place][1])
    cbm_equinox_declin_locations[place] = (cbm_equinox_declin_locations[place][0], cbm_density_locations[place][1])

def plot_map(lats, lngs, mean_locations, median_locations, density_locations, ransac_locations, particle_locations, gmm_locations, particle_mahalanobis_locations, equinox_day_locations, equinox_declin_locations, mode='sun'):
    map_t0 = time.time()

    # Plot locations on a map.
    for place in lats:
        if len(lats[place]) < 50: # Need at least 50 points.
            continue

        min_lat = max(min(lats[place]) - 0.03, -90)
        max_lat = min(max(lats[place]) + 0.03, 90)
        min_lng = max(min(lngs[place]) - 0.03, -180)
        max_lng = min(max(lngs[place]) + 0.03, 180)

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
            elif mode == 'daylength':
                if day_lengths[i] / 3600 >= 11 and day_lengths[i] / 3600 <= 13:
                    colors.append('r')
                else:
                    colors.append('g')

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

        actual_and_pred_lngs = [actual_lng] + [mean_locations[place][1]] + [median_locations[place][1]] + [density_locations[place][1]] + [ransac_locations[place][1]] + [particle_locations[place][1]] + [gmm_locations[place][1]] + [particle_mahalanobis_locations[place][1]] + [equinox_day_locations[place][1]] + [equinox_declin_locations[place][1]]
        actual_and_pred_lats = [actual_lat] + [mean_locations[place][0]] + [median_locations[place][0]] + [density_locations[place][0]] + [ransac_locations[place][0]] + [particle_locations[place][0]] + [gmm_locations[place][0]] + [particle_mahalanobis_locations[place][0]] + [equinox_day_locations[place][0]] + [equinox_declin_locations[place][0]]
        actual_and_pred_colors = ['w', 'm', 'c', mcolors.CSS4_COLORS['fuchsia'], 'xkcd:chartreuse', 'xkcd:navy', 'xkcd:pink', 'xkcd:azure', 'xkcd:goldenrod', 'xkcd:gold']

        guesses = map.scatter(lngs[place], lats[place], s=20, c=colors, latlon=True, zorder=10)
        actual_and_pred = map.scatter(actual_and_pred_lngs, actual_and_pred_lats, s=25, c=actual_and_pred_colors, latlon=True, zorder=10, marker='^')

        #plt.legend(handles=[guesses, actual, mean_guess, median_guess, density_guess])

        if mode == 'sun':
            guess_colors = ['g', 'r', mcolors.CSS4_COLORS['crimson'], 'k']
            legend_labels = ['sunrise and sunset in frames', 'sunrise not in frames', 'sunset not in frames', 'sunrise and sunset not in frames',
                             'actual location', 'mean', 'median', 'gaussian kde', 'RANSAC', 'particle filter', 'GMM',
                             'particle filter (mahalanobis)', 'equinox solstice weighting (day)', 'equinox solstice weighting (declination)']

            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                         [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]
        elif mode == 'season':
            guess_colors = ['b', 'y', 'r', mcolors.CSS4_COLORS['tan']]
            legend_labels = ['winter', 'spring', 'summer', 'fall',
                             'actual location', 'mean', 'median', 'gaussian kde', 'RANSAC', 'particle filter', 'GMM',
                             'particle filter (mahalanobis)', 'equinox solstice weighting', 'equinox solstice weighting (declination)']
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                         [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]
        elif mode == 'daylength':
            guess_colors = ['r', 'g']
            legend_labels = ['11-13 daylength hours', 'other',
                             'actual location', 'mean', 'median', 'gaussian kde', 'RANSAC', 'particle filter', 'GMM',
                             'particle filter (mahalanobis)', 'equinox solstice weighting', 'equinox solstice weighting (declination)']
            handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in guess_colors] + \
                         [plt.plot([], marker="^", ls="", color=color)[0] for color in actual_and_pred_colors]

        plt.legend(handlelist, legend_labels)

        plt.title(place)

        if not os.path.isdir('/srv/glusterfs/vli/maps1/' + mode + '/'):
            os.mkdir('/srv/glusterfs/vli/maps1/' + mode + '/')

        plt.savefig('/srv/glusterfs/vli/maps1/' + mode + '/' + place + '.png')
        plt.close()

    map_t1 = time.time()
    print('Calculating map time (m): ' + str((map_t1 - map_t0) / 60))


plot_map(cbm_lats, lngs, cbm_mean_locations, cbm_median_locations, cbm_density_locations, cbm_ransac_locations, cbm_particle_locations, cbm_gmm_locations, cbm_particle_mahalanobis_locations, cbm_equinox_day_locations, cbm_equinox_declin_locations, 'sun')
plot_map(cbm_lats, lngs, cbm_mean_locations, cbm_median_locations, cbm_density_locations, cbm_ransac_locations, cbm_particle_locations, cbm_gmm_locations, cbm_particle_mahalanobis_locations, cbm_equinox_day_locations, cbm_equinox_declin_locations, 'season')
plot_map(cbm_lats, lngs, cbm_mean_locations, cbm_median_locations, cbm_density_locations, cbm_ransac_locations, cbm_particle_locations, cbm_gmm_locations, cbm_particle_mahalanobis_locations, cbm_equinox_day_locations, cbm_equinox_declin_locations, 'daylength')
sys.stdout.flush()

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

brock_ransac_distances = []
brock_ransac_longitude_err = []
brock_ransac_latitude_err = []

brock_particle_distances = []
brock_particle_longitude_err = []
brock_particle_latitude_err = []

cbm_ransac_distances = []
cbm_ransac_longitude_err = []
cbm_ransac_latitude_err = []

cbm_particle_distances = []
cbm_particle_longitude_err = []
cbm_particle_latitude_err = []

cbm_gmm_distances = []
cbm_gmm_longitude_err = []
cbm_gmm_latitude_err = []

cbm_particle_m_distances = []
cbm_particle_m_longitude_err = []
cbm_particle_m_latitude_err = []

cbm_equinox_day_distances = []
cbm_equinox_day_longitude_err = []
cbm_equinox_day_latitude_err = []

cbm_equinox_declin_distances = []
cbm_equinox_declin_longitude_err = []
cbm_equinox_declin_latitude_err = []

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
    brock_ransac_pred_lat = brock_ransac_locations[place][0]
    brock_ransac_pred_lng = brock_ransac_locations[place][1]
    brock_particle_pred_lat = brock_particle_locations[place][0]
    brock_particle_pred_lng = brock_particle_locations[place][1]

    cbm_mean_pred_lat = cbm_mean_locations[place][0]
    cbm_mean_pred_lng = cbm_mean_locations[place][1]
    cbm_median_pred_lat = cbm_median_locations[place][0]
    cbm_median_pred_lng = cbm_median_locations[place][1]
    cbm_density_pred_lat = cbm_density_locations[place][0]
    cbm_density_pred_lng = cbm_density_locations[place][1]
    cbm_ransac_pred_lat = cbm_ransac_locations[place][0]
    cbm_ransac_pred_lng = cbm_ransac_locations[place][1]
    cbm_particle_pred_lat = cbm_particle_locations[place][0]
    cbm_particle_pred_lng = cbm_particle_locations[place][1]
    cbm_gmm_pred_lat = cbm_gmm_locations[place][0]
    cbm_gmm_pred_lng = cbm_gmm_locations[place][1]
    cbm_particle_m_pred_lat = cbm_particle_mahalanobis_locations[place][0]
    cbm_particle_m_pred_lng = cbm_particle_mahalanobis_locations[place][1]
    cbm_equinox_day_pred_lat = cbm_equinox_day_locations[place][0]
    cbm_equinox_day_pred_lng = cbm_equinox_day_locations[place][1]
    cbm_equinox_declin_pred_lat = cbm_equinox_declin_locations[place][0]
    cbm_equinox_declin_pred_lng = cbm_equinox_declin_locations[place][1]

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
    mean_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_mean_pred_lng))
    median_latitude_err.append(compute_distance(actual_lat, actual_lng, median_pred_lat, actual_lng))
    median_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_median_pred_lng))
    density_latitude_err.append(compute_distance(actual_lat, actual_lng, density_pred_lat, actual_lng))
    density_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_density_pred_lng))

    cbm_mean_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_mean_pred_lat, actual_lng))
    cbm_median_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_median_pred_lat, actual_lng))
    cbm_density_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_density_pred_lat, actual_lng))

    cbm_ransac_distance = compute_distance(actual_lat, actual_lng, cbm_ransac_pred_lat, cbm_ransac_pred_lng)
    cbm_ransac_distances.append(cbm_ransac_distance)
    cbm_ransac_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_ransac_pred_lat, actual_lng))
    cbm_ransac_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_ransac_pred_lng))

    brock_ransac_distance = compute_distance(actual_lat, actual_lng, brock_ransac_pred_lat, brock_ransac_pred_lng)
    brock_ransac_distances.append(brock_ransac_distance)
    brock_ransac_latitude_err.append(compute_distance(actual_lat, actual_lng, brock_ransac_pred_lat, actual_lng))
    brock_ransac_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, brock_ransac_pred_lng))

    brock_particle_distance = compute_distance(actual_lat, actual_lng, brock_particle_pred_lat, brock_particle_pred_lng)
    brock_particle_distances.append(brock_particle_distance)
    brock_particle_latitude_err.append(compute_distance(actual_lat, actual_lng, brock_particle_pred_lat, actual_lng))
    brock_particle_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, brock_particle_pred_lng))

    cbm_particle_distance = compute_distance(actual_lat, actual_lng, cbm_particle_pred_lat, cbm_particle_pred_lng)
    cbm_particle_distances.append(cbm_particle_distance)
    cbm_particle_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_particle_pred_lat, actual_lng))
    cbm_particle_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_particle_pred_lng))

    cbm_gmm_distance = compute_distance(actual_lat, actual_lng, cbm_gmm_pred_lat, cbm_gmm_pred_lng)
    cbm_gmm_distances.append(cbm_gmm_distance)
    cbm_gmm_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_gmm_pred_lat, actual_lng))
    cbm_gmm_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_gmm_pred_lng))

    cbm_particle_m_distance = compute_distance(actual_lat, actual_lng, cbm_particle_m_pred_lat, cbm_particle_m_pred_lng)
    cbm_particle_m_distances.append(cbm_particle_m_distance)
    cbm_particle_m_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_particle_m_pred_lat, actual_lng))
    cbm_particle_m_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_particle_m_pred_lng))

    cbm_equinox_day_distance = compute_distance(actual_lat, actual_lng, cbm_equinox_day_pred_lat, cbm_equinox_day_pred_lng)
    cbm_equinox_day_distances.append(cbm_equinox_day_distance)
    cbm_equinox_day_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_equinox_day_pred_lat, actual_lng))
    cbm_equinox_day_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_equinox_day_pred_lng))

    cbm_equinox_declin_distance = compute_distance(actual_lat, actual_lng, cbm_equinox_declin_pred_lat, cbm_equinox_declin_pred_lng)
    cbm_equinox_declin_distances.append(cbm_equinox_declin_distance)
    cbm_equinox_declin_latitude_err.append(compute_distance(actual_lat, actual_lng, cbm_equinox_declin_pred_lat, actual_lng))
    cbm_equinox_declin_longitude_err.append(compute_distance(actual_lat, actual_lng, actual_lat, cbm_equinox_declin_pred_lng))

    if random.randint(1, 100) < 101:
        if median_distance < 25 or density_distance < 25 or brock_ransac_distance < 25 or \
           cbm_ransac_distance < 25 or cbm_median_distance < 25 or cbm_density_distance < 25 or \
           cbm_particle_distance < 25 or cbm_gmm_distance < 25 or cbm_particle_m_distance < 25 or \
           cbm_equinox_day_distance < 25 or cbm_equinox_declin_distance < 25:
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
        print('Using RANSAC: ' + str(cbm_ransac_distance))
        print('Using particle: ' + str(cbm_particle_distance))
        print('Using GMM: ' + str(cbm_gmm_distance))
        print('Using mahalanobis particle: ' + str(cbm_particle_m_distance))
        print('Using equinox (day): ' + str(cbm_equinox_day_distance))
        print('Using equinox (declin): ' + str(cbm_equinox_declin_distance))
        print('Actual lat, lng: ' + str(actual_lat) + ', ' + str(actual_lng))
        print('Brock Distance')
        print('Mean lat, lng: ' + str(mean_pred_lat) + ', ' + str(mean_pred_lng))
        print('Median lat, lng: ' + str(median_pred_lat) + ', ' + str(median_pred_lng))
        print('Density lat, lng: ' + str(density_pred_lat) + ', ' + str(density_pred_lng))
        print('CBM Distance')
        print('Mean lat, lng: ' + str(cbm_mean_pred_lat) + ', ' + str(cbm_mean_pred_lng))
        print('Median lat, lng: ' + str(cbm_median_pred_lat) + ', ' + str(cbm_median_pred_lng))
        print('Density lat, lng: ' + str(cbm_density_pred_lat) + ', ' + str(cbm_density_pred_lng))
        print('RANSAC lat, lng: ' + str(cbm_ransac_pred_lat) + ', ' + str(cbm_ransac_pred_lng))
        print('Particle lat, lng: ' + str(cbm_particle_pred_lat) + ', ' + str(cbm_particle_pred_lng))
        print('GMM lat, lng: ' + str(cbm_gmm_pred_lat) + ', ' + str(cbm_gmm_pred_lng))
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

    plt.savefig('/srv/glusterfs/vli/maps1/' + prefix + label + '_days_used.png')
    plt.close()

scatter_t0 = time.time()

scatter(days_used, mean_distances, 'mo', 'mean')
scatter(days_used, median_distances, 'co', 'median')
scatter(days_used, density_distances, None, 'gaussian kde', color=mcolors.CSS4_COLORS['fuchsia'], linestyle='None', marker='o')

scatter(days_used, cbm_mean_distances, 'mo', 'mean', cbm=True)
scatter(days_used, cbm_median_distances, 'co', 'median', cbm=True)
scatter(days_used, cbm_density_distances, None, 'gaussian kde', color=mcolors.CSS4_COLORS['fuchsia'], linestyle='None', marker='o', cbm=True)
scatter(days_used, cbm_ransac_distances, None, 'RANSAC', color='xkcd:chartreuse', linestyle='None', marker='o', cbm=True)
scatter(days_used, cbm_particle_distances, None, 'particle filter', color='xkcd:navy', linestyle='None', marker='o', cbm=True)
scatter(days_used, cbm_gmm_distances, None, 'GMM', color='xkcd:pink', linestyle='None', marker='o', cbm=True)
scatter(days_used, cbm_particle_m_distances, None, 'particle filter with mahalanobis distance', color='xkcd:azure', linestyle='None', marker='o', cbm=True)
scatter(days_used, cbm_equinox_day_distances, None, 'equinox solstice weighting (day)', color='xkcd:goldenrod', linestyle='None', marker='o', cbm=True)
scatter(days_used, cbm_equinox_declin_distances, None, 'equinox solstice weighting (declination)', color='xkcd:gold', linestyle='None', marker='o', cbm=True)

scatter_t1 = time.time()
print('Calculating scatter time (m): ' + str((scatter_t1 - scatter_t0) / 60))

def median_rmse(data):
    median = statistics.median(data)
    mse = mean_squared_error(data, [median] * len(data))
    return math.sqrt(mse)

def bar(x, y, ylabel, xlabel, x_labels, title, filename, yerr=None):
    plt.figure(figsize=(24, 12))

    if yerr is not None:
        yerr = [(0,) * len(x), tuple(yerr)] # Only keep top half of error line.

    x = np.arange(len(x))
    #y = bucket_distances
    width = 0.35
    plt.bar(x, y, width, color='r', yerr=yerr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    plt.title(title)
    plt.savefig('/srv/glusterfs/vli/maps1/' + filename)
    plt.close()

# Plot average distance error vs. time interval OVER ALL DAYS.
bucket_size = 5 # minute intervals
buckets = list(range(0, 35, bucket_size))
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]
bucket_labels[-1] = bucket_labels[-1] + '+'
bucket_distances = [[] for x in range(len(buckets))]
cbm_bucket_distances = [[] for x in range(len(buckets))]
bucket_rmses = [0 for x in range(len(buckets))]
cbm_bucket_rmses = [0 for x in range(len(buckets))]
bucket_num_data_pts = [0] * len(buckets)
cbm_bucket_num_data_pts = [0] * len(buckets)
for i in range(len(days)):
    for bIdx, bucket in enumerate(buckets):
        if days[i].interval_min < bucket + bucket_size:
            break

    distance_err = compute_distance(days[i].lat, days[i].lng, latitudes[i], longitudes[i])
    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, cbm_latitudes[i], longitudes[i])

    bucket_distances[bIdx].append(distance_err)
    cbm_bucket_distances[bIdx].append(cbm_distance_err)

for bdIdx, distance_errs in enumerate(bucket_distances):
    if len(distance_errs) > 0:
        bucket_distances[bdIdx] = statistics.median(distance_errs)
        bucket_rmses[bdIdx] = median_rmse(distance_errs)
        bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_bucket_distances):
    if len(distance_errs) > 0:
        cbm_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        cbm_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        cbm_bucket_distances[bdIdx] = 0

#bar(buckets, bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Days vs. Photo Interval (min)', 'interval.png', bucket_rmses)
bar(buckets, cbm_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Days vs. Photo Interval (min)', 'cbm_interval.png', cbm_bucket_rmses) #
print('INTERVAL OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_bucket_num_data_pts)) #

# Plot average distance error vs. sunrise, sunset available over ALL DAYS.
sun_type_labels = ['Both', ' Either Sunrise Only or Sunset Only', 'Neither']
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
        sun_type_distances[1].append(distance_err)
        cbm_sun_type_distances[1].append(cbm_distance_err)
    else:
        sun_type_distances[2].append(distance_err)
        cbm_sun_type_distances[2].append(cbm_distance_err)

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

#bar(season_labels, season_distances, 'Median Distance Error (km)', 'Season', season_labels, 'Median Error (km) Over All Days vs. Season', 'season.png', season_rmses)
bar(season_labels, cbm_season_distances, 'Median Distance Error (km)', 'Season', season_labels, 'Median Error (km) Over All Days vs. Season', 'cbm_season.png', cbm_season_rmses)
print('SEASON OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_season_num_data_pts))

# Plot average distance error vs. day length hours over ALL DAYS.
buckets = list(range(0, 25, 1)) # 1 hour intervals
bucket_labels = [str(x) + '-' + str(x + 1) for x in buckets]
bucket_distances = [[] for x in range(len(buckets))]
cbm_bucket_distances = [[] for x in range(len(buckets))]
#bucket_rmses = [0 for x in range(len(buckets))]
cbm_bucket_rmses = [0 for x in range(len(buckets))]
#bucket_num_data_pts = [0] * len(buckets)
cbm_bucket_num_data_pts = [0] * len(buckets)

for i in range(len(days)):
    for bIdx, bucket in enumerate(buckets):
        if day_lengths[i] / 3600 < bucket + 1:
            break

    #distance_err = compute_distance(days[i].lat, days[i].lng, latitudes[i], longitudes[i])
    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, cbm_latitudes[i], longitudes[i])

    #bucket_distances[bIdx].append(distance_err)
    cbm_bucket_distances[bIdx].append(cbm_distance_err)

#for bdIdx, distance_errs in enumerate(bucket_distances):
#    if len(distance_errs) > 0:
#        bucket_distances[bdIdx] = statistics.median(distance_errs)
#        bucket_rmses[bdIdx] = median_rmse(distance_errs) #
#        bucket_num_data_pts[bdIdx] += len(distance_errs) #
#    else:
#        bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_bucket_distances):
    if len(distance_errs) > 0:
        cbm_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_bucket_rmses[bdIdx] = median_rmse(distance_errs) #
        cbm_bucket_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        cbm_bucket_distances[bdIdx] = 0

#bar(buckets, bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Days vs. Photo Interval (min)', 'interval.png', bucket_rmses)
bar(buckets, cbm_bucket_distances, 'Median Distance Error (km)', 'Day Length Hours', bucket_labels, 'Median Error (km) Over All Days vs. Day Length Hours', 'cbm_day_length.png', cbm_bucket_rmses) #
print('DAY LENGTH OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_bucket_num_data_pts)) #

# Plot average latitude error vs. equinox offset over ALL DAYS.
bucket_size = 7 # day intervals
buckets = list(range(0, 100, bucket_size))
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]
bucket_labels[-1] = bucket_labels[-1] + '+'
bucket_distances = [[] for x in range(len(buckets))]
cbm_bucket_distances = [[] for x in range(len(buckets))]
#bucket_rmses = [0 for x in range(len(buckets))]
cbm_bucket_rmses = [0] * len(buckets)
#bucket_num_data_pts = [0] * len(buckets)
cbm_bucket_num_data_pts = [0] * len(buckets)

for i in range(len(days)):
    for bIdx, bucket in enumerate(buckets):
        if equinox_offsets[i] < bucket + bucket_size:
            break

    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, cbm_latitudes[i], days[i].lng) # Use actual longitude twice.
    cbm_bucket_distances[bIdx].append(cbm_distance_err)

for bdIdx, distance_errs in enumerate(cbm_bucket_distances):
    if len(distance_errs) > 0:
        cbm_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        cbm_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        cbm_bucket_distances[bdIdx] = 0

bar(buckets, cbm_bucket_distances, 'Median Latitude Distance Error (km)', 'Days From Equinox', bucket_labels, 'Median Latitude Error (km) Over All Days vs. Days From Equinox', 'cbm_equinox_lat.png', cbm_bucket_rmses)
print('EQUINOX (LAT) OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_bucket_num_data_pts))

# Plot average longitude error vs. solstice offset over ALL DAYS.
bucket_size = 7 # day intervals
buckets = list(range(0, 100, bucket_size))
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]
bucket_labels[-1] = bucket_labels[-1] + '+'
bucket_distances = [[] for x in range(len(buckets))]
cbm_bucket_distances = [[] for x in range(len(buckets))]
#bucket_rmses = [0 for x in range(len(buckets))]
cbm_bucket_rmses = [0] * len(buckets)
#bucket_num_data_pts = [0] * len(buckets)
cbm_bucket_num_data_pts = [0] * len(buckets)

for i in range(len(days)):
    for bIdx, bucket in enumerate(buckets):
        if solstice_offsets[i] < bucket + bucket_size:
            break

    cbm_distance_err = compute_distance(days[i].lat, days[i].lng, days[i].lat, longitudes[i]) # Use actual latitude twice.
    cbm_bucket_distances[bIdx].append(cbm_distance_err)

for bdIdx, distance_errs in enumerate(cbm_bucket_distances):
    if len(distance_errs) > 0:
        cbm_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        cbm_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        cbm_bucket_distances[bdIdx] = 0

bar(buckets, cbm_bucket_distances, 'Median Longitude Distance Error (km)', 'Days From Solstice', bucket_labels, 'Median Longitude Error (km) Over All Days vs. Days From Solstice', 'cbm_solstice_lng.png', cbm_bucket_rmses)
print('SOLSTICE (LNG) OVER ALL DAYS BUCKETS NUM DATA PTS: ' + str(cbm_bucket_num_data_pts))

def plot_all_places(bucket_size, buckets, bucket_labels, locations, x_data, x_name, method_name, xlabel, ylabel, title, filename, sub_idx=None):
    bucket_distances = [[] for x in range(len(buckets))]
    bucket_rmses = [0] * len(buckets) # for x in range(len(buckets))]
    bucket_num_data_pts = [0] * len(buckets)

    for key in x_data:
        for bIdx, bucket in enumerate(buckets):
            if sub_idx is None:
                if x_data[key] < bucket + bucket_size:
                    break
            else:
                if x_data[key][sub_idx] < bucket + bucket_size:
                    break

        distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                        locations[key][0], locations[key][1])
        bucket_distances[bIdx].append(distance_err)

    for bdIdx, distance_errs in enumerate(bucket_distances):
        if len(distance_errs) > 0:
            bucket_distances[bdIdx] = statistics.median(distance_errs)
            bucket_rmses[bdIdx] = median_rmse(distance_errs)
            bucket_num_data_pts[bdIdx] += len(distance_errs)
        else:
            bucket_distances[bdIdx] = 0

    bar(buckets, bucket_distances, ylabel, xlabel, bucket_labels, title, filename, bucket_rmses)

    print(x_name + ' OVER ALL LOCATIONS (' + method_name + ') BUCKETS NUM DATA PTS: ' + str(bucket_num_data_pts))

# Plot average distance error vs. days used over ALL PLACES.
# Only using CBM model for now.
bucket_size = 10 # days
buckets = list(range(0, 125, bucket_size))
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]
bucket_labels[-1] = bucket_labels[-1] + '+'

days_used_dict = {}
for p_idx, place in enumerate(lats):
    days_used_dict[place] = days_used[p_idx]

plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_median_locations, days_used_dict, 'DAYS USED', 'MEDIAN',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Median vs. # Days Used', 'cbm_days_used_median_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_density_locations, days_used_dict, 'DAYS USED', 'DENSITY',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Gaussian KDE vs. # Days Used', 'cbm_days_used_density_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_ransac_locations, days_used_dict, 'DAYS USED', 'RANSAC',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using RANSAC vs. # Days Used', 'cbm_days_used_ransac_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_locations, days_used_dict, 'DAYS USED', 'PARTICLE',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter vs. # Days Used', 'cbm_days_used_particle_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_gmm_locations, days_used_dict, 'DAYS USED', 'GMM',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using GMM vs. # Days Used', 'cbm_days_used_gmm_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_mahalanobis_locations, days_used_dict, 'DAYS USED', 'PARTICLE (MAHALANOBIS)',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter with Mahalanobis Distance vs. # Days Used', 'cbm_days_used_particle_m_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_day_locations, days_used_dict, 'DAYS USED', 'EQUINOX SOLSTICE (DAY)',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Day) vs. # Days Used', 'cbm_days_used_equinox_day_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_declin_locations, days_used_dict, 'DAYS USED', 'EQUINOX SOLSTICE (DECLIN)',
                '# Days Used', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Solar Declination) vs. # Days Used', 'cbm_days_used_equinox_declin_places.png')


# Plot average distance error vs. intervals over ALL PLACES.
# Only using CBM model for now.
bucket_size = 5 # minute intervals
buckets = list(range(0, 30, bucket_size))
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]
bucket_labels[-1] = bucket_labels[-1] + '+'

plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_median_locations, intervals, 'INTERVAL', 'MEDIAN',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Median vs. Photo Interval (min)', 'cbm_interval_median_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_density_locations, intervals, 'INTERVAL', 'DENSITY',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Gaussian KDE vs. Photo Interval (min)', 'cbm_interval_density_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_ransac_locations, intervals, 'INTERVAL', 'RANSAC',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using RANSAC vs. Photo Interval (min)', 'cbm_interval_ransac_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_locations, intervals, 'INTERVAL', 'PARTICLE',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter vs. Photo Interval (min)', 'cbm_interval_particle_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_gmm_locations, intervals, 'INTERVAL', 'GMM',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using GMM vs. Photo Interval (min)', 'cbm_interval_gmm_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_mahalanobis_locations, intervals, 'INTERVAL', 'PARTICLE (MAHALANOBIS)',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter with Mahalanobis Distance vs. Photo Interval (min)', 'cbm_interval_particle_m_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_day_locations, intervals, 'INTERVAL', 'EQUINOX SOLSTICE (DAY)',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Day) vs. Photo Interval (min)', 'cbm_interval_equinox_day_places.png')
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_declin_locations, intervals, 'INTERVAL', 'EQUINOX SOLSTICE (DECLIN)',
                'Minutes Between Frames', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Solar Declination) vs. Photo Interval (min)', 'cbm_interval_equinox_declin_places.png')

'''
cbm_median_bucket_distances = [[] for x in range(len(buckets))]
cbm_density_bucket_distances = [[] for x in range(len(buckets))]
ransac_bucket_distances = [[] for x in range(len(buckets))]
particle_bucket_distances = [[] for x in range(len(buckets))]
gmm_bucket_distances = [[] for x in range(len(buckets))]

cbm_median_bucket_rmses = [0 for x in range(len(buckets))]
cbm_median_bucket_num_data_pts = [0] * len(buckets)
cbm_density_bucket_rmses = [0 for x in range(len(buckets))]
cbm_density_bucket_num_data_pts = [0] * len(buckets)
ransac_bucket_rmses = [0 for x in range(len(buckets))]
ransac_bucket_num_data_pts = [0] * len(buckets)
particle_bucket_rmses = [0 for x in range(len(buckets))]
particle_bucket_num_data_pts = [0] * len(buckets)
gmm_bucket_rmses = [0 for x in range(len(buckets))]
gmm_bucket_num_data_pts = [0] * len(buckets)

for key in intervals:
    for bIdx, bucket in enumerate(buckets):
        if intervals[key] < bucket + bucket_size:
            break

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_bucket_distances[bIdx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_bucket_distances[bIdx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_ransac_locations[key][0], cbm_ransac_locations[key][1])
    ransac_bucket_distances[bIdx].append(ransac_distance_err)

    particle_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_particle_locations[key][0], cbm_particle_locations[key][1])
    particle_bucket_distances[bIdx].append(particle_distance_err)

    gmm_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                             cbm_gmm_locations[key][0], cbm_gmm_locations[key][1])
    gmm_bucket_distances[bIdx].append(gmm_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_bucket_distances):
    if len(distance_errs) > 0:
        cbm_median_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_median_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        cbm_median_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        cbm_median_bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_bucket_distances):
    if len(distance_errs) > 0:
        cbm_density_bucket_distances[bdIdx] = statistics.median(distance_errs)
        cbm_density_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        cbm_density_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        cbm_density_bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(ransac_bucket_distances):
    if len(distance_errs) > 0:
        ransac_bucket_distances[bdIdx] = statistics.median(distance_errs)
        ransac_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        ransac_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        ransac_bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(particle_bucket_distances):
    if len(distance_errs) > 0:
        particle_bucket_distances[bdIdx] = statistics.median(distance_errs)
        particle_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        particle_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        particle_bucket_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(gmm_bucket_distances):
    if len(distance_errs) > 0:
        gmm_bucket_distances[bdIdx] = statistics.median(distance_errs)
        gmm_bucket_rmses[bdIdx] = median_rmse(distance_errs)
        gmm_bucket_num_data_pts[bdIdx] += len(distance_errs)
    else:
        gmm_bucket_distances[bdIdx] = 0

bar(buckets, cbm_median_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using Median vs. Photo Interval (min)', 'cbm_interval_median_places.png', cbm_median_bucket_rmses)
bar(buckets, cbm_density_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using Gaussian KDE vs. Photo Interval (min)', 'cbm_interval_density_places.png', cbm_density_bucket_rmses)
bar(buckets, ransac_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using RANSAC vs. Photo Interval (min)', 'cbm_interval_ransac_places.png', ransac_bucket_rmses)
bar(buckets, particle_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using Particle Filter vs. Photo Interval (min)', 'cbm_interval_particle_places.png', particle_bucket_rmses)
bar(buckets, gmm_bucket_distances, 'Median Distance Error (km)', 'Minutes Between Frames', bucket_labels, 'Median Error (km) Over All Locations Using GMM vs. Photo Interval (min)', 'cbm_interval_gmm_places.png', gmm_bucket_rmses)
print('INTERVAL OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_bucket_num_data_pts))
print('INTERVAL OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_bucket_num_data_pts))
print('INTERVAL OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_bucket_num_data_pts))
print('INTERVAL OVER ALL LOCATIONS (PARTICLE) BUCKETS NUM DATA PTS: ' + str(particle_bucket_num_data_pts))
print('INTERVAL OVER ALL LOCATIONS (GMM) BUCKETS NUM DATA PTS: ' + str(gmm_bucket_num_data_pts))
'''

# Plot average distance error vs. percentage of days with sunrise and sunset visible over ALL PLACES.
# Only using CBM model for now.
bucket_size = 25
buckets = list(range(0, 100, bucket_size)) # 25% buckets
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]

plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_median_locations, sun_visibles, 'SUN', 'MEDIAN',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Median vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_median_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_density_locations, sun_visibles, 'SUN', 'DENSITY',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Gaussian KDE vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_density_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_ransac_locations, sun_visibles, 'SUN', 'RANSAC',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using RANSAC vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_ransac_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_locations, sun_visibles, 'SUN', 'PARTICLE',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_particle_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_gmm_locations, sun_visibles, 'SUN', 'GMM',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using GMM vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_gmm_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_mahalanobis_locations, sun_visibles, 'SUN', 'PARTICLE (MAHALANOBIS)',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter with Mahalanobis Distance vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_particle_m_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_day_locations, sun_visibles, 'SUN', 'EQUINOX SOLSTICE (DAY)',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Day) vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_equinox_day_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_declin_locations, sun_visibles, 'SUN', 'EQUINOX SOLSTICE (DECLIN)',
                '% of Days With Both Sunrise and Sunset Visible', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Solar Declination) vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_equinox_declin_places.png', 0)

'''
cbm_median_sun_type_distances = [[] for x in range(len(buckets))]
cbm_density_sun_type_distances = [[] for x in range(len(buckets))]
ransac_sun_type_distances = [[] for x in range(len(buckets))]
particle_sun_type_distances = [[] for x in range(len(buckets))]

cbm_median_sun_type_rmses = [0 for x in range(len(buckets))]
cbm_median_sun_type_num_data_pts = [0] * len(buckets)
cbm_density_sun_type_rmses = [0 for x in range(len(buckets))]
cbm_density_sun_type_num_data_pts = [0] * len(buckets)
ransac_sun_type_rmses = [0 for x in range(len(buckets))]
ransac_sun_type_num_data_pts = [0] * len(buckets)
particle_sun_type_rmses = [0 for x in range(len(buckets))]
particle_sun_type_num_data_pts = [0] * len(buckets)

for key in sun_visibles:
    for bIdx, bucket in enumerate(buckets):
        if sun_visibles[key][0] * 100 < bucket + bucket_size:
            break

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_sun_type_distances[bIdx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_sun_type_distances[bIdx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_ransac_locations[key][0], cbm_ransac_locations[key][1])
    ransac_sun_type_distances[bIdx].append(ransac_distance_err)

    particle_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                             cbm_particle_locations[key][0], cbm_particle_locations[key][1])
    particle_sun_type_distances[bIdx].append(particle_distance_err)

for bdIdx, distance_errs in enumerate(cbm_median_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_median_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
        cbm_median_sun_type_rmses[bdIdx] = median_rmse(distance_errs)
        cbm_median_sun_type_num_data_pts[bdIdx] += len(distance_errs)
    else:
        cbm_median_sun_type_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(cbm_density_sun_type_distances):
    if len(distance_errs) > 0:
        cbm_density_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
        cbm_density_sun_type_rmses[bdIdx] = median_rmse(distance_errs)
        cbm_density_sun_type_num_data_pts[bdIdx] += len(distance_errs)
    else:
        cbm_density_sun_type_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(ransac_sun_type_distances):
    if len(distance_errs) > 0:
        ransac_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
        ransac_sun_type_rmses[bdIdx] = median_rmse(distance_errs)
        ransac_sun_type_num_data_pts[bdIdx] += len(distance_errs)
    else:
        ransac_sun_type_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(particle_sun_type_distances):
    if len(distance_errs) > 0:
        particle_sun_type_distances[bdIdx] = statistics.mean(distance_errs)
        particle_sun_type_rmses[bdIdx] = median_rmse(distance_errs)
        particle_sun_type_num_data_pts[bdIdx] += len(distance_errs)
    else:
        particle_sun_type_distances[bdIdx] = 0

bar(buckets, cbm_median_sun_type_distances, 'Median Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using Median vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_median_places.png', cbm_median_sun_type_rmses)
bar(buckets, cbm_density_sun_type_distances, 'Median Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using Gaussian KDE vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_density_places.png', cbm_density_sun_type_rmses)
bar(buckets, ransac_sun_type_distances, 'Median Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using RANSAC vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_ransac_places.png', ransac_sun_type_rmses)
bar(buckets, particle_sun_type_distances, 'Median Distance Error (km)', '% of Days With Both Sunrise and Sunset Visible', bucket_labels, 'Avg. Error (km) Over All Locations Using Particle Filter vs. % of Days with Sunrise and Sunset Visible', 'cbm_sun_particle_places.png', particle_sun_type_rmses)
print('SUN OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_sun_type_num_data_pts)) #
print('SUN OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_sun_type_num_data_pts)) #
print('SUN OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_sun_type_num_data_pts)) #
print('SUN OVER ALL LOCATIONS (PARTICLE) BUCKETS NUM DATA PTS: ' + str(particle_sun_type_num_data_pts)) #
'''

# Average distance error vs. latitude over ALL PLACES.
bucket_size = 5 # 5 degree buckets
buckets = list(range(-90, 90, bucket_size))
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]

plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_median_locations, cbm_median_locations, 'LAT', 'MEDIAN',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Median vs. Latitude', 'cbm_lat_median_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_density_locations, cbm_density_locations, 'LAT', 'DENSITY',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Gaussian KDE vs. Latitude', 'cbm_lat_density_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_ransac_locations, cbm_ransac_locations, 'LAT', 'RANSAC',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using RANSAC vs. Latitude', 'cbm_lat_ransac_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_locations, cbm_particle_locations, 'LAT', 'PARTICLE',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter vs. Latitude', 'cbm_lat_particle_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_gmm_locations, cbm_gmm_locations, 'LAT', 'GMM',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using GMM vs. Latitude', 'cbm_lat_gmm_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_mahalanobis_locations, cbm_particle_mahalanobis_locations, 'LAT', 'PARTICLE (MAHALANOBIS)',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter with Mahalanobis Distance vs. Latitude', 'cbm_lat_particle_m_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_day_locations, cbm_equinox_day_locations, 'LAT', 'EQUINOX SOLSTICE (DAY)',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Day) vs. Latitude', 'cbm_lat_equinox_day_places.png', 0)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_declin_locations, cbm_equinox_declin_locations, 'LAT', 'EQUINOX SOLSTICE (DECLIN)',
                'Latitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Solar Declination) vs. Latitude', 'cbm_lat_equinox_declin_places.png', 0)


'''
cbm_median_lat_distances = [[] for x in range(len(buckets))]
cbm_density_lat_distances = [[] for x in range(len(buckets))]
ransac_lat_distances = [[] for x in range(len(buckets))]
particle_lat_distances = [[] for x in range(len(buckets))]
gmm_lat_distances = [[] for x in range(len(buckets))]

cbm_median_lat_rmses = [0 for x in range(len(buckets))]
cbm_median_lat_num_data_pts = [0] * len(buckets)
cbm_density_lat_rmses = [0 for x in range(len(buckets))]
cbm_density_lat_num_data_pts = [0] * len(buckets)
ransac_lat_rmses = [0 for x in range(len(buckets))]
ransac_lat_num_data_pts = [0] * len(buckets)
particle_lat_rmses = [0 for x in range(len(buckets))]
particle_lat_num_data_pts = [0] * len(buckets)
gmm_lat_rmses = [0 for x in range(len(buckets))]
gmm_lat_num_data_pts = [0] * len(buckets)

for key in lats:
    median_idx = len(buckets) - 1
    density_idx = len(buckets) - 1
    ransac_idx = len(buckets) - 1
    particle_idx = len(buckets) - 1
    gmm_idx = len(buckets) - 1

    for bIdx, bucket in enumerate(buckets):
        if cbm_median_locations[key][0] < bucket + bucket_size:
            median_idx = min(bIdx, median_idx)
        if cbm_density_locations[key][0] < bucket + bucket_size:
            density_idx = min(bIdx, density_idx)
        if cbm_ransac_locations[key][0] < bucket + bucket_size:
            ransac_idx = min(bIdx, ransac_idx)
        if cbm_particle_locations[key][0] < bucket + bucket_size:
            particle_idx = min(bIdx, particle_idx)
        if cbm_gmm_locations[key][0] < bucket + bucket_size:
            gmm_idx = min(bIdx, gmm_idx)

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_lat_distances[median_idx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_lat_distances[density_idx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_ransac_locations[key][0], cbm_ransac_locations[key][1])
    ransac_lat_distances[ransac_idx].append(ransac_distance_err)

    particle_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_particle_locations[key][0],
                                           cbm_particle_locations[key][1])
    particle_lat_distances[particle_idx].append(particle_distance_err)

    gmm_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                             cbm_gmm_locations[key][0],
                                             cbm_gmm_locations[key][1])
    gmm_lat_distances[gmm_idx].append(gmm_distance_err)

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

for bdIdx, distance_errs in enumerate(particle_lat_distances):
    if len(distance_errs) > 0:
        particle_lat_distances[bdIdx] = statistics.median(distance_errs)
        particle_lat_rmses[bdIdx] = median_rmse(distance_errs)  #
        particle_lat_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        particle_lat_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(gmm_lat_distances):
    if len(distance_errs) > 0:
        gmm_lat_distances[bdIdx] = statistics.median(distance_errs)
        gmm_lat_rmses[bdIdx] = median_rmse(distance_errs)  #
        gmm_lat_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        gmm_lat_distances[bdIdx] = 0

bar(buckets, cbm_median_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using Median vs. Latitude', 'cbm_lat_median_places.png', cbm_median_lat_rmses)
bar(buckets, cbm_density_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using Gaussian KDE vs. Latitude', 'cbm_lat_density_places.png', cbm_density_lat_rmses)
bar(buckets, ransac_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using RANSAC vs. Latitude', 'cbm_lat_ransac_places.png', ransac_lat_rmses)
bar(buckets, particle_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using Particle Filter vs. Latitude', 'cbm_lat_particle_places.png', particle_lat_rmses)
bar(buckets, gmm_lat_distances, 'Median Distance Error (km)', 'Latitude', bucket_labels, 'Median Error (km) Over All Locations Using GMM vs. Latitude', 'cbm_lat_gmm_places.png', gmm_lat_rmses)
print('LAT OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_lat_num_data_pts))
print('LAT OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_lat_num_data_pts))
print('LAT OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_lat_num_data_pts))
print('LAT OVER ALL LOCATIONS (PARTICLE) BUCKETS NUM DATA PTS: ' + str(particle_lat_num_data_pts))
print('LAT OVER ALL LOCATIONS (GMM) BUCKETS NUM DATA PTS: ' + str(gmm_lat_num_data_pts))
'''

# Average distance error vs. longitude over ALL PLACES.
bucket_size = 10 # 10 degree buckets
buckets = list(range(-180, 180, bucket_size))
bucket_labels = [str(x) + '-' + str(x + bucket_size) for x in buckets]


plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_median_locations, cbm_median_locations, 'LNG', 'MEDIAN',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Median vs. Longitude', 'cbm_lng_median_places.png', 1)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_density_locations, cbm_density_locations, 'LNG', 'DENSITY',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Gaussian KDE vs. Longitude', 'cbm_lng_density_places.png', 1)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_ransac_locations, cbm_ransac_locations, 'LNG', 'RANSAC',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using RANSAC vs. Longitude', 'cbm_lng_ransac_places.png', 1)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_locations, cbm_particle_locations, 'LNG', 'PARTICLE',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter vs. Longitude', 'cbm_lng_particle_places.png', 1)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_gmm_locations, cbm_gmm_locations, 'LNG', 'GMM',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using GMM vs. Longitude', 'cbm_lng_gmm_places.png', 1)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_particle_mahalanobis_locations, cbm_particle_mahalanobis_locations, 'LNG', 'PARTICLE (MAHALANOBIS)',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Particle Filter with Mahalanobis Distance vs. Longitude', 'cbm_lng_particle_m_places.png', 1)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_day_locations, cbm_equinox_day_locations, 'LNG', 'EQUINOX SOLSTICE (DAY)',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Day) vs. Longitude', 'cbm_lng_equinox_day_places.png', 1)
plot_all_places(bucket_size, buckets, bucket_labels,
                cbm_equinox_declin_locations, cbm_equinox_declin_locations, 'LNG', 'EQUINOX SOLSTICE (DECLIN)',
                'Longitude', 'Median Distance Error (km)', 'Median Error (km) Over All Locations Using Equinox Solstice Weighting (Solar Declination) vs. Longitude', 'cbm_lng_equinox_declin_places.png', 1)

'''
cbm_median_lng_distances = [[] for x in range(len(buckets))]
cbm_density_lng_distances = [[] for x in range(len(buckets))]
ransac_lng_distances = [[] for x in range(len(buckets))]
particle_lng_distances = [[] for x in range(len(buckets))]
gmm_lng_distances = [[] for x in range(len(buckets))]

cbm_median_lng_rmses = [0 for x in range(len(buckets))]
cbm_median_lng_num_data_pts = [0] * len(buckets)
cbm_density_lng_rmses = [0 for x in range(len(buckets))]
cbm_density_lng_num_data_pts = [0] * len(buckets)
ransac_lng_rmses = [0 for x in range(len(buckets))]
ransac_lng_num_data_pts = [0] * len(buckets)
particle_lng_rmses = [0 for x in range(len(buckets))]
particle_lng_num_data_pts = [0] * len(buckets)
gmm_lng_rmses = [0 for x in range(len(buckets))]
gmm_lng_num_data_pts = [0] * len(buckets)

for key in lngs:
    median_idx = len(buckets) - 1
    density_idx = len(buckets) - 1
    ransac_idx = len(buckets) - 1
    particle_idx = len(buckets) - 1
    gmm_idx = len(buckets) - 1

    for bIdx, bucket in enumerate(buckets):
        if cbm_median_locations[key][1] <= bucket + bucket_size:
            median_idx = min(bIdx, median_idx)
        if cbm_density_locations[key][1] <= bucket + bucket_size:
            density_idx = min(bIdx, density_idx)
        if cbm_ransac_locations[key][1] <= bucket + bucket_size:
            ransac_idx = min(bIdx, ransac_idx)
        if cbm_particle_locations[key][1] <= bucket + bucket_size:
            particle_idx = min(bIdx, particle_idx)
        if cbm_gmm_locations[key][1] <= bucket + bucket_size:
            gmm_idx = min(bIdx, gmm_idx)

    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_median_lng_distances[median_idx].append(cbm_median_distance_err)

    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    cbm_density_lng_distances[density_idx].append(cbm_density_distance_err)

    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_ransac_locations[key][0], cbm_ransac_locations[key][1])
    ransac_lng_distances[ransac_idx].append(ransac_distance_err)

    particle_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_particle_locations[key][0], cbm_particle_locations[key][1])
    particle_lng_distances[particle_idx].append(particle_distance_err)

    gmm_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                             cbm_gmm_locations[key][0], cbm_gmm_locations[key][1])
    gmm_lng_distances[gmm_idx].append(gmm_distance_err)

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

for bdIdx, distance_errs in enumerate(particle_lng_distances):
    if len(distance_errs) > 0:
        particle_lng_distances[bdIdx] = statistics.median(distance_errs)
        particle_lng_rmses[bdIdx] = median_rmse(distance_errs)  #
        particle_lng_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        particle_lng_distances[bdIdx] = 0

for bdIdx, distance_errs in enumerate(gmm_lng_distances):
    if len(distance_errs) > 0:
        gmm_lng_distances[bdIdx] = statistics.median(distance_errs)
        gmm_lng_rmses[bdIdx] = median_rmse(distance_errs)  #
        gmm_lng_num_data_pts[bdIdx] += len(distance_errs) #
    else:
        gmm_lng_distances[bdIdx] = 0

bar(buckets, cbm_median_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using Median vs. Longitude', 'cbm_lng_median_places.png', cbm_median_lng_rmses)
bar(buckets, cbm_density_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using Gaussian KDE vs. Longitude', 'cbm_lng_density_places.png', cbm_density_lng_rmses)
bar(buckets, ransac_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using RANSAC vs. Longitude', 'cbm_lng_ransac_places.png', ransac_lng_rmses)
bar(buckets, particle_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using Particle Filter vs. Longitude', 'cbm_lng_particle_places.png', particle_lng_rmses)
bar(buckets, gmm_lng_distances, 'Median Distance Error (km)', 'Longitude', bucket_labels, 'Median Error (km) Over All Locations Using GMM vs. Longitude', 'cbm_lng_gmm_places.png', gmm_lng_rmses)
print('LNG OVER ALL LOCATIONS (MEDIAN) BUCKETS NUM DATA PTS: ' + str(cbm_median_lng_num_data_pts))
print('LNG OVER ALL LOCATIONS (DENSITY) BUCKETS NUM DATA PTS: ' + str(cbm_density_lng_num_data_pts))
print('LNG OVER ALL LOCATIONS (RANSAC) BUCKETS NUM DATA PTS: ' + str(ransac_lng_num_data_pts))
print('LNG OVER ALL LOCATIONS (PARTICLE) BUCKETS NUM DATA PTS: ' + str(particle_lng_num_data_pts))
print('LNG OVER ALL LOCATIONS (GMM) BUCKETS NUM DATA PTS: ' + str(gmm_lng_num_data_pts))
'''

# Num locations vs. error using all methods.
bucket_size = 100 # 100 km buckets
buckets = list(range(0, 2000, bucket_size))
bucket_labels = [str(x // bucket_size) + '-' + str((x + bucket_size) // bucket_size) for x in buckets]
bucket_labels[-1] = bucket_labels[-1] + '+'

cbm_median_errors = [0] * len(buckets) #[[] for x in range(len(buckets))]
cbm_density_errors = [0] * len(buckets)
ransac_errors = [0] * len(buckets)
particle_errors = [0] * len(buckets)
gmm_errors = [0] * len(buckets)
particle_m_errors = [0] * len(buckets)
equinox_day_errors = [0] * len(buckets)
equinox_declin_errors = [0] * len(buckets)

cbm_median_error_rmses = [0 for x in range(len(buckets))]
cbm_median_error_num_data_pts = [0] * len(buckets)
cbm_density_error_rmses = [0 for x in range(len(buckets))]
cbm_density_error_num_data_pts = [0] * len(buckets)
ransac_error_rmses = [0 for x in range(len(buckets))]
ransac_error_num_data_pts = [0] * len(buckets)
particle_error_rmses = [0 for x in range(len(buckets))]
particle_error_num_data_pts = [0] * len(buckets)
gmm_error_rmses = [0 for x in range(len(buckets))]
gmm_error_num_data_pts = [0] * len(buckets)
particle_m_error_rmses = [0 for x in range(len(buckets))]
particle_m_error_num_data_pts = [0] * len(buckets)
equinox_day_error_rmses = [0 for x in range(len(buckets))]
equinox_day_error_num_data_pts = [0] * len(buckets)
equinox_declin_error_rmses = [0 for x in range(len(buckets))]
equinox_declin_error_num_data_pts = [0] * len(buckets)

for key in actual_locations:
    cbm_median_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_median_locations[key][0], cbm_median_locations[key][1])
    cbm_density_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_density_locations[key][0], cbm_density_locations[key][1])
    ransac_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1], cbm_ransac_locations[key][0], cbm_ransac_locations[key][1])
    particle_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                           cbm_particle_locations[key][0], cbm_particle_locations[key][1])
    gmm_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                        cbm_gmm_locations[key][0], cbm_gmm_locations[key][1])
    particle_m_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                        cbm_particle_mahalanobis_locations[key][0], cbm_particle_mahalanobis_locations[key][1])
    equinox_day_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                        cbm_equinox_day_locations[key][0], cbm_equinox_day_locations[key][1])
    equinox_declin_distance_err = compute_distance(actual_locations[key][0], actual_locations[key][1],
                                        cbm_equinox_declin_locations[key][0], cbm_equinox_declin_locations[key][1])

    median_idx = len(buckets) - 1
    density_idx = len(buckets) - 1
    ransac_idx = len(buckets) - 1
    particle_idx = len(buckets) - 1
    gmm_idx = len(buckets) - 1
    particle_m_idx = len(buckets) - 1
    equinox_day_idx = len(buckets) - 1
    equinox_declin_idx = len(buckets) - 1

    for bIdx, bucket in enumerate(buckets):
        if cbm_median_distance_err <= bucket + bucket_size:
            median_idx = min(bIdx, median_idx)
        if cbm_density_distance_err <= bucket + bucket_size:
            density_idx = min(bIdx, density_idx)
        if ransac_distance_err <= bucket + bucket_size:
            ransac_idx = min(bIdx, ransac_idx)
        if particle_distance_err <= bucket + bucket_size:
            particle_idx = min(bIdx, particle_idx)
        if gmm_distance_err <= bucket + bucket_size:
            gmm_idx = min(bIdx, gmm_idx)
        if particle_m_distance_err <= bucket + bucket_size:
            particle_m_idx = min(bIdx, particle_m_idx)
        if equinox_day_distance_err <= bucket + bucket_size:
            equinox_day_idx = min(bIdx, equinox_day_idx)
        if equinox_declin_distance_err <= bucket + bucket_size:
            equinox_declin_idx = min(bIdx, equinox_declin_idx)

    cbm_median_errors[median_idx] += 1
    cbm_density_errors[density_idx] += 1
    ransac_errors[ransac_idx] += 1
    particle_errors[particle_idx] += 1
    gmm_errors[gmm_idx] += 1
    particle_m_errors[particle_idx] += 1
    equinox_day_errors[particle_idx] += 1
    equinox_declin_errors[particle_idx] += 1

bar(buckets, cbm_median_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using Median', 'cbm_error_median.png')
bar(buckets, cbm_density_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using Gaussian KDE', 'cbm_error_density.png')
bar(buckets, ransac_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using RANSAC', 'cbm_error_ransac.png')
bar(buckets, particle_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using Particle Filter', 'cbm_error_particle.png')
bar(buckets, gmm_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using GMM', 'cbm_error_gmm.png')
bar(buckets, particle_m_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using Particle Filter With Mahalanobis Distance', 'cbm_error_particle_m.png')
bar(buckets, equinox_day_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using Equinox Solstice Weighting (Day)', 'cbm_error_equinox_day.png')
bar(buckets, equinox_declin_errors, '# of Places', 'Error ({} km)'.format(bucket_size), bucket_labels, 'Histogram of Error (km) Using Equinox Solstice Weighting (Solar Declination)', 'cbm_error_equinox_declin.png')

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
print('')
sys.stdout.flush()

print('Brock Means Avg. Distance Error: {:.6f}'.format(statistics.mean(mean_distances)))
print('Brock Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(median_distances)))
print('Brock Density Avg. Distance Error: {:.6f}'.format(statistics.mean(density_distances)))
print('Brock RANSAC Avg. Distance Error: {:.6f}'.format(statistics.mean(brock_ransac_distances)))
print('Brock Particle Avg. Distance Error: {:.6f}'.format(statistics.mean(brock_particle_distances)))
print('Brock Means Median Distance Error: {:.6f}'.format(statistics.median(mean_distances)))
print('Brock Medians Median Distance Error: {:.6f}'.format(statistics.median(median_distances)))
print('Brock Density Median Distance Error: {:.6f}'.format(statistics.median(density_distances)))
print('Brock RANSAC Median Distance Error: {:.6f}'.format(statistics.median(brock_ransac_distances)))
print('Brock Particle Median Distance Error: {:.6f}'.format(statistics.median(brock_particle_distances)))
print('Brock Means Max Distance Error: {:.6f}'.format(max(mean_distances)))
print('Brock Means Min Distance Error: {:.6f}'.format(min(mean_distances)))
print('Brock Medians Max Distance Error: {:.6f}'.format(max(median_distances)))
print('Brock Medians Min Distance Error: {:.6f}'.format(min(median_distances)))
print('Brock Density Max Distance Error: {:.6f}'.format(max(density_distances)))
print('Brock Density Min Distance Error: {:.6f}'.format(min(density_distances)))
print('Brock RANSAC Max Distance Error: {:.6f}'.format(max(brock_ransac_distances)))
print('Brock RANSAC Min Distance Error: {:.6f}'.format(min(brock_ransac_distances)))
print('Brock Particle Max Distance Error: {:.6f}'.format(max(brock_particle_distances)))
print('Brock Particle Min Distance Error: {:.6f}'.format(min(brock_particle_distances)))
print('')
print('CBM Means Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_mean_distances)))
print('CBM Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_median_distances)))
print('CBM Density Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_density_distances)))
print('RANSAC Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_ransac_distances)))
print('Particle Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_particle_distances)))
print('GMM Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_gmm_distances)))
print('Particle (Mahalanobis) Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_particle_m_distances)))
print('Equinox (Day) Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_equinox_day_distances)))
print('Equinox (Declin) Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_equinox_declin_distances)))
print('CBM Means Median Distance Error: {:.6f}'.format(statistics.median(cbm_mean_distances)))
print('CBM Medians Median Distance Error: {:.6f}'.format(statistics.median(cbm_median_distances)))
print('CBM Density Median Distance Error: {:.6f}'.format(statistics.median(cbm_density_distances)))
print('RANSAC Median Distance Error: {:.6f}'.format(statistics.median(cbm_ransac_distances)))
print('Particle Median Distance Error: {:.6f}'.format(statistics.median(cbm_particle_distances)))
print('GMM Median Distance Error: {:.6f}'.format(statistics.median(cbm_gmm_distances)))
print('Particle (Mahalanobis) Median Distance Error: {:.6f}'.format(statistics.median(cbm_particle_m_distances)))
print('Equinox (Day) Median Distance Error: {:.6f}'.format(statistics.median(cbm_equinox_day_distances)))
print('Equinox (Declin) Median Distance Error: {:.6f}'.format(statistics.median(cbm_equinox_declin_distances)))
print('CBM Means Max Distance Error: {:.6f}'.format(max(cbm_mean_distances)))
print('CBM Means Min Distance Error: {:.6f}'.format(min(cbm_mean_distances)))
print('CBM Medians Max Distance Error: {:.6f}'.format(max(cbm_median_distances)))
print('CBM Medians Min Distance Error: {:.6f}'.format(min(cbm_median_distances)))
print('CBM Density Max Distance Error: {:.6f}'.format(max(cbm_density_distances)))
print('CBM Density Min Distance Error: {:.6f}'.format(min(cbm_density_distances)))
print('RANSAC Max Distance Error: {:.6f}'.format(max(cbm_ransac_distances)))
print('RANSAC Min Distance Error: {:.6f}'.format(min(cbm_ransac_distances)))
print('Particle Max Distance Error: {:.6f}'.format(max(cbm_particle_distances)))
print('Particle Min Distance Error: {:.6f}'.format(min(cbm_particle_distances)))
print('GMM Max Distance Error: {:.6f}'.format(max(cbm_gmm_distances)))
print('GMM Min Distance Error: {:.6f}'.format(min(cbm_gmm_distances)))
print('Particle (Mahalanobis) Max Distance Error: {:.6f}'.format(max(cbm_particle_m_distances)))
print('Particle (Mahalanobis) Min Distance Error: {:.6f}'.format(min(cbm_particle_m_distances)))
print('Equinox (Day) Max Distance Error: {:.6f}'.format(max(cbm_equinox_day_distances)))
print('Equinox (Day) Min Distance Error: {:.6f}'.format(min(cbm_equinox_day_distances)))
print('Equinox (Declin) Max Distance Error: {:.6f}'.format(max(cbm_equinox_declin_distances)))
print('Equinox (Declin) Min Distance Error: {:.6f}'.format(min(cbm_equinox_declin_distances)))
print('')
print('Means Avg. Longitude Error: {:.6f}'.format(statistics.mean(mean_longitude_err)))
print('Medians Avg. Longitude Error: {:.6f}'.format(statistics.mean(median_longitude_err)))
print('Density Avg. Longitude Error: {:.6f}'.format(statistics.mean(density_longitude_err)))
print('RANSAC Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_ransac_longitude_err)))
print('Particle Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_particle_longitude_err)))
print('GMM Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_gmm_longitude_err)))
print('Particle (Mahalanobis) Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_particle_m_longitude_err)))
print('Equinox (Day) Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_day_longitude_err)))
print('Equinox (Declin) Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_declin_longitude_err)))
print('Means Median Longitude Error: {:.6f}'.format(statistics.median(mean_longitude_err)))
print('Medians Median Longitude Error: {:.6f}'.format(statistics.median(median_longitude_err)))
print('Density Median Longitude Error: {:.6f}'.format(statistics.median(density_longitude_err)))
print('RANSAC Median Longitude Error: {:.6f}'.format(statistics.median(cbm_ransac_longitude_err)))
print('Particle Median Longitude Error: {:.6f}'.format(statistics.median(cbm_particle_longitude_err)))
print('GMM Median Longitude Error: {:.6f}'.format(statistics.median(cbm_gmm_longitude_err)))
print('Particle (Mahalanobis) Median Longitude Error: {:.6f}'.format(statistics.median(cbm_particle_m_longitude_err)))
print('Equinox (Day) Median Longitude Error: {:.6f}'.format(statistics.median(cbm_equinox_day_longitude_err)))
print('Equinox (Declin) Median Longitude Error: {:.6f}'.format(statistics.median(cbm_equinox_declin_longitude_err)))
print('Means Max Longitude Error: {:.6f}'.format(max(mean_longitude_err)))
print('Means Min Longitude Error: {:.6f}'.format(min(mean_longitude_err)))
print('Medians Max Longitude Error: {:.6f}'.format(max(median_longitude_err)))
print('Medians Min Longitude Error: {:.6f}'.format(min(median_longitude_err)))
print('Density Max Longitude Error: {:.6f}'.format(max(density_longitude_err)))
print('Density Min Longitude Error: {:.6f}'.format(min(density_longitude_err)))
print('RANSAC Max Longitude Error: {:.6f}'.format(max(cbm_ransac_longitude_err)))
print('RANSAC Min Longitude Error: {:.6f}'.format(min(cbm_ransac_longitude_err)))
print('Particle Max Longitude Error: {:.6f}'.format(max(cbm_particle_longitude_err)))
print('Particle Min Longitude Error: {:.6f}'.format(min(cbm_particle_longitude_err)))
print('GMM Max Longitude Error: {:.6f}'.format(max(cbm_gmm_longitude_err)))
print('GMM Min Longitude Error: {:.6f}'.format(min(cbm_gmm_longitude_err)))
print('Particle (Mahalanobis) Max Longitude Error: {:.6f}'.format(max(cbm_particle_m_longitude_err)))
print('Particle (Mahalanobis) Min Longitude Error: {:.6f}'.format(min(cbm_particle_m_longitude_err)))
print('Equinox (Day) Max Longitude Error: {:.6f}'.format(max(cbm_equinox_day_longitude_err)))
print('Equinox (Day) Min Longitude Error: {:.6f}'.format(min(cbm_equinox_day_longitude_err)))
print('Equinox (Declin) Max Longitude Error: {:.6f}'.format(max(cbm_equinox_declin_longitude_err)))
print('Equinox (Declin) Min Longitude Error: {:.6f}'.format(min(cbm_equinox_declin_longitude_err)))
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
print('RANSAC Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_ransac_latitude_err)))
print('Particle Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_particle_latitude_err)))
print('GMM Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_gmm_latitude_err)))
print('Particle (Mahalanobis) Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_particle_m_latitude_err)))
print('Equinox (Day) Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_day_latitude_err)))
print('Equinox (Declin) Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_declin_latitude_err)))
print('CBM Means Median Latitude Error: {:.6f}'.format(statistics.median(cbm_mean_latitude_err)))
print('CBM Medians Median Latitude Error: {:.6f}'.format(statistics.median(cbm_median_latitude_err)))
print('CBM Density Median Latitude Error: {:.6f}'.format(statistics.median(cbm_density_latitude_err)))
print('RANSAC Median Latitude Error: {:.6f}'.format(statistics.median(cbm_ransac_latitude_err)))
print('Particle Median Latitude Error: {:.6f}'.format(statistics.median(cbm_particle_latitude_err)))
print('GMM Median Latitude Error: {:.6f}'.format(statistics.median(cbm_gmm_latitude_err)))
print('Particle (Mahalanobis) Median Latitude Error: {:.6f}'.format(statistics.median(cbm_particle_m_latitude_err)))
print('Equinox (Day) Median Latitude Error: {:.6f}'.format(statistics.median(cbm_equinox_day_latitude_err)))
print('Equinox (Declin) Median Latitude Error: {:.6f}'.format(statistics.median(cbm_equinox_declin_latitude_err)))
print('CBM Means Max Latitude Error: {:.6f}'.format(max(cbm_mean_latitude_err)))
print('CBM Means Min Latitude Error: {:.6f}'.format(min(cbm_mean_latitude_err)))
print('CBM Medians Max Latitude Error: {:.6f}'.format(max(cbm_median_latitude_err)))
print('CBM Medians Min Latitude Error: {:.6f}'.format(min(cbm_median_latitude_err)))
print('CBM Density Max Latitude Error: {:.6f}'.format(max(cbm_density_latitude_err)))
print('CBM Density Min Latitude Error: {:.6f}'.format(min(cbm_density_latitude_err)))
print('RANSAC Max Latitude Error: {:.6f}'.format(max(cbm_ransac_latitude_err)))
print('RANSAC Min Latitude Error: {:.6f}'.format(min(cbm_ransac_latitude_err)))
print('Particle Max Latitude Error: {:.6f}'.format(max(cbm_particle_latitude_err)))
print('Particle Min Latitude Error: {:.6f}'.format(min(cbm_particle_latitude_err)))
print('GMM Max Latitude Error: {:.6f}'.format(max(cbm_gmm_latitude_err)))
print('GMM Min Latitude Error: {:.6f}'.format(min(cbm_gmm_latitude_err)))
print('Particle (Mahalanobis) Max Latitude Error: {:.6f}'.format(max(cbm_particle_m_latitude_err)))
print('Particle (Mahalanobis) Min Latitude Error: {:.6f}'.format(min(cbm_particle_m_latitude_err)))
print('Equinox (Day) Max Latitude Error: {:.6f}'.format(max(cbm_equinox_day_latitude_err)))
print('Equinox (Day) Min Latitude Error: {:.6f}'.format(min(cbm_equinox_day_latitude_err)))
print('Equinox (Declin) Max Latitude Error: {:.6f}'.format(max(cbm_equinox_declin_latitude_err)))
print('Equinox (Declin) Min Latitude Error: {:.6f}'.format(min(cbm_equinox_declin_latitude_err)))
print('')
sys.stdout.flush()

# Compute error for places under 50 km.
print('Using only places with at least 50 days.')



subset_idx = []
for p_idx, place in enumerate(lats):
    if len(lats) < 50:
        continue

    subset_idx.append(p_idx)

print('# places with at least 50 days: {}'.format(len(subset_idx)))
print('')

cbm_mean_distances = [cbm_mean_distances[x] for x in subset_idx]
cbm_median_distances = [cbm_median_distances[x] for x in subset_idx]
cbm_density_distances = [cbm_density_distances[x] for x in subset_idx]
cbm_ransac_distances = [cbm_ransac_distances[x] for x in subset_idx]
cbm_particle_distances = [cbm_particle_distances[x] for x in subset_idx]
cbm_gmm_distances = [cbm_gmm_distances[x] for x in subset_idx]
cbm_particle_m_distances = [cbm_particle_m_distances[x] for x in subset_idx]
cbm_equinox_day_distances = [cbm_equinox_day_distances[x] for x in subset_idx]
cbm_equinox_declin_distances = [cbm_equinox_declin_distances[x] for x in subset_idx]

mean_longitude_err = [mean_longitude_err[x] for x in subset_idx]
median_longitude_err = [median_longitude_err[x] for x in subset_idx]
density_longitude_err = [density_longitude_err[x] for x in subset_idx]
cbm_ransac_longitude_err = [cbm_ransac_longitude_err[x] for x in subset_idx]
cbm_particle_longitude_err = [cbm_particle_longitude_err[x] for x in subset_idx]
cbm_gmm_longitude_err = [cbm_gmm_longitude_err[x] for x in subset_idx]
cbm_particle_m_longitude_err = [cbm_particle_m_longitude_err[x] for x in subset_idx]
cbm_equinox_day_longitude_err = [cbm_equinox_day_longitude_err[x] for x in subset_idx]
cbm_equinox_declin_longitude_err = [cbm_equinox_declin_longitude_err[x] for x in subset_idx]

cbm_mean_latitude_err = [cbm_mean_latitude_err[x] for x in subset_idx]
cbm_median_latitude_err = [cbm_median_latitude_err[x] for x in subset_idx]
cbm_density_latitude_err = [cbm_density_latitude_err[x] for x in subset_idx]
cbm_ransac_latitude_err = [cbm_ransac_latitude_err[x] for x in subset_idx]
cbm_particle_latitude_err = [cbm_particle_latitude_err[x] for x in subset_idx]
cbm_gmm_latitude_err = [cbm_gmm_latitude_err[x] for x in subset_idx]
cbm_particle_m_latitude_err = [cbm_particle_m_latitude_err[x] for x in subset_idx]
cbm_equinox_day_latitude_err = [cbm_equinox_day_latitude_err[x] for x in subset_idx]
cbm_equinox_declin_latitude_err = [cbm_equinox_declin_latitude_err[x] for x in subset_idx]

print('CBM Means Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_mean_distances)))
print('CBM Medians Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_median_distances)))
print('CBM Density Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_density_distances)))
print('RANSAC Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_ransac_distances)))
print('Particle Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_particle_distances)))
print('GMM Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_gmm_distances)))
print('Particle (Mahalanobis) Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_particle_m_distances)))
print('Equinox (Day) Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_equinox_day_distances)))
print('Equinox (Declin) Avg. Distance Error: {:.6f}'.format(statistics.mean(cbm_equinox_declin_distances)))
print('CBM Means Median Distance Error: {:.6f}'.format(statistics.median(cbm_mean_distances)))
print('CBM Medians Median Distance Error: {:.6f}'.format(statistics.median(cbm_median_distances)))
print('CBM Density Median Distance Error: {:.6f}'.format(statistics.median(cbm_density_distances)))
print('RANSAC Median Distance Error: {:.6f}'.format(statistics.median(cbm_ransac_distances)))
print('Particle Median Distance Error: {:.6f}'.format(statistics.median(cbm_particle_distances)))
print('GMM Median Distance Error: {:.6f}'.format(statistics.median(cbm_gmm_distances)))
print('Particle (Mahalanobis) Median Distance Error: {:.6f}'.format(statistics.median(cbm_particle_m_distances)))
print('Equinox (Day) Median Distance Error: {:.6f}'.format(statistics.median(cbm_equinox_day_distances)))
print('Equinox (Declin) Median Distance Error: {:.6f}'.format(statistics.median(cbm_equinox_declin_distances)))
print('CBM Means Max Distance Error: {:.6f}'.format(max(cbm_mean_distances)))
print('CBM Means Min Distance Error: {:.6f}'.format(min(cbm_mean_distances)))
print('CBM Medians Max Distance Error: {:.6f}'.format(max(cbm_median_distances)))
print('CBM Medians Min Distance Error: {:.6f}'.format(min(cbm_median_distances)))
print('CBM Density Max Distance Error: {:.6f}'.format(max(cbm_density_distances)))
print('CBM Density Min Distance Error: {:.6f}'.format(min(cbm_density_distances)))
print('RANSAC Max Distance Error: {:.6f}'.format(max(cbm_ransac_distances)))
print('RANSAC Min Distance Error: {:.6f}'.format(min(cbm_ransac_distances)))
print('Particle Max Distance Error: {:.6f}'.format(max(cbm_particle_distances)))
print('Particle Min Distance Error: {:.6f}'.format(min(cbm_particle_distances)))
print('GMM Max Distance Error: {:.6f}'.format(max(cbm_gmm_distances)))
print('GMM Min Distance Error: {:.6f}'.format(min(cbm_gmm_distances)))
print('Particle (Mahalanobis) Max Distance Error: {:.6f}'.format(max(cbm_particle_m_distances)))
print('Particle (Mahalanobis) Min Distance Error: {:.6f}'.format(min(cbm_particle_m_distances)))
print('Equinox (Day) Max Distance Error: {:.6f}'.format(max(cbm_equinox_day_distances)))
print('Equinox (Day) Min Distance Error: {:.6f}'.format(min(cbm_equinox_day_distances)))
print('Equinox (Declin) Max Distance Error: {:.6f}'.format(max(cbm_equinox_declin_distances)))
print('Equinox (Declin) Min Distance Error: {:.6f}'.format(min(cbm_equinox_declin_distances)))
print('')
print('Means Avg. Longitude Error: {:.6f}'.format(statistics.mean(mean_longitude_err)))
print('Medians Avg. Longitude Error: {:.6f}'.format(statistics.mean(median_longitude_err)))
print('Density Avg. Longitude Error: {:.6f}'.format(statistics.mean(density_longitude_err)))
print('RANSAC Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_ransac_longitude_err)))
print('Particle Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_particle_longitude_err)))
print('GMM Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_gmm_longitude_err)))
print('Particle (Mahalanobis) Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_particle_m_longitude_err)))
print('Equinox (Day) Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_day_longitude_err)))
print('Equinox (Declin) Avg. Longitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_declin_longitude_err)))
print('Means Median Longitude Error: {:.6f}'.format(statistics.median(mean_longitude_err)))
print('Medians Median Longitude Error: {:.6f}'.format(statistics.median(median_longitude_err)))
print('Density Median Longitude Error: {:.6f}'.format(statistics.median(density_longitude_err)))
print('RANSAC Median Longitude Error: {:.6f}'.format(statistics.median(cbm_ransac_longitude_err)))
print('Particle Median Longitude Error: {:.6f}'.format(statistics.median(cbm_particle_longitude_err)))
print('GMM Median Longitude Error: {:.6f}'.format(statistics.median(cbm_gmm_longitude_err)))
print('Particle (Mahalanobis) Median Longitude Error: {:.6f}'.format(statistics.median(cbm_particle_m_longitude_err)))
print('Equinox (Day) Median Longitude Error: {:.6f}'.format(statistics.median(cbm_equinox_day_longitude_err)))
print('Equinox (Declin) Median Longitude Error: {:.6f}'.format(statistics.median(cbm_equinox_declin_longitude_err)))
print('Means Max Longitude Error: {:.6f}'.format(max(mean_longitude_err)))
print('Means Min Longitude Error: {:.6f}'.format(min(mean_longitude_err)))
print('Medians Max Longitude Error: {:.6f}'.format(max(median_longitude_err)))
print('Medians Min Longitude Error: {:.6f}'.format(min(median_longitude_err)))
print('Density Max Longitude Error: {:.6f}'.format(max(density_longitude_err)))
print('Density Min Longitude Error: {:.6f}'.format(min(density_longitude_err)))
print('RANSAC Max Longitude Error: {:.6f}'.format(max(cbm_ransac_longitude_err)))
print('RANSAC Min Longitude Error: {:.6f}'.format(min(cbm_ransac_longitude_err)))
print('Particle Max Longitude Error: {:.6f}'.format(max(cbm_particle_longitude_err)))
print('Particle Min Longitude Error: {:.6f}'.format(min(cbm_particle_longitude_err)))
print('GMM Max Longitude Error: {:.6f}'.format(max(cbm_gmm_longitude_err)))
print('GMM Min Longitude Error: {:.6f}'.format(min(cbm_gmm_longitude_err)))
print('Particle (Mahalanobis) Max Longitude Error: {:.6f}'.format(max(cbm_particle_m_longitude_err)))
print('Particle (Mahalanobis) Min Longitude Error: {:.6f}'.format(min(cbm_particle_m_longitude_err)))
print('Equinox (Day) Max Longitude Error: {:.6f}'.format(max(cbm_equinox_day_longitude_err)))
print('Equinox (Day) Min Longitude Error: {:.6f}'.format(min(cbm_equinox_day_longitude_err)))
print('Equinox (Declin) Max Longitude Error: {:.6f}'.format(max(cbm_equinox_declin_longitude_err)))
print('Equinox (Declin) Min Longitude Error: {:.6f}'.format(min(cbm_equinox_declin_longitude_err)))
print('')
print('CBM Means Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_mean_latitude_err)))
print('CBM Medians Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_median_latitude_err)))
print('CBM Density Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_density_latitude_err)))
print('RANSAC Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_ransac_latitude_err)))
print('Particle Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_particle_latitude_err)))
print('GMM Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_gmm_latitude_err)))
print('Particle (Mahalanobis) Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_particle_m_latitude_err)))
print('Equinox (Day) Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_day_latitude_err)))
print('Equinox (Declin) Avg. Latitude Error: {:.6f}'.format(statistics.mean(cbm_equinox_declin_latitude_err)))
print('CBM Means Median Latitude Error: {:.6f}'.format(statistics.median(cbm_mean_latitude_err)))
print('CBM Medians Median Latitude Error: {:.6f}'.format(statistics.median(cbm_median_latitude_err)))
print('CBM Density Median Latitude Error: {:.6f}'.format(statistics.median(cbm_density_latitude_err)))
print('RANSAC Median Latitude Error: {:.6f}'.format(statistics.median(cbm_ransac_latitude_err)))
print('Particle Median Latitude Error: {:.6f}'.format(statistics.median(cbm_particle_latitude_err)))
print('GMM Median Latitude Error: {:.6f}'.format(statistics.median(cbm_gmm_latitude_err)))
print('Particle (Mahalanobis) Median Latitude Error: {:.6f}'.format(statistics.median(cbm_particle_m_latitude_err)))
print('Equinox (Day) Median Latitude Error: {:.6f}'.format(statistics.median(cbm_equinox_day_latitude_err)))
print('Equinox (Declin) Median Latitude Error: {:.6f}'.format(statistics.median(cbm_equinox_declin_latitude_err)))
print('CBM Means Max Latitude Error: {:.6f}'.format(max(cbm_mean_latitude_err)))
print('CBM Means Min Latitude Error: {:.6f}'.format(min(cbm_mean_latitude_err)))
print('CBM Medians Max Latitude Error: {:.6f}'.format(max(cbm_median_latitude_err)))
print('CBM Medians Min Latitude Error: {:.6f}'.format(min(cbm_median_latitude_err)))
print('CBM Density Max Latitude Error: {:.6f}'.format(max(cbm_density_latitude_err)))
print('CBM Density Min Latitude Error: {:.6f}'.format(min(cbm_density_latitude_err)))
print('RANSAC Max Latitude Error: {:.6f}'.format(max(cbm_ransac_latitude_err)))
print('RANSAC Min Latitude Error: {:.6f}'.format(min(cbm_ransac_latitude_err)))
print('Particle Max Latitude Error: {:.6f}'.format(max(cbm_particle_latitude_err)))
print('Particle Min Latitude Error: {:.6f}'.format(min(cbm_particle_latitude_err)))
print('GMM Max Latitude Error: {:.6f}'.format(max(cbm_gmm_latitude_err)))
print('GMM Min Latitude Error: {:.6f}'.format(min(cbm_gmm_latitude_err)))
print('Particle (Mahalanobis) Max Latitude Error: {:.6f}'.format(max(cbm_particle_m_latitude_err)))
print('Particle (Mahalanobis) Min Latitude Error: {:.6f}'.format(min(cbm_particle_m_latitude_err)))
print('Equinox (Day) Max Latitude Error: {:.6f}'.format(max(cbm_equinox_day_latitude_err)))
print('Equinox (Day) Min Latitude Error: {:.6f}'.format(min(cbm_equinox_day_latitude_err)))
print('Equinox (Declin) Max Latitude Error: {:.6f}'.format(max(cbm_equinox_declin_latitude_err)))
print('Equinox (Declin) Min Latitude Error: {:.6f}'.format(min(cbm_equinox_declin_latitude_err)))