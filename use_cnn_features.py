#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, argparse, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle

sys.path.append('/home/vli/webcam-location') # For importing .py files in the same directory on the cluster.
import constants
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='Use CNN Features')
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

with open(sunrise_dir + '/features/sunrise_train_input.pkl') as sunrise_train_input_f:
    sunrise_train_input = pickle.load(sunrise_train_input_f)
with open(sunrise_dir + '/features/sunrise_train_output.pkl') as sunrise_train_output_f:
    sunrise_train_output = pickle.load(sunrise_train_output_f)
with open(sunset_dir + '/features/sunset_train_input.pkl') as sunset_train_input_f:
    sunset_train_input = pickle.load(sunset_train_input_f)
with open(sunset_dir + '/features/sunset_train_output.pkl') as sunset_train_output_f:
    sunset_train_output = pickle.load(sunset_train_output_f)
with open(sunrise_dir + '/features/sunrise_test_input.pkl') as sunrise_test_input_f:
    sunrise_test_input = pickle.load(sunrise_test_input_f)
with open(sunrise_dir + '/features/sunrise_test_output.pkl') as sunrise_test_output_f:
    sunrise_test_output = pickle.load(sunrise_test_output_f)
with open(sunset_dir + '/features/sunset_test_input.pkl') as sunset_test_input_f:
    sunset_test_input = pickle.load(sunset_test_input_f)
with open(sunset_dir + '/features/sunset_test_output.pkl') as sunset_test_output_f:
    sunset_test_output = pickle.load(sunset_test_output_f)

### Scale data to mean 0, unit variance. ###
sunrise_scaler = StandardScaler()
sunrise_scaler.fit(sunrise_train_input)
scaled_sunrise_train_input = sunrise_scaler.transform(sunrise_train_input)
scaled_sunrise_test_input = sunrise_scaler.transform(sunrise_test_input)

sunset_scaler = StandardScaler()
sunset_scaler.fit(sunset_train_input)
scaled_sunset_train_input = sunset_scaler.transform(sunset_train_input)
scaled_sunset_test_input = sunset_scaler.transform(sunset_test_input)

iterations = 0
while True: # Keep training until job is killed.
    if iterations % 20 == 0:
        print('Iterations: {}'.format(iterations))
        sys.stdout.flush()

    ### Dimension Reduction ###
    sunrise_pca = PCA()
    sunrise_pca.fit(scaled_sunrise_train_input)
    sunset_pca = PCA()
    sunset_pca.fit(scaled_sunset_train_input)

    explained_variances = list(np.arange(0.6, 1, 0.025)) # % of variance explained determines how many components to keep
    pca_idx = random.randint(0, len(explained_variances) - 1)
    explained_variance = explained_variances[pca_idx]

    sunrise_pca_dims = 0
    total_explained = 0
    for component in sunrise_pca.explained_variance_ratio_:
        total_explained += component
        sunrise_pca_dims += 1

        if total_explained >= explained_variance:
            break

    sunset_pca_dims = 0
    total_explained = 0
    for component in sunset_pca.explained_variance_ratio_:
        total_explained += component
        sunset_pca_dims += 1

        if total_explained >= explained_variance:
            break

    sunrise_pca = PCA(pca_dims=sunrise_pca_dims)
    sunrise_pca.fit(scaled_sunrise_train_input)
    reduced_sunrise_train_input = sunrise_pca.transform(scaled_sunrise_train_input)
    reduced_sunrise_test_input = sunrise_pca.transform(scaled_sunrise_test_input)
    sunset_pca = PCA(pca_dims=sunset_pca_dims)
    sunset_pca.fit(scaled_sunset_train_input)
    reduced_sunset_train_input = sunset_pca.transform(scaled_sunset_train_input)
    reduced_sunset_test_input = sunset_pca.transform(scaled_sunset_test_input)

    ### Regression ###

    alphas = list(np.arange(1e-5, 5, 1e-4))
    alpha_idx = random.randint(0, len(alphas))
    alpha = alphas[alpha_idx]
    sunrise_ridge = Ridge(alpha=alpha)
    sunrise_ridge.fit(reduced_sunrise_train_input, sunrise_train_output)
    sunrise_ridge_y = sunrise_ridge.predict(reduced_sunrise_test_input)
    sunrise_ridge_err = mean_squared_error(sunrise_test_output, sunrise_ridge_y)

    sunset_ridge = Ridge(alpha=alphas[alpha_idx])
    sunset_ridge.fit(reduced_sunset_train_input, sunset_train_output)
    sunset_ridge_y = sunset_ridge.predict(reduced_sunset_test_input)
    sunset_ridge_err = mean_squared_error(sunset_test_output, sunset_ridge_y)

    if os.path.isfile(sunrise_dir + '/ridge/sunrise_best_params.txt'):
        with open(sunrise_dir + '/ridge/sunrise_best_params.txt', 'r') as sunrise_ridge_f:
            sunrise_ridge_lines = sunrise_ridge_f.read().splitlines()
            sunrise_ridge_best_err = float(sunrise_ridge_lines[-1].split()[1])

    if os.path.isfile(sunset_dir + '/ridge/sunset_best_params.txt'):
        with open(sunset_dir + '/ridge/sunset_best_params.txt', 'r') as sunset_ridge_f:
            sunset_ridge_lines = sunset_ridge_f.read().splitlines()
            sunset_ridge_best_err = float(sunset_ridge_lines[-1].split()[1])

    # Better test error, save these parameters and model.
    if not os.path.isfile(sunrise_dir + '/ridge/sunrise_best_params.txt') or sunrise_ridge_err < sunrise_ridge_best_err:
        with open(sunrise_dir + '/ridge/sunrise_best_params.txt', 'w') as sunrise_ridge_params_f:
            sunrise_ridge_params_f.write('PCA: {:.6f} {}\n'.format(explained_variance, sunrise_pca_dims))
            sunrise_ridge_params_f.write('alpha: {}\n'.format(alpha))
            sunrise_ridge_params_f.write('MSE: {:.6f}'.format(sunrise_ridge_err))

        with open(sunrise_dir + '/ridge/sunrise_ridge_mdl.pkl', 'wb') as sunrise_ridge_mdl_f:
            pickle.dump(sunrise_ridge, sunrise_ridge_mdl_f)


    if not os.path.isfile(sunset_dir + '/ridge/sunset_best_params.txt') or sunset_ridge_err < sunset_ridge_best_err:
        with open(sunset_dir + '/ridge/sunset_best_params.txt', 'w') as sunset_ridge_params_f:
            sunset_ridge_params_f.write('PCA: {:.6f} {}\n'.format(explained_variance, sunset_pca_dims))
            sunset_ridge_params_f.write('alpha: {}\n'.format(alpha))
            sunset_ridge_params_f.write('MSE: {:.6f}'.format(sunset_ridge_err))

        with open(sunset_dir + '/ridge/sunset_ridge_mdl.pkl', 'wb') as sunset_ridge_mdl_f:
            pickle.dump(sunset_ridge, sunset_ridge_mdl_f)



    iterations += 1





