#!/srv/glusterfs/vli/.pyenv/shims/python

import torch, torchvision, argparse, os, datetime, time, math, pandas as pd, sys, random, statistics, numpy as np, scipy, pickle
from multiprocessing import Process

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
from sklearn.feature_selection import f_regression

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

load_t0 = time.time()
with open(sunrise_dir + '/features/sunrise_train_input.pkl', 'rb') as sunrise_train_input_f:
    sunrise_train_input = pickle.load(sunrise_train_input_f)
with open(sunrise_dir + '/features/sunrise_train_output.pkl', 'rb') as sunrise_train_output_f:
    sunrise_train_output = pickle.load(sunrise_train_output_f)
with open(sunset_dir + '/features/sunset_train_input.pkl', 'rb') as sunset_train_input_f:
    sunset_train_input = pickle.load(sunset_train_input_f)
with open(sunset_dir + '/features/sunset_train_output.pkl', 'rb') as sunset_train_output_f:
    sunset_train_output = pickle.load(sunset_train_output_f)
with open(sunrise_dir + '/features/sunrise_test_input.pkl', 'rb') as sunrise_test_input_f:
    sunrise_test_input = pickle.load(sunrise_test_input_f)
with open(sunrise_dir + '/features/sunrise_test_output.pkl', 'rb') as sunrise_test_output_f:
    sunrise_test_output = pickle.load(sunrise_test_output_f)
with open(sunset_dir + '/features/sunset_test_input.pkl', 'rb') as sunset_test_input_f:
    sunset_test_input = pickle.load(sunset_test_input_f)
with open(sunset_dir + '/features/sunset_test_output.pkl', 'rb') as sunset_test_output_f:
    sunset_test_output = pickle.load(sunset_test_output_f)
load_t1 = time.time()
print('Load Time (min): {:.6f}'.format((load_t1 - load_t0) / 60))
sys.stdout.flush()

### Scale data to mean 0, unit variance. ###
sunrise_scaler = StandardScaler()
sunrise_scaler.fit(sunrise_train_input)
scaled_sunrise_train_input = sunrise_scaler.transform(sunrise_train_input)
scaled_sunrise_test_input = sunrise_scaler.transform(sunrise_test_input)

sunset_scaler = StandardScaler()
sunset_scaler.fit(sunset_train_input)
scaled_sunset_train_input = sunset_scaler.transform(sunset_train_input)
scaled_sunset_test_input = sunset_scaler.transform(sunset_test_input)

all_dims_sunrise_pca = PCA()
all_dims_sunrise_pca.fit(scaled_sunrise_train_input)
all_dims_sunset_pca = PCA()
all_dims_sunset_pca.fit(scaled_sunset_train_input)

def dim_reduction(train_input, test_input, train_output, mode='sunrise'):
    pca_or_kbest = random.randint(0, 1)

    if pca_or_kbest == 0: # PCA
        pca_t0 = time.time()
        explained_variances = list(np.arange(0.5, 1, 0.01))  # % of variance explained determines how many components to keep
        pca_idx = random.randint(0, len(explained_variances) - 1)
        explained_variance = explained_variances[pca_idx]

        if mode == 'sunrise':
            pca_dims = 0
            total_explained = 0
            for component in all_dims_sunrise_pca.explained_variance_ratio_:
                total_explained += component
                pca_dims += 1

                if total_explained >= explained_variance:
                    break
        else:
            pca_dims = 0
            total_explained = 0
            for component in all_dims_sunset_pca.explained_variance_ratio_:
                total_explained += component
                pca_dims += 1

                if total_explained >= explained_variance:
                    break

        new_pca = PCA(n_components=pca_dims)
        new_pca.fit(train_input)
        reduced_train_input = new_pca.transform(train_input)
        reduced_test_input = new_pca.transform(test_input)

        pca_t1 = time.time()
        print('PCA Time (min): {:.6f}'.format((pca_t1 - pca_t0) / 60))
        sys.stdout.flush()

        del explained_variances
        del new_pca

        return (reduced_train_input, reduced_test_input, 'pca', total_explained, pca_dims)

    else: # SelectKBest
        kbest_t0 = time.time()
        dims = train_input.shape[1]
        k = random.randint(math.ceil(dims / 2), dims)

        kbest = SelectKBest(score_func=f_regression, k=k)
        kbest.fit(train_input, train_output)
        reduced_train_input = kbest.transform(train_input)
        reduced_test_input = kbest.transform(test_input)

        kbest_t1 = time.time()
        print('KBest Time (min): {:.6f}'.format((kbest_t1 - kbest_t0) / 60))
        sys.stdout.flush()

        del kbest

        return (reduced_train_input, reduced_test_input, 'kbest', k)

def ridge(train_input, test_input, train_output, test_output, dim_red_mode='pca', explained_var=None, mode='sunrise'):
    ridge_t0 = time.time()

    alphas = list(np.arange(1e-5, 5, 1e-4))
    alpha_idx = random.randint(0, len(alphas) - 1)
    alpha = alphas[alpha_idx]

    dims = train_input.shape[1]

    model = Ridge(alpha=alpha)
    model.fit(train_input, train_output)
    y = model.predict(test_input)
    mean_err = mean_squared_error(test_output, y)

    if mode == 'sunrise':
       dir = sunrise_dir + '/ridge/'
    else:
       dir = sunset_dir + '/ridge/'

    if os.path.isfile(dir + mode + '_best_params.txt'):
        with open(dir + mode + '_best_params.txt', 'r') as params_f:
            lines = params_f.read().splitlines()
            best_err = float(lines[-1].split()[1])
    else:
        best_err = float('inf')

    # Better test error, save these parameters, model, and predictions.
    if mean_err < best_err:
        with open(dir + mode + '_best_params.txt', 'w') as params_f:
            if dim_red_mode == 'pca':
                params_f.write('PCA: {:.6f} {}\n'.format(explained_var, dims))
            else: # SelectKBest
                params_f.write('SelectKBest: {}\n'.format(dims))
            params_f.write('alpha: {}\n'.format(alpha))
            params_f.write('MSE: {:.6f}'.format(mean_err))

        with open(sunrise_dir + mode + '_mdl.pkl', 'wb') as mdl_f:
            pickle.dump(model, mdl_f)

        with open(sunrise_dir + mode + '_pred.pkl', 'wb') as pred_f:
            pickle.dump(y, pred_f)

    del alphas
    del y
    del model

    ridge_t1 = time.time()
    print('Ridge Time (min): {:.6f}'.format((ridge_t1 - ridge_t0) / 60))
    sys.stdout.flush()

def train_model(model_name, mode='sunrise'):
    while True:
        if mode == 'sunrise':
            reduced = dim_reduction(scaled_sunrise_train_input,
                                    scaled_sunrise_test_input,
                                    sunrise_train_output,
                                    'sunrise')
            train_output = sunrise_train_output
            test_output = sunrise_test_output
        else:
            reduced = dim_reduction(scaled_sunset_train_input,
                                    scaled_sunset_test_input,
                                    sunset_train_output,
                                    'sunset')
            train_output = sunset_train_output
            test_output = sunset_test_output

        if model_name == 'ridge':
            ridge(reduced[0],
                  reduced[1],
                  train_output,
                  test_output,
                  reduced[2],
                  reduced[3],
                  mode)
        elif model_name == 'lasso':
            pass
        elif model_name == 'nn':
            pass
        elif model_name == 'svr':
            pass

        del reduced

sunrise_ridge_p = Process(target=train_model, args=('ridge', 'sunrise'))
sunset_ridge_p = Process(target=train_model, args=('ridge', 'sunset'))
sunrise_ridge_p.start()
sunset_ridge_p.start()










