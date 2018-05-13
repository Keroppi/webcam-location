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

### Scale data to mean 0, unit variance. ###
scale_pca_t0 = time.time()
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

# Only try one configuration of dimension reduction to save memory.
sunrise_reduced = dim_reduction(scaled_sunrise_train_input,
                                scaled_sunrise_test_input,
                                sunrise_train_output,
                                'sunrise')
sunset_reduced = dim_reduction(scaled_sunset_train_input,
                               scaled_sunset_test_input,
                               sunset_train_output,
                               'sunset')

del scaled_sunrise_train_input
del scaled_sunrise_test_input
del scaled_sunset_train_input
del scaled_sunset_test_input
del sunrise_scaler
del sunset_scaler

scale_pca_t1 = time.time()
print('Scale data, initial PCA, and dimension red. time (min): {:.6f}'.format((scale_pca_t1 - scale_pca_t0) / 60))
sys.stdout.flush()

def model(model_name, model, model_params,
          train_input, test_input, train_output, test_output,
          dim_red_mode='pca', explained_var=None, mode='sunrise'):
    model_t0 = time.time()

    dims = train_input.shape[1]

    if type(model_params) is tuple:
        model = model(*model_params)
    elif type(model_params) is dict:
        model = model(**model_params)

    model.fit(train_input, train_output)
    y = model.predict(test_input)
    mean_err = mean_squared_error(test_output, y)

    if mode == 'sunrise':
       dir = sunrise_dir + '/' + model_name + '/'
    else:
       dir = sunset_dir + '/' + model_name + '/'

    if not os.path.isdir(dir):
        os.mkdir(dir)

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
            params_f.write('Params: {}\n'.format(model_params))
            params_f.write('MSE: {:.6f}'.format(mean_err))

        with open(dir + mode + '_mdl.pkl', 'wb') as mdl_f:
            pickle.dump(model, mdl_f)

        with open(dir + mode + '_pred.pkl', 'wb') as pred_f:
            pickle.dump(y, pred_f)

    del y
    del model

    model_t1 = time.time()
    print(model_name + ' time (min): {:.6f}'.format((model_t1 - model_t0) / 60))
    sys.stdout.flush()

def ridge(train_input, test_input, train_output, test_output,
          dim_red_mode='pca', explained_var=None, mode='sunrise'):
    alphas = list(np.arange(1e-5, 5, 1e-4))
    alpha = random.choice(alphas)

    tols = list(np.arange(1e-5, 1e-2, 1e-4))
    tol = random.choice(tols)

    del alphas
    del tols

    model('ridge', Ridge, {'alpha':alpha, 'tol':tol},
          train_input, test_input, train_output, test_output,
          dim_red_mode, explained_var, mode)


def lasso(train_input, test_input, train_output, test_output,
          dim_red_mode='pca', explained_var=None, mode='sunrise'):
    alphas = list(np.arange(1e-5, 5, 1e-4))
    alpha = random.choice(alphas)

    tols = list(np.arange(1e-5, 1e-2, 1e-4))
    tol = random.choice(tols)

    del alphas
    del tols

    model('lasso', Lasso, {'alpha':alpha, 'tol':tol},
          train_input, test_input, train_output, test_output,
          dim_red_mode, explained_var, mode)


def nn(train_input, test_input, train_output, test_output,
       dim_red_mode='pca', explained_var=None, mode='sunrise'):

    dims = train_input.shape[1]
    num_hidden_layers = random.randint(2, 8)

    hidden_layer_sizes = []
    for i in range(num_hidden_layers):
        min_width = max(math.ceil(dims / 20), 2) # at least 2
        max_width = 2 * dims # at least 2

        layer_size = random.randint(min_width, max_width)
        hidden_layer_sizes.append(layer_size)
    hidden_layer_sizes = tuple(hidden_layer_sizes)

    activations = ['identity', 'logistic', 'tanh', 'relu']
    activation_idx = random.randint(0, len(activations) - 1)
    activation = activations[activation_idx]

    alphas = list(np.arange(1e-5, 0.1, 1e-4))
    alpha = random.choice(alphas)

    tols = list(np.arange(1e-4, 1e-2, 1e-3))
    tol = random.choice(tols)

    params = {'hidden_layer_sizes':hidden_layer_sizes,
              'activation':activation,
              'alpha':alpha,
              'learning_rate':'adaptive',
              'max_iter':200,
              'tol':tol}

    del alphas
    del tols

    try:
        model('nn', MLPRegressor, params,
              train_input, test_input, train_output, test_output,
              dim_red_mode, explained_var, mode)
    except MemoryError as e:
        print('MemoryError caused by neural network.')
        sys.stdout.flush()
        pass # Doesn't matter, let's just try again.


def svr(train_input, test_input, train_output, test_output,
       dim_red_mode='pca', explained_var=None, mode='sunrise'):

    Cs = list(np.arange(1e-4, 10, 1e-3))
    C = random.choice(Cs)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = random.choice(kernels)

    epsilons = list(np.arange(1e-5, 0.01, 1e-4))
    epsilon = random.choice(epsilons)

    degrees = list(range(2, 6))
    degree = random.choice(degrees)

    tols = list(np.arange(1e-4, 1e-2, 1e-3))
    tol = random.choice(tols)

    params = {'C':C,
              'epsilon':epsilon,
              'kernel':kernel,
              'degree':degree,
              'tol':tol,
              'cache_size':5000}

    del Cs
    del epsilons
    del tols

    model('svr', SVR, params,
          train_input, test_input, train_output, test_output,
          dim_red_mode, explained_var, mode)

def train_model(model_name, mode='sunrise'):
    if mode == 'sunrise':
        reduced = sunrise_reduced
        train_output = sunrise_train_output
        test_output = sunrise_test_output
    else:
        reduced = sunset_reduced
        train_output = sunset_train_output
        test_output = sunset_test_output

    while True:
        if model_name == 'ridge':
            ridge(reduced[0],
                  reduced[1],
                  train_output,
                  test_output,
                  reduced[2],
                  reduced[3],
                  mode)
        elif model_name == 'lasso':
            lasso(reduced[0],
                  reduced[1],
                  train_output,
                  test_output,
                  reduced[2],
                  reduced[3],
                  mode)
        elif model_name == 'nn':
            nn(reduced[0],
               reduced[1],
               train_output,
               test_output,
               reduced[2],
               reduced[3],
               mode)
        elif model_name == 'svr':
            svr(reduced[0],
                reduced[1],
                train_output,
                test_output,
                reduced[2],
                reduced[3],
                mode)

print('Creating threads.')
sys.stdout.flush()

#sunrise_ridge_p = Process(target=train_model, args=('ridge', 'sunrise'))
#sunset_ridge_p = Process(target=train_model, args=('ridge', 'sunset'))
#sunrise_lasso_p = Process(target=train_model, args=('lasso', 'sunrise'))
#sunset_lasso_p = Process(target=train_model, args=('lasso', 'sunset'))
sunrise_nn_p = Process(target=train_model, args=('nn', 'sunrise'))
sunset_nn_p = Process(target=train_model, args=('nn', 'sunset'))
#sunrise_svr_p = Process(target=train_model, args=('svr', 'sunrise'))
#sunset_svr_p = Process(target=train_model, args=('svr', 'sunset'))

print('Starting threads.')
sys.stdout.flush()

#sunrise_ridge_p.start()
#sunset_ridge_p.start()
#sunrise_lasso_p.start()
#sunset_lasso_p.start()
sunrise_nn_p.start()
sunset_nn_p.start()
#sunrise_svr_p.start()
#sunset_svr_p.start()








