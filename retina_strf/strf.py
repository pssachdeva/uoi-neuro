import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import time

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score

from PyUoI.UoI_Lasso import UoI_Lasso

parser = argparse.ArgumentParser()

parser.add_argument('--data_path')
parser.add_argument('--recording', type=int)
parser.add_argument('--neuron', type=int)
parser.add_argument('--window_length', type=float, default=0.5)
parser.add_argument('--results_path')
parser.add_argument('--random_nums', default='ran1.bin')
args = parser.parse_args()

t = time.time() 
# load data
data_path = args.data_path
data = loadmat(data_path, struct_as_record=False)

# extract session
recording = args.recording
sessions = data['stimulus'].ravel()
session = sessions[recording]

# extract spikes
spikes = data['spikes']

# extract useful quantities
n_cells = np.asscalar(np.asscalar(data['datainfo']).Ncell)
n_frames = np.asscalar(session.Nframes)
frame_length = np.asscalar(session.frame)
onset = np.asscalar(session.onset)
window_length = 0.5 # in seconds
n_frames_per_window = int(np.round(window_length/frame_length))
neuron = args.neuron

# timepoints
timestamps = np.arange(n_frames) * frame_length + onset
n_timestamps = timestamps.size - 1

# extract number of spatial dimensions
params = np.asscalar(session.param)
Nx = np.asscalar(params.x)/np.asscalar(params.dx)
Ny = np.asscalar(params.y)/np.asscalar(params.dy)
n_spatial_dims = int(Nx * Ny)

####################
## CREATE DATASET ##
####################

# extract design matrix
byte = np.fromfile('ran1.bin', count=n_timestamps*n_spatial_dims//8, dtype='uint8')
X = np.unpackbits(byte).astype('float32')
X = 2 * X - 1
X = X.reshape((n_timestamps, n_spatial_dims)).T

# extract response matrix
spike_times = spikes[neuron, recording]
# bin spike train
binned_spikes, _ = np.histogram(spike_times, bins=timestamps)
# put spikes in array
binned_spikes[:n_frames_per_window-1] = 0
n_spikes = np.sum(binned_spikes)
Y = binned_spikes/n_spikes

####################
### PERFORM FITS ###
####################

# spike-triggered average #
sta = np.zeros((n_frames_per_window, n_spatial_dims))
sta_r2 = np.zeros(n_frames_per_window)

# ols #
ols_strf = np.zeros((n_frames_per_window, n_spatial_dims))
ols_intercepts = np.zeros(n_frames_per_window)
ols_r2 = np.zeros(n_frames_per_window)

# ridge #
ridge_strf = np.zeros((n_frames_per_window, n_spatial_dims))
ridge_intercepts = np.zeros(n_frames_per_window)
ridge_r2 = np.zeros(n_frames_per_window)

# lasso #
lasso_strf = np.zeros((n_frames_per_window, n_spatial_dims))
lasso_intercepts = np.zeros(n_frames_per_window)
lasso_r2 = np.zeros(n_frames_per_window)

# uoi #
uoi_strf = np.zeros((n_frames_per_window, n_spatial_dims))
uoi_intercepts = np.zeros(n_frames_per_window)
uoi_r2 = np.zeros(n_frames_per_window)

# iterate over frames in window
for frame in range(n_frames_per_window):
	print(frame)
	### STA ###

	# calculate STA
	sta[frame, :] = np.dot(X, Y)
	# calculate explained variance
	sta_r2[frame] = r2_score(
		Y, np.dot(X.T, sta[frame, :])
	)

	### OLS ###
	ols = LinearRegression()
	ols.fit(X.T, Y)
	ols_strf[frame, :] = ols.coef_.T
	ols_intercepts[frame] = ols.intercept_
	ols_r2[frame] = r2_score(
		Y, ols.predict(X.T)
	)

	### ridge ###
	ridge = RidgeCV(cv=5)
	ridge.fit(X.T, Y)
	ridge_strf[frame, :] = ridge.coef_.T
	ridge_intercepts[frame] = ridge.intercept_
	ridge_r2[frame] = r2_score(
		Y, ridge.predict(X.T)
	)

	### lasso ###
	lasso = LassoCV(normalize=True, cv=5)
	lasso.fit(X.T, Y)
	lasso_strf[frame, :] = lasso.coef_.T
	lasso_intercepts[frame] = lasso.intercept_
	lasso_r2[frame] = r2_score(
		Y, lasso.predict(X.T)
	)

	### uoi ###
	uoi = UoI_Lasso(
		n_lambdas=30,
		normalize=True,
		fit_intercept=True,
		estimation_score='r2',
		n_boots_sel=30,
		n_boots_est=30,
		selection_thres_min=1.0
	)
	uoi.fit(X.T, Y)
	uoi_strf[frame, :] = uoi.coef_.T
	uoi_intercepts[frame] = uoi.intercept_
	uoi_r2[frame] = r2_score(
		Y, uoi.intercept_ + np.dot(X.T, uoi.coef_)
	)

	# move the window up
	Y = np.roll(Y, -1, axis=0)

###################
## STORE RESULTS ##
###################

# create results file
results_path = args.results_path
results = h5py.File(results_path, 'w')

results['sta/strf'] = sta
results['sta/r2'] = sta_r2
results['ols/strf'] = ols_strf
results['ols/intercept'] = ols_intercepts
results['ols/r2'] = ols_r2
results['ridge/strf'] = ridge_strf
results['ridge/intercept'] = ridge_intercepts
results['ridge/r2'] = ridge_r2
results['lasso/strf'] = lasso_strf
results['lasso/intercept'] = lasso_intercepts
results['lasso/r2'] = lasso_r2
results['uoi/strf'] = uoi_strf
results['uoi/intercept'] = uoi_intercepts
results['uoi/r2'] = uoi_r2
results.close()

print(time.time() - t)