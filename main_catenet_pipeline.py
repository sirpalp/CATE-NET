
from updated_rossler import RosslerFitter, plot_attractor_fit
from lstm_model import train_lstm
from graph_utils import generate_dpgm
import mne
import numpy as np

# Example file paths (update with your own)
eeg_path = "chb01_18.edf"
ross_path = "scaled_rossler.npy"

# Load EEG and precomputed attractor
raw = mne.io.read_raw_edf(eeg_path, preload=True)
ross_data_scaled = np.load(ross_path)

# Fit Rössler attractor to EEG
fitter = RosslerFitter(raw, ross_data_scaled)
eeg_curve, params = fitter.fit(num_trials=50)

# Plot attractor alignment
t_inds = [int(i / fitter.time_end * fitter.num_pts_ross) for i in fitter.t_eeg]
ross_sampled = ross_data_scaled[:, t_inds]
plot_attractor_fit(eeg_curve, ross_sampled)

# Prepare features for LSTM (mock example)
X = eeg_curve.T.reshape(1, -1, 3)  # Single sample, shape (1, 256, 3)
y = np.array([2])  # 0: interictal-free, 1: interictal, 2: ictal

# Train and evaluate LSTM
model, metrics, predictions = train_lstm(X, y, input_size=3)

# Generate brain state transition graph
G, matrix = generate_dpgm(predictions, state_labels=["Free", "Inter", "Ictal"])
