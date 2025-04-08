
import numpy as np
from hyperopt import fmin, tpe, hp
import matplotlib.pyplot as plt

def calculate_bounding_box_center(x, y, z):
    return np.array([(np.min(x) + np.max(x)) / 2,
                     (np.min(y) + np.max(y)) / 2,
                     (np.min(z) + np.max(z)) / 2])

def rotate_curve(curve, pitch, roll, yaw):
    x, y, z = curve[0, :], curve[1, :], curve[2, :]
    rotation_matrix = np.array([
        [np.cos(yaw) * np.cos(pitch), 
         -np.sin(yaw) * np.cos(roll) + np.cos(yaw) * np.sin(pitch) * np.sin(roll),
         np.sin(yaw) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll)],
        [np.sin(yaw) * np.cos(pitch), 
         np.cos(yaw) * np.cos(roll) + np.sin(yaw) * np.sin(pitch) * np.sin(roll),
         -np.cos(yaw) * np.sin(roll) + np.sin(yaw) * np.sin(pitch) * np.cos(roll)],
        [-np.sin(pitch),
         np.cos(pitch) * np.sin(roll),
         np.cos(pitch) * np.cos(roll)]
    ])
    rotated_curve = np.dot(rotation_matrix, np.vstack((x, y, z)))
    return [rotated_curve[0, :], rotated_curve[1, :], rotated_curve[2, :]]

def plot_attractor_fit(eeg_curve, ross_curve, title="Rössler-EEG Fit Comparison"):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*ross_curve, label='Rössler Attractor', color='blue', alpha=0.6)
    ax.plot(*eeg_curve, label='Transformed EEG', color='red', alpha=0.8)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

class RosslerFitter:
    def __init__(self, raw_eeg, ross_data_scaled, num_pts_eeg=256, num_pts_ross=100000, time_end=550):
        self.raw_eeg = raw_eeg
        self.ross_data_scaled = ross_data_scaled
        self.num_pts_eeg = num_pts_eeg
        self.num_pts_ross = num_pts_ross
        self.time_end = time_end
        self.eeg_ind_start = 1
        self.t_eeg = np.linspace(0, 1, num_pts_eeg)
        self.search_space = {
            'd1': hp.quniform('d1', 0, raw_eeg._data.shape[0]-1, 1),
            'd2': hp.quniform('d2', 0, raw_eeg._data.shape[0]-1, 1),
            'd3': hp.quniform('d3', 0, raw_eeg._data.shape[0]-1, 1),
            't1': hp.uniform('t1', -1, 1),
            't2': hp.uniform('t2', -1, 1),
            't3': hp.uniform('t3', -1, 1),
            'r1': hp.uniform('r1', -np.pi, np.pi),
            'r2': hp.uniform('r2', -np.pi, np.pi),
            'r3': hp.uniform('r3', -np.pi, np.pi)
        }

    def objective(self, params):
        d1, d2, d3 = int(params['d1']), int(params['d2']), int(params['d3'])
        t1, t2, t3 = params['t1'], params['t2'], params['t3']
        r1, r2, r3 = params['r1'], params['r2'], params['r3']
        if d1 == d2 or d1 == d3 or d2 == d3:
            return float('inf')
        eeg_data = self.raw_eeg._data[[d1, d2, d3], self.eeg_ind_start:self.num_pts_eeg + self.eeg_ind_start]
        eeg_data = (eeg_data - np.min(eeg_data, axis=1, keepdims=True)) / (np.max(eeg_data, axis=1, keepdims=True) - np.min(eeg_data, axis=1, keepdims=True))
        bbc = calculate_bounding_box_center(*eeg_data)
        eeg_data = eeg_data - bbc.reshape(-1, 1)
        eeg_data = rotate_curve(eeg_data, r1, r2, r3)
        eeg_data = eeg_data + bbc.reshape(-1, 1)
        eeg_data[0] += t1
        eeg_data[1] += t2
        eeg_data[2] += t3
        t_eeg_inds = [int(num / self.time_end * self.num_pts_ross) for num in self.t_eeg]
        dist = np.sum(np.linalg.norm(eeg_data - self.ross_data_scaled[:, t_eeg_inds], axis=0))
        return dist

    def fit(self, num_trials=100):
        result = fmin(fn=self.objective, space=self.search_space, algo=tpe.suggest, max_evals=num_trials,
                      rstate=np.random.default_rng(42))
        self.result = result
        d1, d2, d3 = int(result['d1']), int(result['d2']), int(result['d3'])
        t1, t2, t3 = result['t1'], result['t2'], result['t3']
        r1, r2, r3 = result['r1'], result['r2'], result['r3']
        eeg_data = self.raw_eeg._data[[d1, d2, d3], self.eeg_ind_start:self.num_pts_eeg + self.eeg_ind_start]
        eeg_data = (eeg_data - np.min(eeg_data, axis=1, keepdims=True)) / (np.max(eeg_data, axis=1, keepdims=True) - np.min(eeg_data, axis=1, keepdims=True))
        bbc = calculate_bounding_box_center(*eeg_data)
        eeg_data = eeg_data - bbc.reshape(-1, 1)
        eeg_data = rotate_curve(eeg_data, r1, r2, r3)
        eeg_data = eeg_data + bbc.reshape(-1, 1)
        eeg_data[0] += t1
        eeg_data[1] += t2
        eeg_data[2] += t3
        self.eeg_transformed = eeg_data
        return eeg_data, result
