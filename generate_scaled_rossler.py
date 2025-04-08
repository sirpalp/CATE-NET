import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def rossler_attractor(t, X, a=0.1, b=0.1, c=14.0):
    x, y, z = X
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]

def generate_scaled_rossler(time_end=550, num_pts=100000, save_path="scaled_rossler.npy", plot=False):
    t_span = [0, time_end]
    t_eval = np.linspace(t_span[0], t_span[1], num_pts)
    sol = solve_ivp(rossler_attractor, t_span, [0, 1, 0], t_eval=t_eval)
    ross_data = sol.y
    row_mins = np.min(ross_data, axis=1, keepdims=True)
    row_maxs = np.max(ross_data, axis=1, keepdims=True)
    ross_scaled = (ross_data - row_mins) / (row_maxs - row_mins)
    np.save(save_path, ross_scaled)
    print(f"Saved scaled Rössler attractor to: {save_path}")
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*ross_scaled, alpha=0.8)
        ax.set_title("Scaled Rössler Attractor")
        plt.show()

if __name__ == "__main__":
    generate_scaled_rossler(plot=True)