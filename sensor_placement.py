"""
Calculation of optimal sensor placements from a FE solution.
"""

from os import path
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
from matplotlib.animation import FFMpegWriter, FuncAnimation

from osp_core import OptimalSensorPlacement

if "-show-mean" in argv:
    argv.remove("-show-mean")
    show_mean = 1
else:
    show_mean = 0
if "-no-anim" in argv:
    argv.remove("-no-anim")
    show_anim = False
else:
    show_anim = True

if len(argv) < 2:
    raise Exception(f"usage: python {argv[0]} datafile [n_PODs [n_sensors [nX|pixelfile [n_skip]]]]"
                    " [-show-mean] [-no-anim]")
if not path.isfile(argv[1]):
    raise Exception(f"{argv[1]}: No such file")

# Number of initial steps to skip
n_skip = 0 if len(argv) < 6 else int(argv[5])

# Loading data
D = np.loadtxt(argv[1])
X = np.transpose(D[n_skip:,1:])  # remove the first (time) column and n_skip rows
# The rows of X are now the image pixels / sampling points
# and the columns are the time instances
print("Dimension sensor data array:", X.shape)
x_min = np.min(X)
x_max = np.max(X)

# Which frame to plot results for
image_index = int(argv[6]) if len(argv) > 6 else np.random.choice(X.shape[1])

# Choose parameters for the sensor placement
# POD components
n_POD_components = 5 if len(argv) < 3 else int(argv[2])

# Sensors
# Number of sensors has to be larger or equal to the number of POD components
n_sensors = max(10 if len(argv) < 4 else int(argv[3]), n_POD_components)
print(f"n_POD_components={n_POD_components} n_sensors={n_sensors}")

# Predefined sensors
sensor_list = []
if len(argv) > 7 and path.isfile(argv[7]):
    with open(argv[7], 'r', encoding='utf-8') as fd:
        sensors = fd.read()
        sensor_list = [int(n) for n in sensors.split()]

def calculate_POD(n, mean_centering=True):
    """
    Function for finding first n_POD_components using svd.
    Note: Using X as global variable.
    """
    if mean_centering:
        # Find the mean over the whole time series
        X_mean = np.mean(X, axis=1, keepdims=True)
        # Subtract the mean values from each time instance
        X_ = X - X_mean
    else:
        X_ = X
        X_mean = np.zeros((X.shape[0],1))
    U, _, _ = np.linalg.svd(X_, full_matrices=False)
    return U[:,:n], X_mean

# Finding POD basis (Psi_r)
Psi_r, X_MEAN = calculate_POD(n_POD_components)
print(f"Range mean original data: [{np.min(X_MEAN)},{np.max(X_MEAN)}]")

def get_img_size(picx, n_pt):
    """Function for finding pixel frame dimensions."""
    if path.isfile(picx):
        indx = np.transpose(np.loadtxt(picx))
        return indx.shape[0], indx.shape[1], indx
    n_x = int(picx)
    return n_x, int(n_pt / n_x), None

# Sampling points in each coordinate direction
# nX*nY has to equal the number of rows in X
n_pts = X.shape[0]
nX, nY, idx = (50, int(n_pts / 50), None) if len(argv) < 5 else get_img_size(argv[4], n_pts)
print(f"Frame dimension: {nX}x{nY}")

# Use the "jet" colormap, but with the out-of-range pixels white
mymap = plt.cm.get_cmap("jet").copy()
mymap.set_under("w")

def remap(x, empty=-1):
    """
    Function for expanding the 1D array or list into a 2D raster image.
    Note: Using nX, nY and idx as global variables.
    """
    if idx is None:
        return x.reshape(nY, nX) if isinstance(x, np.ndarray) else np.array(x).reshape(nY, nX)
    data = np.full((nY, nX), float(empty))
    for j in range(nY):
        for i in range(nX):
            if idx[i,j] > 0:
                data[nY-1-j,i] = x[int(idx[i,j])-1]
    return data

# Showing the mean and the first 8 POD components
n_show = 9 - show_mean if n_POD_components >  9 - show_mean else n_POD_components
y_max = max(-x_min, x_max)
plt.subplot(3, 3, 1)
if show_mean:
    plt.imshow(remap(X_MEAN), vmin=-y_max, vmax=y_max, cmap=mymap)
for k in range(n_show):
    y_max = np.max(np.abs(Psi_r[:,k]))
    plt.subplot(3, 3, 1+show_mean+k)
    plt.imshow(remap(Psi_r[:,k]), vmin=-y_max, vmax=y_max, cmap=mymap)
    print(f"POD {k+1} : [{np.min(Psi_r[:,k])}, {np.max(Psi_r[:,k])}]")
plt.show()

def find_sensor_placement_QR(num_eigen, num_sensors):
    """
    Function for finding optimal sensor placement.
    Note: Using Psi_r as global variable.
    """
    M = Psi_r @ Psi_r.T if num_sensors > num_eigen else Psi_r.T
    _, _, P = sp.qr(M, pivoting=True)
    c = np.zeros((num_sensors, Psi_r.shape[0]))
    c[np.arange(num_sensors), P[:num_sensors]] = 1
    return c

# Finding the sensor placement matrix C
C = find_sensor_placement_QR(n_POD_components, n_sensors)

# Showing the sensor placement
plt.imshow(remap(np.sum(C, axis=0), empty=0), cmap="gray")
plt.show()
# The white pixels are the pixels in the image where one measures the image

# Create measurements sensor placement
Y = np.dot(C, X - X_MEAN)

# Reconstruct from the measurements
Theta = np.linalg.pinv(C @ Psi_r)
X_hat = np.dot(Psi_r, np.dot(Theta, Y)) + X_MEAN
X_err = X - X_hat

def print_range(heading, V, V_ref=None):
    """Prints some characteristics of a numerical vector."""
    _x_min = np.min(V)
    _x_max = np.max(V)
    x_norm = np.linalg.norm(V, axis=0)
    x_nmax = np.max(x_norm)
    ix_pos = np.argmax(x_norm)
    if V_ref is None:
        print(f"{heading} [{_x_min},{_x_max}] Max L2-norm: {x_nmax} at step {ix_pos+1}")
    else:
        x_ref = np.linalg.norm(V_ref, axis=0)[ix_pos]
        print(f"{heading} [{_x_min},{_x_max}] Max L2-norm: {x_nmax} at step {ix_pos+1}"
              f" ({100*x_nmax/x_ref}% of L2(X)={x_ref})")

print_range("Range original data:     ", X)
print_range("Range reconstructed data:", X_hat)
print_range("Error range:             ", X_err, X)

X_norm = np.linalg.norm(X, axis=0)
E_norm = np.linalg.norm(X_err, axis=0)
r_norm = E_norm / X_norm
ir_pos = np.argmax(r_norm)
Xn_ref = np.mean(X_norm)
print(f"Max relative error: {100*np.max(r_norm)}% of L2(X)={X_norm[ir_pos]} at step {ir_pos+1}",
      f" or {100*np.max(E_norm)/Xn_ref}% of mean(L2(X))={Xn_ref}")

def plot_results(S, img_index):
    """
    Plots the OSP results.
    Note: Using X, Y, X_hat, X_err, X_norm, E_norm and r_norm as global variables.
    """
    # Show image measurement and reconstruction of one (random) image
    if img_index == -1:  # show frame with max solution norm
        iidx = np.argmax(X_norm)
    elif img_index == -2:  # show frame with max error
        iidx = np.argmax(E_norm)
    elif img_index == -3:  # show frame with max relative error
        iidx = np.argmax(r_norm)
    else:
        iidx = img_index
    print("Showing the reconstruction of image", iidx+1)

    v_min = np.min(X[:,iidx])
    v_max = np.max(X[:,iidx])

    plt.subplot(4, 1, 1)  # the original image
    plt.imshow(remap(X[:,iidx]), vmin=min(v_min,-v_max), vmax=max(v_max,-v_min), cmap=mymap)

    y_img = (Y[:,iidx]*S.T).T.sum(axis=0)
    y_img = [ 0.0 if y_val == 0 else 0.6 for y_val in y_img ]

    plt.subplot(4, 1, 2)  # the measurements
    plt.imshow(remap(y_img), vmin=-1, vmax=1, cmap="seismic")

    plt.subplot(4, 1, 3)  # the reconstructed image
    plt.imshow(remap(X_hat[:,iidx]), vmin=min(v_min,-v_max), vmax=max(v_max,-v_min), cmap=mymap)

    _min = np.min(X_err)
    _max = np.max(X_err)
    plt.subplot(4, 1, 4)  # error plot
    plt.imshow(remap(X_err[:,iidx]), vmin=min(_min,-_max), vmax=max(_max,-_min), cmap=mymap)

    plt.show()

plot_results(C, image_index)

if show_anim:
    # Now do some animation of the original vs. the reconstructed image
    e_min = np.min(X_err)
    e_max = np.max(X_err)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    img1 = ax1.imshow(remap(X[:,0])    , vmin=x_min, vmax=x_max, cmap=mymap, animated=True)
    img2 = ax2.imshow(remap(X_hat[:,0]), vmin=x_min, vmax=x_max, cmap=mymap, animated=True)
    img3 = ax3.imshow(remap(X_err[:,0]), vmin=e_min, vmax=e_max, cmap=mymap, animated=True)

    def update_fig(i):
        """Creates a new frame for the animation."""
        img1.set_array(remap(X[:,i]))
        img2.set_array(remap(X_hat[:,i]))
        img3.set_array(remap(X_err[:,i]))
        fig.suptitle(f'Frame #{i}', fontsize=12)
        return img1, img2, img3, fig

    mpfile = argv[1].rsplit(".", 1)[0] + "_" + str(n_POD_components) + "POD.mp4"
    print("Saving animation to", mpfile, "...")
    nfr = X.shape[1] if X.shape[1] < 100 else 100
    fps = 5   # Frames per second
    wrt = FFMpegWriter(fps=fps)
    ani = FuncAnimation(fig, update_fig, frames=nfr, interval=1000/fps)
    ani.save(mpfile, writer=wrt)
    plt.show()

if len(sensor_list) > 0:  # Using Adil Rasheed's new OSP module
    osp = OptimalSensorPlacement(len(sensor_list) + n_POD_components)
    osp.fit(X - X_MEAN)

    n_total = len(sensor_list) + n_sensors
    print(f"Preselected sensors: {sensor_list}")
    print(f"Adding {n_sensors} new sensors")
    locs = osp.select_sensors(n_total, preselected=sensor_list)
    print("All sensor locations:", locs)

    # Get sensor measurements
    Y = X[locs, :] - X_MEAN[locs, :]

    # Reconstruct from the measurements
    X_hat = osp.reconstruct(Y, locs) + X_MEAN
    X_err = X - X_hat
    print_range("Range original data:     ", X)
    print_range("Range reconstructed data:", X_hat)
    print_range("Error range:             ", X_err, X)

    X_norm = np.linalg.norm(X, axis=0)
    E_norm = np.linalg.norm(X_err, axis=0)
    r_norm = E_norm / X_norm
    ir_pos = np.argmax(r_norm)
    Xn_ref = np.mean(X_norm)
    print(f"Max relative error: {100*np.max(r_norm)}% of L2(X)={X_norm[ir_pos]} at step {ir_pos+1}",
          f" or {100*np.max(E_norm)/Xn_ref}% of mean(L2(X))={Xn_ref}")

    plot_results(osp.get_measurement_matrix(locs), image_index)
