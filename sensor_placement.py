"""
Calculation of optimal sensor placements from a FE solution.
"""

import sys

from os import path
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp

from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.tracebacklimit = 0

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
    raise Exception(f"usage: python3 {argv[0]} datafile [n_PODs [n_sensors [nX|pixelfile]]]"
                    " [-show-mean] [-no-anim]")
if not path.isfile(argv[1]):
    raise Exception(f"{argv[1]}: No such file")

# Loading data
D = np.loadtxt(argv[1])
X = np.transpose(D[:,1:])  # remove the first (time) column
# The rows of X are now the image pixels / sampling points
# and the columns are the time instances
print("Dimension sensor data array:", X.shape)
x_min = np.min(X)
x_max = np.max(X)

# Choose parameters for the sensor placement
# POD components
n_POD_components = 5 if len(argv) < 3 else int(argv[2])

# Sensors
# Number of sensors has to be larger or equal to the number of POD components
n_sensors = max(10 if len(argv) < 4 else int(argv[3]), n_POD_components)
print(f"n_POD_components={n_POD_components} n_sensors={n_sensors}")

def calculate_POD(X, n, mean_centering=True):
    """Function for finding first n_POD_components using svd."""
    if mean_centering:
        # Find the mean over the whole time series
        X_mean = np.mean(X, axis=1, keepdims=True)
        # Subtract the mean values from each time instance
        X_ = X - X_mean
    else:
        X_ = X
        X_mean = np.zeros((X.shape[0],1))
    U, S, V = np.linalg.svd(X_, full_matrices=False)
    return U[:,:n], X_mean

# Finding POD basis (Psi_r)
Psi_r, X_mean = calculate_POD(X, n_POD_components)
print(f"Range mean original data: [{np.min(X_mean)},{np.max(X_mean)}]")

def get_img_size(picx, n_pts):
    """Function for finding pixel frame dimensions."""
    if path.isfile(picx):
        idx = np.transpose(np.loadtxt(picx))
        return idx.shape[0], idx.shape[1], idx
    nX = int(picx)
    return nX, int(n_pts / nX), None

# Sampling points in each coordinate direction
# nX*nY has to equal the number of rows in X
n_pts = X.shape[0]
nX, nY, idx = (50, int(n_pts / 50), None) if len(argv) < 5 else get_img_size(argv[4], n_pts)
print(f"Frame dimension: {nX}x{nY}")

# Use the "jet" colormap, but with the out-of-range pixels white
mymap = plt.cm.get_cmap("jet").copy()
mymap.set_under("w")

def remap(X, empty=-1):
    """
    Function for expanding the 1D array or list into a 2D raster image.
    Note: Using nX, nY and idx as global variables.
    """
    if idx is None:
        return X.reshape(nY, nX) if isinstance(X, np.ndarray) else np.array(X).reshape(nY, nX)
    data = np.full((nY, nX), float(empty))
    for j in range(nY):
        for i in range(nX):
            if idx[i,j] > 0:
                data[nY-1-j,i] = X[int(idx[i,j])-1]
    return data

# Showing the mean and the first 8 POD components
n_show = 9 - show_mean if n_POD_components >  9 - show_mean else n_POD_components
y_max = max(-x_min, x_max)
plt.subplot(3, 3, 1)
if show_mean:
    plt.imshow(remap(X_mean), vmin=-y_max, vmax=y_max, cmap=mymap)
for i in range(n_show):
    y_max = np.max(np.abs(Psi_r[:,i]))
    plt.subplot(3, 3, 1+show_mean+i)
    plt.imshow(remap(Psi_r[:,i]), vmin=-y_max, vmax=y_max, cmap=mymap)
    print(f"POD {i+1} : [{np.min(Psi_r[:,i])}, {np.max(Psi_r[:,i])}]")
plt.show()

def find_sensor_placement_QR(Psi_r, num_eigen, num_sensors):
    """Function for finding optimal sensor placement."""
    M = Psi_r @ Psi_r.T if num_sensors > num_eigen else Psi_r.T
    Q, R, P = sp.qr(M, pivoting=True)
    C = np.zeros((num_sensors, Psi_r.shape[0]))
    C[np.arange(num_sensors), P[:num_sensors]] = 1
    return C

# Finding the sensor placement matrix C
C = find_sensor_placement_QR(Psi_r, n_POD_components, n_sensors)

# Showing the sensor placement
plt.imshow(remap(np.sum(C, axis=0), empty=0), cmap="gray")
plt.show()
# The white pixels are the pixels in the image where one measures the image

# Create measurements sensor placement
Y = np.dot(C, X - X_mean)

# Reconstruct from the measurements
Theta = np.linalg.pinv(C @ Psi_r)
X_hat = np.dot(Psi_r, np.dot(Theta, Y)) + X_mean
X_err = X - X_hat

X_norm = np.linalg.norm(X, axis=0)
Y_norm = np.linalg.norm(X_hat, axis=0)
E_norm = np.linalg.norm(X_err, axis=0)
x_norm = np.max(X_norm)
y_norm = np.max(Y_norm)
e_norm = np.max(E_norm)
ix_pos = np.argmax(X_norm)
iy_pos = np.argmax(Y_norm)
ie_pos = np.argmax(E_norm)

print("Range original data:      "
      f"[{x_min},{x_max}] Max L2-norm: {x_norm} at step {ix_pos+1}")
print("Range reconstructed data: "
      f"[{np.min(X_hat)},{np.max(X_hat)}] Max L2-norm: {y_norm} at step {iy_pos+1}")
print("Error range:              "
      f"[{np.min(X_err)},{np.max(X_err)}] Max L2-norm: {e_norm} at step {ie_pos+1}")

print(f"Max relative error: {100*e_norm/X_norm[ie_pos]}% of L2(X)={X_norm[ie_pos]}")

# Show image measurement and reconstruction of one (random) image
image_index = int(argv[5]) if len(argv) > 5 else np.random.choice(X.shape[1])
if image_index == -1:  # show frame with max solution norm
    image_index = ix_pos
elif image_index == -2:  # show frame with max error
    image_index = ie_pos
print("Showing the reconstruction of image", image_index)

v_min = np.min(X[:,image_index])
v_max = np.max(X[:,image_index])

plt.subplot(4, 1, 1)  # the original image
plt.imshow(remap(X[:,image_index]), vmin=min(v_min,-v_max), vmax=max(v_max,-v_min), cmap=mymap)

y_img = (Y[:,image_index]*C.T).T.sum(axis=0)
y_img = [ 0.0 if y_val == 0 else 0.6 for y_val in y_img ]

plt.subplot(4, 1, 2)  # the measurements
plt.imshow(remap(y_img), vmin=-1, vmax=1, cmap="seismic")

plt.subplot(4, 1, 3)  # the reconstructed image
plt.imshow(remap(X_hat[:,image_index]), vmin=min(v_min,-v_max), vmax=max(v_max,-v_min), cmap=mymap)

e_min = np.min(X_err)  # 0.1*x_min
e_max = np.max(X_err)  # 0.1*x_max
plt.subplot(4, 1, 4)  # error plot
plt.imshow(remap(X_err[:,image_index]), vmin=min(e_min,-e_max), vmax=max(e_max,-e_min), cmap=mymap)

plt.show()

if not show_anim:
    sys.exit(0)

# Now do some animation of the original vs. the reconstructed image

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
img1 = ax1.imshow(remap(X[:,0])    , vmin=x_min, vmax=x_max, cmap=mymap, animated=True)
img2 = ax2.imshow(remap(X_hat[:,0]), vmin=x_min, vmax=x_max, cmap=mymap, animated=True)
img3 = ax3.imshow(remap(X_err[:,0]), vmin=e_min, vmax=e_max, cmap=mymap, animated=True)

def update_fig(i):
    """Creates a new frame for the animation. """
    img1.set_array(remap(X[:,i]))
    img2.set_array(remap(X_hat[:,i]))
    img3.set_array(remap(X_err[:,i]))
    fig.suptitle(f'Frame #{i}', fontsize=12)
    return img1, img2, img3, fig

mpfile = argv[1].rsplit(".", 1)[0] + "_" + str(n_POD_components) + "POD.mp4"
print("Saving animation to", mpfile, "...")
nfr = 100 # X.shape[1]
fps = 5   # Frames per second
wrt = FFMpegWriter(fps=fps)
ani = FuncAnimation(fig, update_fig, frames=nfr, interval=1000/fps)
ani.save(mpfile, writer=wrt)
plt.show()
