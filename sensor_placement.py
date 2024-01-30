"""
Calculation of optimal sensor placements from a FE solution.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp

from matplotlib.animation import FuncAnimation
from os import path
from sys import argv

if len(argv) < 2:
    raise Exception(f"usage: python3 {argv[0]} datafile [n_PODs [n_sensors [nX]]]")
elif not path.isfile(argv[1]):
    raise Exception(f"{argv[1]}: No such file")

# Loading data
D = np.loadtxt(argv[1])
X = np.transpose(D[:,1:])  # remove the first (time) column
# The rows of X are now the image pixels / sampling points
# and the columns are the time instances
print("Dimension sensor data array:", X.shape)

# Choose parameters for the sensor placement
# POD components
n_POD_components = 5 if len(argv) < 3 else int(argv[2])

# Sensors
# Number of sensors has to be larger or equal to the number of POD components
n_sensors = 10 if len(argv) < 4 else int(argv[3])

# Sampling points in each coordinate direction
# nX*nY has to equal the number of rows in X
nX = 50 if len(argv) < 5 else int(argv[4])
nY = int(X.shape[0] / nX)
print(f"n_POD_components={n_POD_components} n_sensors={n_sensors} nX={nX} nY={nY}")

# Function for finding first n_POD_components using svd
def calculate_POD(X, n, mean_centering=True):
    if mean_centering:
        # Find the mean over the while time series
        X_mean = np.mean(X, axis=1, keepdims=True)
        # Subtract the mean values from each time instance
        X_ = X - np.tile(X_mean, (1,X.shape[1]))
    else:
        X_ = X
        X_mean = np.zeros((X.shape[0],1))
    U, S, V = np.linalg.svd(X_)
    return U[:,:n], X_mean

# Finding POD basis (Psi_r)
Psi_r, X_mean = calculate_POD(X, n_POD_components)

# Showing the mean and the first 8 POD components
n_show = 8 if n_POD_components > 8 else n_POD_components
plt.subplot(3, 3, 1)
plt.imshow(X_mean.reshape(nY,nX), cmap="jet")
for i in range(n_show):
    plt.subplot(3, 3, i+2)
    plt.imshow(Psi_r[:,i].reshape(nY,nX), cmap="jet")
plt.show()

# Function for finding optimal sensor placement 
def find_sensor_placement_QR(Psi_r, num_eigen, num_sensors):
    M = Psi_r @ Psi_r.T if num_sensors > num_eigen else Psi_r.T
    Q, R, P = sp.qr(M, pivoting=True)
    C = np.zeros((num_sensors, Psi_r.shape[0]))
    C[np.arange(num_sensors), P[:num_sensors]] = 1
    return C

# Finding the sensor placement matrix C
C = find_sensor_placement_QR(Psi_r, n_POD_components, n_sensors)

# Showing the sensor placement
plt.imshow(np.sum(C, axis=0).reshape(nY,nX), cmap='gray')
plt.show()
# The white pixels are the pixels in the image where one measures the image

# Create measurements sensor placement
Y = np.dot(C, (X - np.tile(X_mean, (1,X.shape[1]))))

# Reconstruct from the measurements
Theta = np.linalg.pinv(C @ Psi_r)
X_hat = np.dot(Psi_r, np.dot(Theta, Y)) + np.tile(X_mean, (1,Y.shape[1]))
X_err = X - X_hat
x_min = np.min(X)
x_max = np.max(X)
e_min = 0.1*x_min
e_max = 0.1*x_max

X_norm = np.linalg.norm(X, axis=0)
Y_norm = np.linalg.norm(X_hat, axis=0)
E_norm = np.linalg.norm(X_err, axis=0)
x_norm = np.max(X_norm)
y_norm = np.max(Y_norm)
e_norm = np.max(E_norm)
ix_pos = np.argmax(X_norm)
iy_pos = np.argmax(Y_norm)
ie_pos = np.argmax(E_norm)

print(f"Range original data:      [{x_min},{x_max}] Max L2-norm: {x_norm} at step {ix_pos+1}")
print(f"Range reconstructed data: [{np.min(X_hat)},{np.max(X_hat)}] Max L2-norm: {y_norm} at step {iy_pos+1}")
print(f"Error range:              [{np.min(X_err)},{np.max(X_err)}] Max L2-norm: {e_norm} at step {ie_pos+1}")
print(f"Max relative error: {100*e_norm/X_norm[ie_pos]}% of L2(X)={X_norm[ie_pos]}")

# Show image measurement and reconstruction of one (random) image
image_index = int(argv[5]) if len(argv) > 5 else np.random.choice(X.shape[1])
print("Showing the reconstruction of image", image_index)

plt.subplot(4, 1, 1)  # the original image
plt.imshow(X[:,image_index].reshape(nY,nX), vmin=x_min, vmax=x_max, cmap="jet")

y_img = (Y[:,image_index]*C.T).T.sum(axis=0).reshape(nY,nX)
y_max = np.max(np.abs(y_img))
plt.subplot(4, 1, 2)  # the measurements
plt.imshow(y_img, cmap="seismic", vmin=-y_max, vmax=y_max)

plt.subplot(4, 1, 3)  # the reconstructed image
plt.imshow(X_hat[:,image_index].reshape(nY,nX), vmin=x_min, vmax=x_max, cmap="jet")

plt.subplot(4, 1, 4)  # error plot
plt.imshow(X_err[:,image_index].reshape(nY,nX), vmin=e_min, vmax=e_max, cmap="jet")

plt.show()

# Now do some animation of the original vs. the reconstructed image

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
img1 = ax1.imshow(X[:,0].reshape(nY,nX)    , vmin=x_min, vmax=x_max, cmap="jet", animated=True)
img2 = ax2.imshow(X_hat[:,0].reshape(nY,nX), vmin=x_min, vmax=x_max, cmap="jet", animated=True)
img3 = ax3.imshow(X_err[:,0].reshape(nY,nX), vmin=e_min, vmax=e_max, cmap="jet", animated=True)

def update_fig(i):
    img1.set_array(X[:,i].reshape(nY,nX))
    img2.set_array(X_hat[:,i].reshape(nY,nX))
    img3.set_array(X_err[:,i].reshape(nY,nX))
    fig.suptitle(f'Frame #{i}', fontsize=12)
    return img1, img2, img3, fig

nfr = 100 # X.shape[1]
fps = 5   # Frames per second
ani = FuncAnimation(fig, update_fig, frames=nfr, interval=1000/fps)
plt.show()
