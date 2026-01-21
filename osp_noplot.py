"""
Calculation of optimal sensor placements from a FE solution.
"""

from os import path
from sys import argv

import numpy as np
import scipy.linalg as sp

from osp_core import OptimalSensorPlacement

if len(argv) < 2:
    raise Exception(f"usage: python {argv[0]} datafile"
                    " [n_PODs [n_sensor [n_ignore [sensor_list]]]]")
if not path.isfile(argv[1]):
    raise Exception(f"{argv[1]}: No such file")

# Number of initial steps to skip
n_skip = 0 if len(argv) < 5 else int(argv[4])

# Select which algoritm to use
USE_OSP_CORE = len(argv) > 5

# Loading data
D = np.loadtxt(argv[1])
X = np.transpose(D[n_skip:,1:])  # remove the first (time) column and n_skip rows
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
n_sensors = 10 if len(argv) < 4 else int(argv[3])

print(f"n_POD_components={n_POD_components} n_sensors={n_sensors} n_skip={n_skip}")

# Predefined sensors
sensor_list = []
if USE_OSP_CORE and path.isfile(argv[5]):
    with open(argv[5], 'r', encoding='utf-8') as fd:
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
    U, S, _ = np.linalg.svd(X_, full_matrices=False)
    print("POD singular values:", S[:n])
    return U[:,:n], X_mean

def find_sensor_placement_QR(num_eigen, num_sensors):
    """
    Function for finding optimal sensor placement.
    Note: Using Psi_r as global variable.
    """
    M = Psi_r @ Psi_r.T if num_sensors > num_eigen else Psi_r.T
    _, _, P = sp.qr(M, pivoting=True)
    S = np.zeros((num_sensors, Psi_r.shape[0]))
    S[np.arange(num_sensors), P[:num_sensors]] = 1
    return S, P[:num_sensors]

if USE_OSP_CORE:  # Using Adil Rasheed's new OSP module
    osp = OptimalSensorPlacement(n_POD_components)

    # Find the mean over the whole time series
    X_MEAN = np.mean(X, axis=1, keepdims=True)
    # Subtract the mean values from each time instance
    osp.fit(X - X_MEAN)

    n_total = len(sensor_list) + n_sensors
    if n_total > n_sensors:
        print(f"Preselected sensors: {sensor_list}")
        print(f"Adding {n_sensors} new sensors")
        # Method 1: Conditional OSP (Algorithm 1)
        locs = osp.select_sensors(n_total, preselected=sensor_list)
    else:
        # Method 2: Fresh OSP (start from scratch)
        locs = osp.select_sensors(n_total)
    print("All sensor locations:", locs)

    # Get sensor measurements
    Y = X[locs, :] - X_MEAN[locs, :]

    # Reconstruct from the measurements
    X_hat = osp.reconstruct(Y, locs) + X_MEAN

else:  # Using the old procedure (should be similar to Method 2 above)

    # Finding POD basis (Psi_r)
    Psi_r, X_MEAN = calculate_POD(n_POD_components)
    print(f"Range mean original data: [{np.min(X_MEAN)},{np.max(X_MEAN)}]")

    # Finding the sensor placement matrix C
    C, locs = find_sensor_placement_QR(n_POD_components, n_sensors)
    print("Sensor locations:", locs)

    # Create measurements sensor placement
    Y = np.dot(C, X - X_MEAN)

    # Reconstruct from the measurements
    Theta = np.linalg.pinv(C @ Psi_r)
    X_hat = np.dot(Psi_r, np.dot(Theta, Y)) + X_MEAN

X_err = X - X_hat

X_norm = np.linalg.norm(X, axis=0)
Y_norm = np.linalg.norm(X_hat, axis=0)
E_norm = np.linalg.norm(X_err, axis=0)
x_norm = np.max(X_norm)
y_norm = np.max(Y_norm)
e_norm = np.max(E_norm)
r_norm = E_norm / X_norm
ix_pos = np.argmax(X_norm)
iy_pos = np.argmax(Y_norm)
ie_pos = np.argmax(E_norm)
ir_pos = np.argmax(r_norm)
Xn_ref = np.mean(X_norm)

print(f"Range original data:      [{x_min},{x_max}] Max L2-norm: {x_norm} at step {ix_pos+1}")
print(f"Range reconstructed data: [{np.min(X_hat)},{np.max(X_hat)}]"
      f" Max L2-norm: {y_norm} at step {iy_pos+1}")
print(f"Error range:              [{np.min(X_err)},{np.max(X_err)}]"
      f" Max L2-norm: {e_norm} at step {ie_pos+1}",
      f" ({100*e_norm/X_norm[ie_pos]}% of L2(X)={X_norm[ie_pos]})")
print(f"Max relative error: {100*np.max(r_norm)}% of L2(X)={X_norm[ir_pos]} at step {ir_pos+1}",
      f" or {100*e_norm/Xn_ref}% of mean(L2(X))={Xn_ref}")

# Save to file
filenam = argv[1].rsplit(".", 1)[0] + "_n" + str(n_sensors)
hatfile = filenam + "_hat.dat"
errfile = filenam + "_err.dat"
locfile = filenam + "_loc.dat"
np.savetxt(hatfile,np.transpose(np.insert(X_hat,0,D[n_skip:,0],axis=0)),fmt="%.6g")
np.savetxt(errfile,np.transpose(np.insert(X_err,0,D[n_skip:,0],axis=0)),fmt="%.6g")
np.savetxt(locfile,locs,fmt="%i")
