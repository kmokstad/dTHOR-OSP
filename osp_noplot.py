"""
Calculation of optimal sensor placements from a FE solution.
"""

import numpy as np
import scipy.linalg as sp

from os import path
from sys import argv

if len(argv) < 2:
    raise Exception(f"usage: python3 {argv[0]} datafile [n_PODs [n_sensors]]")
elif not path.isfile(argv[1]):
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
n_sensors = 10 if len(argv) < 4 else int(argv[3])

print(f"n_POD_components={n_POD_components} n_sensors={n_sensors}")

# Function for finding first n_POD_components using svd
def calculate_POD(X, n, mean_centering=True):
    if mean_centering:
        # Find the mean over the whole time series
        X_mean = np.mean(X, axis=1, keepdims=True)
        # Subtract the mean values from each time instance
        X_ = X - X_mean
    else:
        X_ = X
        X_mean = np.zeros((X.shape[0],1))
    U, S, V = np.linalg.svd(X_, full_matrices=False)
    print("POD singular values:", S[:n])
    return U[:,:n], X_mean

# Finding POD basis (Psi_r)
Psi_r, X_mean = calculate_POD(X, n_POD_components)
print(f"Range mean original data: [{np.min(X_mean)},{np.max(X_mean)}]")

# Function for finding optimal sensor placement
def find_sensor_placement_QR(Psi_r, num_eigen, num_sensors):
    M = Psi_r @ Psi_r.T if num_sensors > num_eigen else Psi_r.T
    Q, R, P = sp.qr(M, pivoting=True)
    C = np.zeros((num_sensors, Psi_r.shape[0]))
    C[np.arange(num_sensors), P[:num_sensors]] = 1
    return C, P[:num_sensors]

# Finding the sensor placement matrix C
C, P = find_sensor_placement_QR(Psi_r, n_POD_components, n_sensors)
print("Sensor locations:", P)

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

print(f"Range original data:      [{x_min},{x_max}] Max L2-norm: {x_norm} at step {ix_pos+1}")
print(f"Range reconstructed data: [{np.min(X_hat)},{np.max(X_hat)}] Max L2-norm: {y_norm} at step {iy_pos+1}")
print(f"Error range:              [{np.min(X_err)},{np.max(X_err)}] Max L2-norm: {e_norm} at step {ie_pos+1}")
print(f"Max relative error: {100*e_norm/X_norm[ie_pos]}% of L2(X)={X_norm[ie_pos]}")

# Save to file
filenam = argv[1].rsplit(".", 1)[0] + "_n" + str(n_sensors)
hatfile = filenam + "_hat.dat"
errfile = filenam + "_err.dat"
locfile = filenam + "_loc.dat"
np.savetxt(hatfile,np.transpose(np.insert(X_hat,0,D[:,0],axis=0)))
np.savetxt(errfile,np.transpose(np.insert(X_err,0,D[:,0],axis=0)))
np.savetxt(locfile,P,fmt='%i')
