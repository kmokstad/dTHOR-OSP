"""
Calculation of optimal sensor placements from a set of modes from ROM analysis.
"""

import numpy as np
import scipy.linalg as sp

from os import path
from sys import argv

if len(argv) < 2:
    raise Exception(f"usage: python3 {argv[0]} datafile [n_sensors]")
elif not path.isfile(argv[1]):
    raise Exception(f"{argv[1]}: No such file")

# Loading mode shapes data
Psi_r = np.loadtxt(argv[1])

# Choose parameters for the sensor placement
# POD components (from ROM)
n_POD_components = Psi_r.shape[1]

# Sensors
# Number of sensors has to be larger or equal to the number of POD components
n_sensors = n_POD_components if len(argv) < 3 else int(argv[2])
if n_sensors < n_POD_components:
    n_sensors = n_POD_components

print(f"n_POD_components={n_POD_components} n_sensors={n_sensors}")

# Function for finding optimal sensor placement
def find_sensor_placement_QR(Psi_r, num_eigen, num_sensors):
    M = Psi_r @ Psi_r.T if num_sensors > num_eigen else Psi_r.T
    Q, R, P = sp.qr(M, pivoting=True)
    #C = np.zeros((num_sensors, Psi_r.shape[0]))
    #C[np.arange(num_sensors), P[:num_sensors]] = 1
    return None, P[:num_sensors]

# Finding the sensor placement matrix C
C, P = find_sensor_placement_QR(Psi_r, n_POD_components, n_sensors)
print("Sensor locations:", P)

# Save to file
filenam = argv[1].rsplit(".", 1)[0] + "_n" + str(n_sensors)
locfile = filenam + "_loc.dat"
np.savetxt(locfile,P,fmt='%i')
