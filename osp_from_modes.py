"""
Calculation of optimal sensor placements from a set of modes from ROM analysis.
"""

from os import path
from sys import argv

import numpy as np
import scipy.linalg as sp

if len(argv) < 2:
    raise Exception(f"usage: python {argv[0]} datafile [n_sensors]")
if not path.isfile(argv[1]):
    raise Exception(f"{argv[1]}: No such file")

# Loading mode shapes data
Psi_r = np.loadtxt(argv[1])

# Choose parameters for the sensor placement
# POD components (from ROM)
n_POD_components = Psi_r.shape[1]

# Sensors
# Number of sensors has to be larger or equal to the number of POD components
n_sensors = n_POD_components if len(argv) < 3 else max(int(argv[2]),n_POD_components)

print(f"n_POD_components={n_POD_components} n_sensors={n_sensors}")

def find_sensor_placement_QR(num_eigen, num_sensors):
    """
    Function for finding optimal sensor placement.
    Note: Using Psi_r as global variable.
    """
    M = Psi_r @ Psi_r.T if num_sensors > num_eigen else Psi_r.T
    _, _, P = sp.qr(M, pivoting=True)
    #C = np.zeros((num_sensors, Psi_r.shape[0]))
    #C[np.arange(num_sensors), P[:num_sensors]] = 1
    return None, P[:num_sensors]

# Finding the sensor placement matrix C
_, locs = find_sensor_placement_QR(n_POD_components, n_sensors)
print("Sensor locations:", locs)

# Save to file
filenam = argv[1].rsplit(".", 1)[0] + "_n" + str(n_sensors)
locfile = filenam + "_loc.dat"
np.savetxt(locfile,locs,fmt="%i")
