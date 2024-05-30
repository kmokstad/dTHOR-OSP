# Optimal sensor placement (OSP)

This python module can be used for calculating the optimal sensor placement
based on sampled FE solution data. It uses the `numpy` methods `linalg.svd()`
and `linalg.pinv()` as well as the `scipy` method `linalg.qr()` for doing the
main core of the computations.

Usage:

    python3 sensor_placement.py <datafile> <num_POD> <num_sensor> [<nX>|<pixelfile> [<image_idx>]]
    python3 osp_noplot.py <datafile> <num_POD> <num_sensor>

The first script `sensor_placement.py` works for 2D problems only where the
values in `datafile` are samples in a Cartesian grid. `nX` is then the number
of data values in horizontal direction in each time frame, and the number
of values in vertical direction `nY` is then derived from the total number
of valus per time frame divided by `nX`. Alternatively, the `pixelfile` can be
specified which gives the index into a compressed array for each time frame,
or zero if the pixel is outside the rectangular domain. The `datafile` is
assumed to contain one time frame per line. This script will also
plot the results and create animations using the `matplotlib` module.

The second script `osp_noplot.py` can be used for any data set - also 3D,
and contains no plotting. It only computes the optimal locations of the
specified number of sensors, the reconstructed data field based on the sensor
values, as well as the error (difference between the original and reconstructed
data fields). All these calculations are also performed by the first script.
