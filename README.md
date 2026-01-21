# Optimal sensor placement (OSP)

This python module can be used for calculating the optimal sensor placement
based on sampled FE solution data. It uses the [numpy](https://numpy.org/) methods
[linalg.svd()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) and
[linalg.pinv()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html)
as well as the [scipy](https://scipy.org/) method
[linalg.qr()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html)
for doing the main core of the computations.

Usage:

    python3 sensor_placement.py <datafile> <num_POD> <num_sensor> [<nX>|<pixelfile> [<image_idx>]]
    python3 osp_noplot.py <datafile> <num_POD> <num_sensor> [<num_skip> [<sensor_list>]]

The first script [sensor_placement.py](sensor_placement.py) works for 2D problems only,
where the values in `<datafile>` are samples in a Cartesian grid.
The parameter `<nX>` is then the number of data values in horizontal direction in each time frame,
and the number of values in vertical direction (`nY`) is derived from the total number
of values per time frame divided by `<nX>`. Alternatively, the `<pixelfile>` can be
specified which gives the index into a compressed array for each time frame,
or zero if the pixel is outside the rectangular domain. The `<datafile>` is
assumed to contain one time frame per line. This script will also
plot the results and create animations using the [matplotlib](https://matplotlib.org/) module.

The second script [osp_noplot.py](osp_noplot.py) can be used for any data set - also 3D,
and contains no plotting. It only computes the optimal locations of the
specified number of sensors, the reconstructed data field based on the sensor
values, as well as the error (difference between the original and reconstructed
data fields). All these calculations are also performed by the first script.

The optional `<num_skip>` parameter can be used to skip the given number of
initial frames from the `<datafile>`, to ignore initial transients in the simulation results.
`<sensor_list>` is an optional file containing a list if predefined sensor indices.
If specified, a constrained OSP is performed where the sepcified number of sensors comes
in addition to the predefined ones.

The results from the second script are saved to ASCII-files for visualization by external programs.
For instance, if the `<datafile>` here contains the time history of a nodal result quantity,
the result files will have identical topology and can be loaded as nodal result fields
into the program that created the `<datafile>` for further processing and visualization.
