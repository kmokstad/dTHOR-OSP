"""
Optimal Sensor Placement (OSP) Framework
========================================

Implementation of the data-driven optimal sensor placement method based on:
K. Manohar, B. W. Brunton, J. N. Kutz, and S. L. Brunton,
"Data-Driven Sparse Sensor Placement for Reconstruction,"
IEEE Control Systems Magazine, 2017.

This module provides:
- Low-rank basis computation via SVD/POD
- D-optimal sensor placement using QR pivoting
- Conditional sensor placement with preselected sensors
- State reconstruction from sparse measurements

This code is a subset of a module provided by Adil Rasheed, Jan 2026.
Slightly modified to resolve some pylint and isort issues.
"""

from typing import List, Optional

from numpy import (arange, array, concatenate, cumsum, max, mean, ndarray,
                   searchsorted, setdiff1d, sqrt, sum, zeros)
from numpy.linalg import cond, det, norm
from scipy.linalg import lstsq, qr, svd


class OptimalSensorPlacement:
    """
    Data-driven optimal sensor placement for state reconstruction.

    Given training data, this class computes an optimal set of sensor
    locations that enable accurate reconstruction of high-dimensional
    states from sparse measurements.

    Attributes:
        Psi_r (ndarray): Reduced basis matrix (n x r)
        singular_values (ndarray): Singular values from SVD
        r (int): Number of modes retained
        n (int): State dimension
        sensor_indices (ndarray): Selected sensor locations
    """

    def __init__(self, n_modes: Optional[int] = None,
                 energy_threshold: Optional[float] = None):
        """
        Initialize the OSP framework.

        Args:
            n_modes: Number of POD modes to retain. If None, determined by energy_threshold.
            energy_threshold: Fraction of total energy to retain (e.g., 0.99 for 99%).
                              Used only if n_modes is None.
        """
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold if energy_threshold else 0.99
        self.n = None
        self.r = None
        self.Psi_r = None
        self.singular_values = None
        self.sensor_indices = None

    def fit(self, X: ndarray) -> 'OptimalSensorPlacement':
        """
        Compute the reduced basis from training data.

        Performs SVD: X = Ψ Σ V^T and truncates to r modes.

        Args:
            X: Training data matrix (n x m), where n is state dimension
               and m is number of snapshots.

        Returns:
            self: The fitted OSP object.
        """
        self.n, _ = X.shape

        # Compute SVD: X = Psi @ Sigma @ V^T
        Psi, sigma, _ = svd(X, full_matrices=False)

        # Determine number of modes to retain
        if self.n_modes is None:
            # Use energy threshold
            total_energy = sum(sigma**2)
            cumulative_energy = cumsum(sigma**2) / total_energy
            self.r = searchsorted(cumulative_energy, self.energy_threshold) + 1
        else:
            self.r = self.n_modes

        # Truncate to r modes
        self.r = min(self.r, len(sigma))
        self.Psi_r = Psi[:, :self.r]
        self.singular_values = sigma

        return self

    def select_sensors(self, n_sensors: int,
                       preselected: Optional[List[int]] = None) -> ndarray:
        """
        Select optimal sensor locations using QR pivoting.

        Implements the D-optimal criterion:
            γ* = argmax det[(C Ψ_r)^T (C Ψ_r)]

        Args:
            n_sensors: Total number of sensors to select (p).
            preselected: List of indices for preselected sensors (optional).

        Returns:
            sensor_indices: Array of selected sensor indices.
        """
        if self.Psi_r is None:
            raise RuntimeError("Must call fit() before select_sensors()")

        if preselected is not None:
            return self._conditional_sensor_selection(n_sensors, preselected)

        # Standard case: no preselected sensors
        # For p >= r: Apply QR with pivoting on Psi_r^T
        # The pivots give the sensor locations

        if n_sensors >= self.r:
            # Oversampling case: use Psi_r @ Psi_r^T for more pivots
            # But practically, we can just take more pivots from QR on Psi_r^T
            A = self.Psi_r.T  # (r x n)
        else:
            A = self.Psi_r.T  # (r x n)

        # QR factorization with column pivoting: A @ P = Q @ R
        # scipy.linalg.qr returns permutation indices directly
        _, _, pivot_indices = qr(A, pivoting=True)

        # Select first p pivots as sensor locations
        self.sensor_indices = pivot_indices[:n_sensors]

        return self.sensor_indices

    def _conditional_sensor_selection(self, n_total: int,
                                      preselected: List[int]) -> ndarray:
        """
        Algorithm 1: Conditional QR-based Optimal Sensor Placement.

        Select additional sensors when some are already fixed.

        Args:
            n_total: Total number of sensors desired (q + p).
            preselected: Indices of existing sensors (γ_0).

        Returns:
            sensor_indices: Combined array [preselected, new_sensors].
        """
        preselected = array(preselected)
        q = len(preselected)

        if q >= n_total:
            self.sensor_indices = preselected
            return self.sensor_indices

        # Step 1: Form A = Psi_r^T (r x n)
        A = self.Psi_r.T.copy()

        # Step 2: Define index sets
        available = setdiff1d(arange(self.n), preselected)

        # Step 3-4: Create permutation with preselected first
        A_permuted = A[:, concatenate([preselected, available])]

        # Step 5-6: Perform QR with pivoting, but we need to keep
        # the first q columns fixed. We do this by:
        # - Factoring out the preselected columns first
        # - Then applying QR pivoting to the remainder

        # Extract preselected and available parts
        A_fixed = A_permuted[:, :q]
        A_free = A_permuted[:, q:]

        # Orthogonalize against fixed sensors using modified Gram-Schmidt
        # Project out the component in the span of A_fixed
        if q > 0:
            Q_fixed, _ = qr(A_fixed, mode='economic')
            # Residual after projecting out fixed sensor directions
            A_residual = A_free - Q_fixed @ (Q_fixed.T @ A_free)
        else:
            A_residual = A_free

        # Apply QR pivoting to residual
        _, _, pivot_indices = qr(A_residual, pivoting=True)

        # Step 7-8: Map back to original indices
        p = n_total - q  # Number of new sensors to select
        new_sensors = available[pivot_indices[:p]]

        # Step 9: Combine preselected and new sensors
        self.sensor_indices = concatenate([preselected, new_sensors])

        return self.sensor_indices

    def get_measurement_matrix(self, sensor_indices: Optional[ndarray] = None) -> ndarray:
        """
        Construct the measurement matrix C.

        C selects rows from the identity matrix corresponding to sensor locations.

        Args:
            sensor_indices: Sensor locations. Uses stored indices if None.

        Returns:
            C: Measurement matrix (p x n).
        """
        if sensor_indices is None:
            sensor_indices = self.sensor_indices

        if sensor_indices is None:
            raise RuntimeError("No sensor indices available. Call select_sensors() first.")

        p = len(sensor_indices)
        C = zeros((p, self.n))
        for i, idx in enumerate(sensor_indices):
            C[i, idx] = 1.0

        return C

    def get_theta(self, sensor_indices: Optional[ndarray] = None) -> ndarray:
        """
        Compute Θ = C @ Ψ_r, the reduced measurement matrix.

        Args:
            sensor_indices: Sensor locations. Uses stored indices if None.

        Returns:
            Theta: Reduced measurement matrix (p x r).
        """
        if sensor_indices is None:
            sensor_indices = self.sensor_indices

        # Theta = C @ Psi_r is equivalent to selecting rows of Psi_r
        return self.Psi_r[sensor_indices, :]

    def reconstruct(self, y: ndarray,
                    sensor_indices: Optional[ndarray] = None) -> ndarray:
        """
        Reconstruct full state from sparse measurements.

        Solves: â = Θ† y, then x̂ = Ψ_r @ â

        Args:
            y: Measurements vector (p,) or matrix (p x k) for k samples.
            sensor_indices: Sensor locations. Uses stored indices if None.

        Returns:
            x_hat: Reconstructed state(s) (n,) or (n x k).
        """
        Theta = self.get_theta(sensor_indices)

        # Least squares solution: â = Θ† @ y
        if y.ndim == 1:
            a_hat, _, _, _ = lstsq(Theta, y)
            x_hat = self.Psi_r @ a_hat
        else:
            a_hat, _, _, _ = lstsq(Theta, y)
            x_hat = self.Psi_r @ a_hat

        return x_hat

    def compute_condition_number(self, sensor_indices: Optional[ndarray] = None) -> float:
        """
        Compute the condition number of Θ.

        Lower condition number indicates better sensor placement.

        Args:
            sensor_indices: Sensor locations. Uses stored indices if None.

        Returns:
            cond: Condition number κ(Θ).
        """
        return cond(self.get_theta(sensor_indices))

    def compute_d_optimal_criterion(self, sensor_indices: Optional[ndarray] = None) -> float:
        """
        Compute the D-optimal criterion: det(Θ^T @ Θ).

        Higher value indicates better sensor placement.

        Args:
            sensor_indices: Sensor locations. Uses stored indices if None.

        Returns:
            det_M: Determinant of the information matrix.
        """
        Theta = self.get_theta(sensor_indices)
        return det(Theta.T @ Theta)

    def compute_reconstruction_error(self, X_test: ndarray,
                                     sensor_indices: Optional[ndarray] = None) -> dict:
        """
        Evaluate reconstruction accuracy on test data.

        Args:
            X_test: Test data matrix (n x m_test).
            sensor_indices: Sensor locations. Uses stored indices if None.

        Returns:
            metrics: Dictionary with error statistics.
        """
        if sensor_indices is None:
            sensor_indices = self.sensor_indices

        # Get measurements
        y_test = X_test[sensor_indices, :]

        # Reconstruct
        X_reconstructed = self.reconstruct(y_test, sensor_indices)

        # Compute errors
        errors = X_test - X_reconstructed

        # Relative error per sample
        norms_true = norm(X_test, axis=0)
        norms_error = norm(errors, axis=0)
        relative_errors = norms_error / (norms_true + 1e-10)

        return {
            'mean_relative_error': mean(relative_errors),
            'max_relative_error': max(relative_errors),
            'rmse': sqrt(mean(errors**2)),
            'relative_errors': relative_errors
        }

    def explained_variance_ratio(self) -> ndarray:
        """
        Return the fraction of variance explained by each mode.

        Returns:
            ratios: Array of explained variance ratios.
        """
        if self.singular_values is None:
            raise RuntimeError("Must call fit() first.")

        total = sum(self.singular_values**2)
        return self.singular_values**2 / total
