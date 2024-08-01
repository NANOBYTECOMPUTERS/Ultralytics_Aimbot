
from logic.config_watcher import cfg
import numpy as np
import scipy.linalg

class FilterHandler:

    def __init__(self):

        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        self.Q = None  # Process noise covariance
        self.R = None  # Measurement noise covariance
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
         
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
        
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    
    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        filter_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, filter_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((filter_gain, projected_cov, filter_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")



class Equations(FilterHandler):  
    def __init__(self):
        ndim, dt = 4, 1.

        # Motion Model (State Transition Matrix)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)  # Initialize identity matrix for 8x8 dimensions (position and velocity in x, y)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt  # Set velocity terms in motion matrix (position change = velocity * time)

        # Measurement Model (Observation Matrix)
        self._update_mat = np.eye(ndim, 2 * ndim)  # Maps the state (position, velocity) to the measurement (position)

        # Standard Deviations (Tuning Parameters)
        self._std_weight_position = 1. / 40  # Weight for position uncertainty
        self._std_weight_velocity = 1. / 160  # Weight for velocity uncertainty

    # Initiation
    def initiate(self, measurement: np.ndarray) -> tuple:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)  # Assume initial velocity is zero
        mean = np.r_[mean_pos, mean_vel]     # Combine position and velocity into a single state vector

        # Covariance Matrix (Initial Uncertainty)
        std = [
            2 * self._std_weight_position * measurement[2],  # Uncertainty in x position
            2 * self._std_weight_position * measurement[3],  # Uncertainty in y position
            10 * self._std_weight_velocity * measurement[2], # Uncertainty in x velocity
            10 * self._std_weight_velocity * measurement[3], # Uncertainty in y velocity
            1e-2, 1e-2, 1e-5, 1e-5,                       # Small uncertainties for other state variables 
        ]
        covariance = np.diag(np.square(std))  # Diagonal covariance matrix (assuming no correlation between variables)
        return mean, covariance

    # Projection Step (for Kalman Gain Calculation)
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        std = [
            self._std_weight_position * mean[3],  # Uncertainty in projected x position
            self._std_weight_position * mean[3],  # Uncertainty in projected y position
            1e-1, 1e-1                             # Small uncertainties for other state variables
        ]
        innovation_cov = np.diag(np.square(std))  # Covariance of the innovation (difference between predicted and measured values)

        mean = np.dot(self._update_mat, mean)      # Project the mean state to the measurement space
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))  # Project the covariance
        return mean, covariance + innovation_cov  # Add the innovation covariance 
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        # Process Noise (Tuning Parameters)
        std_pos = [
            self._std_weight_position * mean[3],    # Uncertainty in predicted x position (proportional to width)
            self._std_weight_position * mean[3],    # Uncertainty in predicted y position (proportional to height)
            1e-2, 1e-2                             # Small uncertainties for other state variables (width, height)
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],    # Uncertainty in x velocity
            self._std_weight_velocity * mean[3],    # Uncertainty in y velocity
            1e-5, 1e-5                             # Tiny uncertainties for changes in width and height
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # Covariance matrix of process noise

        # Prediction Step
        mean = np.dot(mean, self._motion_mat.T)  # Predict new mean using the motion model
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov  # Predict new covariance

        return mean, covariance
    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        # Process Noise (Tuning Parameters for Multiple Objects)
        std_pos = [
            self._std_weight_position * mean[:, 3],         # Uncertainties in x positions (proportional to widths)
            self._std_weight_position * mean[:, 3],         # Uncertainties in y positions (proportional to heights)
            1e-2 * np.ones_like(mean[:, 3]),               # Small uncertainties for widths
            self._std_weight_position * mean[:, 3]          # Uncertainties for heights
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],         # Uncertainties in x velocities
            self._std_weight_velocity * mean[:, 3],         # Uncertainties in y velocities
            1e-5 * np.ones_like(mean[:, 3]),               # Tiny uncertainties for changes in widths
            self._std_weight_velocity * mean[:, 3]          # Uncertainties for changes in heights
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T          # Square the standard deviations and transpose

        # Motion Covariance (for Each Object)
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]  # Create a diagonal covariance matrix for each object
        motion_cov = np.asarray(motion_cov)                        # Convert to NumPy array

        # Prediction Step (for Multiple Objects)
        mean = np.dot(mean, self._motion_mat.T)                   # Predict new means for all objects
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))   # Precompute a part of the covariance update
        covariance = np.dot(left, self._motion_mat.T) + motion_cov          # Predict new covariances for all objects

        return mean, covariance
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        projected_mean, projected_cov = self.project(mean, covariance)  # Project state and covariance to measurement space

        # Calculate Kalman Gain
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)  # Cholesky decomposition for numerical stability
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                            np.dot(covariance, self._update_mat.T).T,
                                            check_finite=False).T  # Compute Kalman gain using Cholesky factor

        # Innovation (Difference Between Measurement and Prediction)
        innovation = measurement - projected_mean

        # Update State and Covariance
        new_mean = mean + np.dot(innovation, kalman_gain.T)                                # Update mean using Kalman gain and innovation
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))  # Update covariance

        return new_mean, new_covariance
    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, only_position: bool = False, metric: str = 'maha') -> np.ndarray:
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError('Invalid distance metric')
        
    def multi_update(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, only_position: bool = False) -> tuple:
        projected_mean, projected_cov = self.project(mean, covariance)  # Project state and covariance to measurement space
        if only_position:
            projected_mean, projected_cov = projected_mean[:2], projected_cov[:2, :2]
            measurements = measurements[:, :2]
            mean, covariance = mean[:2], covariance[:2, :2]
        d = measurements - projected_mean
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)  # Cholesky decomposition for numerical stability
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                            np.dot(covariance, self._update_mat.T).T,
                                            check_finite=False).T  # Compute Kalman gain using Cholesky factor
        new_mean = mean + np.dot(d, kalman_gain.T)  # Update mean using Kalman gain and innovation
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))  # Update covariance

        return new_mean, new_covariance
    def multi_gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, only_position: bool = False, metric: str = 'maha') -> np.ndarray:
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:, :2], covariance[:, :2, :2]
            measurements = measurements[:, :, :2]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=2)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.transpose((0, 2, 1)), lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError('Invalid distance metric')
        