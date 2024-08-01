from logic.filter import FilterXYAH
from logic.config_watcher import cfg
import numpy as np
import scipy.linalg

class Equations(FilterXYAH):  
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(3 * ndim, 3 * ndim)  # 3x the dimensions for position, velocity, acceleration
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
            self._motion_mat[i, 2*ndim + i] = 0.5 * dt**2
            self._motion_mat[ndim + i, 2*ndim + i] = dt

        self._update_mat = np.eye(ndim, 3 * ndim)  # Observation matrix

        # Adjust noise weights as needed
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self._std_weight_accel = 1. / 1000
    def initiate(self, measurement: np.ndarray) -> tuple:

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean_acc = np.zeros_like(mean_pos)  # Initialize acceleration to 0
        mean = np.r_[mean_pos, mean_vel, mean_acc] 

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            1e-2,
            1e-2,
            1e-5,
            1e-5,
            1e-3, 1e-3, 1e-3, 1e-3,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        std = [
            self._std_weight_position * mean[3], self._std_weight_position * mean[3], 1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
   
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        std_pos = [
            self._std_weight_position * mean[3], self._std_weight_position * mean[3], 1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[3], 1e-5,
            self._std_weight_velocity * mean[3]]
        std_acc = [
            self._std_weight_accel * mean[3], self._std_weight_accel * mean[3],
            1e-5, self._std_weight_accel * mean[3],
        ]  # Add std for acceleration
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_acc])) 
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
    
    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:

        std_pos = [
            self._std_weight_position * mean[:, 3], self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]), self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3], self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]), self._std_weight_velocity * mean[:, 3]]
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
        equations_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(covariance, self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, equations_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((equations_gain, projected_cov, equations_gain.T))
        return new_mean, new_covariance

    def gating_distance(self,
                        mean: np.ndarray,
                        covariance: np.ndarray,
                        measurements: np.ndarray,
                        only_position: bool = False,
                        metric: str = 'maha') -> np.ndarray:

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
