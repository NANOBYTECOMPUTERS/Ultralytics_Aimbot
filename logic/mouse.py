import torch
import win32con, win32api
import torch.nn as nn
import time
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH as KalmanFilterXYAH
import numpy as np
import scipy.linalg
from logic.config_watcher import cfg
from logic.visual import visuals
from logic.shooting import shooting
from logic.buttons import Buttons

if cfg.arduino_move or cfg.arduino_shoot:
    from logic.arduino import arduino

class Mouse_net(nn.Module):
    def __init__(self, arch):
        super(Mouse_net, self).__init__()
        self.fc1 = nn.Linear(10, 128, arch)
        self.fc2 = nn.Linear(128, 128, arch)
        self.fc3 = nn.Linear(128, 64, arch)
        self.fc4 = nn.Linear(64, 2, arch)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class KalmanFilterXYWH(KalmanFilterXYAH):  # Create a subclass
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray) -> tuple:

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            1e-2,
            1e-2,
            1e-5,
            1e-5,
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
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

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
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(covariance, self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
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

class MouseThread:
    def __init__(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.mean = None
        self.covariance = None
        self.kalman_filter = KalmanFilterXYWH()
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        
        self.bScope = False
        
        self.arch = self.get_arch()
        
        if cfg.mouse_ghub:
            from logic.ghub import gHub
            self.ghub = gHub
            
        if cfg.AI_mouse_net:
            self.device = torch.device(self.arch)
            self.model = Mouse_net(arch=self.arch).to(self.device)
            try:
                self.model.load_state_dict(torch.load('mouse_net.pth', map_location=self.device))
            except Exception as e:
                print(e)
                print('Please train AI mouse model, or download latest trained mouse_net.pth model from repository and place in base folder. Instruction here: https://github.com/SunOner/mouse_net')
                exit()
            self.model.eval()

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        if 'cpu' in cfg.AI_device:
            return 'cpu'
        return f'cuda:{cfg.AI_device}'

    def process_data(self, data):
        target_x, target_y, target_w, target_h, target_cls = data
        target_center_x = target_x + (target_w / 2)
        target_center_y = target_y + (target_h / 2)

        if cfg.AI_mouse_net == False:
            if (cfg.show_window and cfg.show_target_line) or (cfg.show_overlay and cfg.show_target_line):
                visuals.draw_target_line(target_x, target_y, target_cls)

        # Check bScope with original target coordinates
        self.bScope = self.check_target_in_scope(target_x, target_y, target_w, target_h, self.bScope_multiplier) if cfg.auto_shoot or cfg.triggerbot else False
        self.bScope = cfg.force_click or self.bScope

        if not self.disable_prediction:
            current_time = time.time()

            # Prediction and update using NumPy
            measurement = np.array([target_x, target_y, target_w, target_h])
            if self.mean is None or self.covariance is None:
                self.mean, self.covariance = self.kalman_filter.initiate(measurement)
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, measurement)

            predicted_x, predicted_y = self.mean[:2]
        else:
            predicted_x, predicted_y = target_center_x, target_center_y # if prediction is disabled use the current target_center
        
        # Calculate movement for both original and predicted positions
        move_x_raw, move_y_raw = self.calc_movement(target_center_x, target_center_y, target_w, target_h, target_cls)  
        move_x, move_y = self.calc_movement(predicted_x, predicted_y, target_w, target_h, target_cls) 

        if (cfg.show_window and cfg.show_history_points) or (cfg.show_overlay and cfg.show_history_points):
            visuals.draw_history_point_add_point(move_x_raw, move_y_raw)  # Show original position in history
            
        if (cfg.show_window and cfg.show_target_prediction_line) or (cfg.show_overlay and cfg.show_target_prediction_line):
            visuals.draw_predicted_position(predicted_x, predicted_y, target_cls) # Show prediction if enabled

        shooting.queue.put((self.bScope, self.get_shooting_key_state()))
        self.move_mouse(move_x, move_y)  # Move based on the predicted position
    
    def calc_movement(self, target_x, target_y, target_w, target_h, target_cls):
        measurement = np.array([target_x, target_y, target_w, target_h])
        if self.mean is None or self.covariance is None:
            self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        smoothed_measurement = self.kalman_filter.update(self.mean, self.covariance, measurement)[0]
        self.mean[:2] = smoothed_measurement[:2]

        target_x, target_y = smoothed_measurement[:2]  

        if not cfg.AI_mouse_net:
            offset_x = target_x - self.center_x
            offset_y = target_y - self.center_y

            # Calculate target velocity based on Kalman filter
            target_vel_x = self.mean[4] # velocity x from kalman_filter
            target_vel_y = self.mean[5] # velocity y from kalman_filter

            # Predict future position of the target
            predicted_target_x = target_x + target_vel_x * cfg.prediction_interval  # cfg.prediction_interval is in seconds
            predicted_target_y = target_y + target_vel_y * cfg.prediction_interval

            # Adjust offset to lead the target
            offset_x = predicted_target_x - self.center_x
            offset_y = predicted_target_y - self.center_y


            degrees_per_pixel_x = self.fov_x / self.screen_width
            degrees_per_pixel_y = self.fov_y / self.screen_height

            mouse_move_x = offset_x * degrees_per_pixel_x
            move_x = (mouse_move_x / 360) * (self.dpi * (1 / self.mouse_sensitivity))

            mouse_move_y = offset_y * degrees_per_pixel_y
            move_y = (mouse_move_y / 360) * (self.dpi * (1 / self.mouse_sensitivity))

            return move_x, move_y
        else:
            input_data = [
                self.screen_width,
                self.screen_height,
                self.center_x,
                self.center_y,
                self.dpi,
                self.mouse_sensitivity,
                self.fov_x,
                self.fov_y,
                target_x,
                target_y
            ]
            
        input_data = torch.tensor([self.screen_width, self.screen_height, target_x, target_y, target_w, target_h], 
                                  dtype=torch.float32).to(self.device)
        with torch.no_grad():
            move_x, move_y = self.model(input_data).cpu().numpy()
        return move_x, move_y

        
    def move_mouse(self, x, y):
        if x is not None and y is not None:  # Combined None checks
            is_shooting = self.get_shooting_key_state()
            if (is_shooting and not cfg.mouse_auto_aim and not cfg.triggerbot) or cfg.mouse_auto_aim:
                x, y = int(x), int(y)  # Convert to integers once

                if not cfg.mouse_ghub and not cfg.arduino_move:  # Native
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
                elif cfg.mouse_ghub:  # GHub
                    self.ghub.mouse_xy(x, y)
                elif cfg.arduino_move:  # Arduino
                    arduino.move(x, y)
    
    def get_shooting_key_state(self):
        for key_name in cfg.hotkey_targeting_list:
            key_code = Buttons.KEY_CODES.get(key_name.strip())
            if key_code is not None:
                state = win32api.GetKeyState(key_code) if cfg.mouse_lock_target else win32api.GetAsyncKeyState(key_code)
                if state < 0 or state == 1:
                    return True
        return False
      
    def check_target_in_scope(self, target_x, target_y, target_w, target_h, reduction_factor):
        # Optimized calculations using NumPy:
        center = np.array([self.center_x, self.center_y])
        target_center = np.array([target_x + target_w / 2, target_y + target_h / 2])
        target_size = np.array([target_w * reduction_factor / 2, target_h * reduction_factor / 2])

        self.bScope = np.all(np.abs(target_center - center) < target_size)


    def update_settings(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
                
mouse = MouseThread()