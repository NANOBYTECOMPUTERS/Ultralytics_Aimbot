import queue
import threading
import time
import math
import torch
import win32api
import win32con
import torch.nn as nn
import torch.nn.functional as F

from ctypes import *
from os import path
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from logic.buttons import Buttons
from logic.config_watcher import cfg
from logic.visual import visuals


LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)
            
class MOUSEINPUT(Structure):
    _fields_ = (('dx', LONG),
                ('dy', LONG),
                ('mouseData', DWORD),
                ('dwFlags', DWORD),
                ('time', DWORD),
                ('dwExtraInfo', ULONG_PTR))

class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT),)

class INPUT(Structure):
    _fields_ = (('type', DWORD),
                ('union', _INPUTunion))

class Mouse_net(nn.Module):
    def __init__(self, arch):  # arch
        super(Mouse_net, self).__init__()
        self.fc1 = nn.Linear(10, 256, device=arch)  # Pass arch to layers
        self.bn1 = nn.BatchNorm1d(256, device=arch)
        self.fc2 = nn.Linear(256, 256, device=arch)
        self.bn2 = nn.BatchNorm1d(256, device=arch)
        self.fc3 = nn.Linear(256, 128, device=arch)
        self.bn3 = nn.BatchNorm1d(128, device=arch)
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, 2, device=arch)

    def forward(self, x):
        x = F.silu(self.bn1(self.fc1(x)))
        x = F.silu(self.bn2(self.fc2(x)))
        x = self.dropout(F.silu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.5, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = 0.0
        self.dx_prev = 0.5
        self.t_prev = -1



    def __call__(self, x):
        t_e = 1.0 / self.freq
        def exponential_smoothing(a, x, x_prev):
         return a * x + (1 - a) * x_prev

        def smoothing_factor(t_e, cutoff):
         return 1.0 / (1.0 + 2.0 * math.pi * cutoff * t_e)
        if self.t_prev != -1:
            t_e = t_prev + t_e

        a_d = smoothing_factor(t_e, self.dcutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t_e
        return x_hat
    
class MouseThread():
    def __init__(self, filter_mode="raw"):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.prev_x = 0
        self.prev_y = 0
        self.filter_mode = filter_mode
        self.min_cutoff = 0.1
        self.beta = 0.9
        self.d_cutoff = 1.0

         # Calc max_distance as class
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2)

        # Initializes Kal filter
        if filter_mode == "kalman" or filter_mode == "kalman_one_euro":
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            # Assuming a time interval of 0.02 seconds @60fps its 0.166
            dt = 0.017

            # State transition matrix
            F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

            # Observation matrix
            H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])

            # Initial state covariance matrix
            P = np.diag([100, 100, 10, 1])

            # Process noise covariance matrix
            Q = np.diag([0.1, 0.1, 0.01, 0.01])

            # Measurement noise covariance matrix
            R = np.diag([5, 5])

            # Initialize Kalman filter
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.kf.F = F
            self.kf.H = H
            self.kf.P = P
            self.kf.Q = Q
            self.kf.R = R

        # Initialize filter objects if needed
        if filter_mode == "kalman_one_euro":
            self.x_filter = OneEuroFilter(1.0 / cfg.bettercam_capture_fps / 4, self.min_cutoff, self.beta, self.d_cutoff)
            self.y_filter = OneEuroFilter(1.0 / cfg.bettercam_capture_fps / 4 , self.min_cutoff, self.beta, self.d_cutoff)

        
  
        self.is_targeting = False
        self.button_pressed = False

        self.arch = f'cuda:{cfg.AI_device}'
        if 'cpu' in cfg.AI_device:
            self.arch = 'cpu'

        if cfg.AI_mouse_net:
            self.device = torch.device(f'{self.arch}')
            self.model = Mouse_net(arch=self.arch).to(self.device)
            try:
                self.model.load_state_dict(torch.load('mouse_net.pth', map_location=self.device))
            except Exception as e:
                print(e)
                print('need mouse_net model')
                exit()
            self.model.eval()  
            self.model.to(self.device)
    
    def process_data(self, data):
        if self.get_shooting_key_state():  # Check hotkey
            target_x, target_y, target_w, target_h = data

            self.is_targeting = self.check_target_in_scope(target_x, target_y, target_w, target_h) if cfg.auto_shoot or cfg.triggerbot else False
            self.is_targeting = True if cfg.force_click else self.is_targeting
            x, y = self.predict_target_position(target_x, target_y)

            # Filter target coordinates directly
            if self.filter_mode != "raw":
                target_x, target_y = self.get_filtered_mouse_position(target_x, target_y)

            x, y = self.calc_movement(x, y)
            self.move_mouse(x, y)
            self.shoot(self.is_targeting)
        else:
            self.button_pressed = False
        
    def get_shooting_key_state(self):
        for key_name in cfg.hotkey_targeting_list:
            key_code = Buttons.KEY_CODES.get(key_name.strip())
            if key_code is not None:
                if cfg.mouse_lock_target:
                    state = win32api.GetKeyState(key_code)
                else:
                    state = win32api.GetAsyncKeyState(key_code)
                if state < 0 or state == 1:
                    return True
        return False

    def predict_target_position(self, target_x, target_y):
        velocity_x = target_x - self.prev_x
        velocity_y = target_y - self.prev_y
        
        if velocity_x == 0 and velocity_y == 0:
            return target_x, target_y
        
        if velocity_x == 0:
            velocity_x = 1
        if velocity_y == 0:
            velocity_y = 1
        
        if velocity_x < 0:
            velocity_x = -velocity_x
        if velocity_y < 0:
            velocity_y = -velocity_y
        
        if velocity_x > velocity_y:
            velocity_x, velocity_y = velocity_y, velocity_x      
        
        if velocity_x == 0 and velocity_y == 0:
            return target_x, target_y
        
        prediction_distance = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
        
        if prediction_distance == 0:
            return target_x, target_y
    
        predicted_x = target_x + velocity_x + (velocity_x / abs(velocity_x)) * prediction_distance if velocity_x != 0 else target_x 
        predicted_y = target_y + velocity_y + (velocity_y / abs(velocity_y)) * prediction_distance if velocity_y != 0 else target_y

        self.prev_x = target_x
        self.prev_y = target_y

        return predicted_x, predicted_y
    
    def calc_movement(self, target_x, target_y):
        """Calculates the required mouse movement to aim at the target."""
        if cfg.AI_mouse_net == False:
            offset_x = target_x - self.center_x
            offset_y = target_y - self.center_y

            # Calculate overall target distance from the center
            target_distance = math.sqrt(offset_x ** 2 + offset_y ** 2)

            # Calculate target distance in x and y components
            target_distance_x = abs(offset_x)
            target_distance_y = abs(offset_y)

            # Degrees per pixel by FOV and resolution
            degrees_per_pixel_x = self.fov_x / self.screen_width
            degrees_per_pixel_y = self.fov_y / self.screen_height

            # Calculate raw mouse move in degrees
            mouse_move_x = offset_x * degrees_per_pixel_x
            mouse_move_y = offset_y * degrees_per_pixel_y

            # Convert degrees to mouse movement in pixels
            move_x = (mouse_move_x / 360) * (self.dpi * (1 / self.mouse_sensitivity))
            move_y = (mouse_move_y / 360) * (self.dpi * (1 / self.mouse_sensitivity))

            # Acceleration based on target distance (example)
            accel_x = 2.0 if target_distance_x < 20 else 5.0
            accel_y = 1.0 if target_distance_y < 20 else 0.5
            move_x *= accel_x
            move_y *= accel_y

            # Dynamic sensitivity scaling
            scaling_factor = 1.2 - 0.7 * (target_distance / self.max_distance)
            move_x *= scaling_factor
            move_y *= scaling_factor
        
        else:  # If AI_mouse_net is enabled
            input_data = [self.screen_width, self.screen_height, self.center_x, self.center_y,
                          self.dpi, self.mouse_sensitivity, self.fov_x, self.fov_y,
                          target_x, target_y]

            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                move = self.model(input_tensor).cpu().numpy()

            move_x, move_y = move[0][0], move[0][1]

        # Clamp movement values to prevent overflow
        max_movement = 300
        move_x = max(-max_movement, min(max_movement, move_x))
        move_y = max(-max_movement, min(max_movement, move_y))
 
        # Visualize the predicted position (optional)
        if cfg.show_window and cfg.show_target_prediction_line:
            visuals.draw_predicted_position(target_x, target_y)

        return move_x, move_y #moved outside of if statement
    
    def get_filtered_mouse_position(self, x_raw, y_raw):
        if self.filter_mode == "raw":
            return x_raw, y_raw
        elif self.filter_mode == "kalman":
            # Kalman filter prediction and update
            self.kf.predict()
            self.kf.update(np.array([x_raw, y_raw]).reshape(-1, 1))
            return self.kf.x[:2]  # Predicted position
        elif self.filter_mode == "kalman_one_euro":
            # Kalman filter prediction and update
            self.kf.predict()
            self.kf.update(np.array([x_raw, y_raw]).reshape(-1, 1))
            x_hat, y_hat = self.kf.x[:2]  # Predicted position
            dx, dy = self.kf.x[2:]  # Predicted velocity

            # Apply OneEuroFilter after Kalman Filter
            x_hat = self.x_filter(x_hat.item())
            y_hat = self.y_filter(y_hat.item())

            return x_hat, y_hat
        else:
            raise ValueError("Invalid filter mode")
               
    def move_mouse(self, x, y):
        if x is None or y is None:
            return

        x_filtered, y_filtered = self.get_filtered_mouse_position(x, y)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x_filtered), int(y_filtered), 0, 0)  # I am only using win32api

    def shoot(self, is_targeting):
        """Handles mouse clicks for shooting using win32api."""
        if cfg.auto_shoot:
            should_shoot = (cfg.triggerbot and is_targeting) or (
                not cfg.triggerbot and (self.get_shooting_key_state() or cfg.mouse_auto_aim) and is_targeting
            )

            if should_shoot and not self.button_pressed:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)  # I am only using win32api
                self.button_pressed = True
            elif not should_shoot and self.button_pressed:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)  # I am only using win32api
                self.button_pressed = False
                
    def check_target_in_scope(self, target_x, target_y, target_w, target_h):
        x1 = (target_x - target_w)
        x2 = (target_x + target_w)
        y1 = (target_y - target_h)
        y2 = (target_y + target_h)

        if (self.center_x > x1 and self.center_x < x2 and self.center_y > y1 and self.center_y < y2) :
            return True
        else:
            return False

    def Update_settings(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
                
mouse = MouseThread()
