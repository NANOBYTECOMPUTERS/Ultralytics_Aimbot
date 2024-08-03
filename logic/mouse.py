#[mouse.py]
import torch
import win32con, win32api
import torch.nn as nn
import time
from logic.equations import Equations
import numpy as np
from logic.config_watcher import cfg
from logic.visual import visuals
from logic.shooting import shooting
from logic.buttons import Buttons

from logic.rzctl import RZCONTROL
import os





if cfg.arduino_move or cfg.arduino_shoot:
    from logic.arduino import arduino
#(neural network disabled until update)
#class Mouse_net(nn.Module):
 #   def __init__(self, arch):
  #      super(Mouse_net, self).__init__()
   #     self.fc1 = nn.Linear(10, 128, arch)
    #    self.fc2 = nn.Linear(128, 128, arch)
     #   self.fc3 = nn.Linear(128, 64, arch)
      #  self.fc4 = nn.Linear(64, 2, arch)

    #def forward(self, x):
     #   x = torch.relu(self.fc1(x))
      #  x = torch.relu(self.fc2(x))
       # x = torch.relu(self.fc3(x))
        #x = self.fc4(x)
        #return x


class MouseThread:
    def __init__(self):
        self.dpi = cfg.mouse_dpi
        self.mouse_sensitivity = cfg.mouse_sensitivity
        self.fov_x = cfg.mouse_fov_width
        self.fov_y = cfg.mouse_fov_height
        self.mouse_speed_factor = 1
        self.disable_prediction = cfg.disable_prediction
        self.prediction_interval = cfg.prediction_interval
        self.bScope_multiplier = cfg.bScope_multiplier
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2
        self.center_y = self.screen_height / 2
        self.mean = None
        self.covariance = None
        self.equations_filter = Equations()
        self.prev_x = 0
        self.prev_y = 0
        self.prev_time = None
        self.magnet_distance_threshold = cfg.magnet_distance_threshold
        self.magnet_pull_strength = cfg.magnet_pull_strength
        self.close_smooth_ammount = cfg.close_smooth_ammount
        self.close_smooth_distance = cfg.close_smooth_distance
        self.bScope = False
        self.magnet_pull = True
        self.adjust_x = 100
        self.adjust_y = 60
        self.arch = self.get_arch()
        
        if cfg.mouse_ghub:
            from logic.ghub import gHub
            self.ghub = gHub
        
        if cfg.razer_mouse:
            dll_name = "rzctl.dll" 
            script_directory = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_directory, dll_name)
            self.rzr = RZCONTROL(dll_path)
            if not self.rzr.init():
                print("Failed to initialize rzctl")
            
        # if cfg.AI_mouse_net:
        #    self.device = torch.device(self.arch)
         #   self.model = Mouse_net(arch=self.arch).to(self.device)
          #  try:
           #     self.model.load_state_dict(torch.load('mouse_net.pth', map_location=self.device))
        #     except Exception as e:
         #       print(e)
          #      print('Please train AI mouse model')
           #     exit()
            #self.model.eval()

    def get_arch(self):
        if cfg.AI_enable_AMD:
            return f'hip:{cfg.AI_device}'
        if 'cpu' in cfg.AI_device:
            return 'cpu'
        return f'cuda:{cfg.AI_device}'

    def process_data(self, data):
        target_x, target_y, target_w, target_h, target_cls, player_box_sizes, head_box_sizes = data

        target_center_x = target_x + (target_w / 2)
        target_center_y = target_y + (target_h / 2)

        #if cfg.AI_mouse_net == False:
         #   if (cfg.show_window and cfg.show_target_line) or (cfg.show_overlay and cfg.show_target_line):
          #      visuals.draw_target_line(target_x, target_y, target_cls)

        # Check bScope with original target coordinates
        self.bScope = self.check_target_in_scope(target_x, target_y, target_w, target_h, self.bScope_multiplier) if cfg.auto_shoot or cfg.triggerbot else False
        self.bScope = cfg.force_click or self.bScope

        if not self.disable_prediction:
            current_time = time.time()

            # Prediction and update using NumPy
            measurement = np.array([target_x, target_y, target_w, target_h])
            if self.mean is None or self.covariance is None:
                self.mean, self.covariance = self.equations_filter.initiate(measurement)
            self.mean, self.covariance = self.equations_filter.predict(self.mean, self.covariance)
            self.mean, self.covariance = self.equations_filter.update(self.mean, self.covariance, measurement)

            predicted_x, predicted_y = self.mean[:2]
        else:
            predicted_x, predicted_y = target_center_x, target_center_y # if prediction is disabled use the current target_center
        
        # Calculate movement for both original and predicted positions
        move_x_raw, move_y_raw = self.calc_movement(target_center_x, target_center_y, target_w, target_h, target_cls, player_box_sizes, head_box_sizes)  
        move_x, move_y = self.calc_movement(predicted_x, predicted_y, target_w, target_h, target_cls, player_box_sizes, head_box_sizes) 

        if (cfg.show_window and cfg.show_history_points) or (cfg.show_overlay and cfg.show_history_points):
            visuals.draw_history_point_add_point(move_x_raw, move_y_raw)  # Show original position in history
            
        if (cfg.show_window and cfg.show_target_prediction_line) or (cfg.show_overlay and cfg.show_target_prediction_line):
            visuals.draw_predicted_position(predicted_x, predicted_y, target_cls) # Show prediction if enabled

        shooting.queue.put((self.bScope, self.get_shooting_key_state()))
        self.move_mouse(move_x, move_y)  # Move based on the predicted position
    
    def calc_movement(self, target_x, target_y, target_w, target_h, target_cls, player_box_sizes, head_box_sizes): 
        #if not cfg.AI_mouse_net:
            # Target center calculation
            target_center_x = target_x + target_w / 2
            target_center_y = target_y + target_h / 2

            offset_x = target_center_x - self.center_x
            offset_y = target_center_y - self.center_y

            # Calculate distance to the target
            distance_to_target = np.sqrt(offset_x**2 + offset_y**2)
                    # Calculate target's angle and consider its size
            target_angle = np.arctan2(offset_y, offset_x) 
            target_angle_degrees = np.degrees(target_angle)

            # Adjust angle based on target size (experiment with different scaling factors)
            angle_adjustment = (target_w / self.screen_width) * 5  # Example scaling 
            target_angle_degrees += angle_adjustment if offset_x < 0 else -angle_adjustment 

            # Dynamic speed adjustment based on distance and target size
            base_speed_factor = self.mouse_speed_factor

            # Adjust speed based on distance (similar to before)
            if distance_to_target > 200:   # Far away
                speed_factor = base_speed_factor
            elif distance_to_target > 50:  # Closer
                speed_factor = base_speed_factor * (0.5 + 0.5 * (distance_to_target - 50) / 150) 
            else:                          # Very close
                speed_factor = base_speed_factor * 0.2  

            # Adjust speed based on box size (different logic for head and player)
            if target_cls == 7:  # Headshot 
                avg_head_box_area = np.mean([w * h for w, h in head_box_sizes]) if head_box_sizes else 0
                target_box_area = target_w * target_h
                box_size_factor = target_box_area / avg_head_box_area if avg_head_box_area > 0 else 1.0
            else:  # Player
                avg_player_box_area = np.mean([w * h for w, h in player_box_sizes]) if player_box_sizes else 0
                target_box_area = target_w * target_h
                box_size_factor = target_box_area / avg_player_box_area if avg_player_box_area > 0 else 1.0

            speed_factor *= box_size_factor 

            # Adjust speed based on player box size (if player detected)
            if target_cls != 7:  # Assuming class 7 is for heads
                avg_player_box_area = np.mean([w * h for w, h in player_box_sizes]) if player_box_sizes else 0
                target_box_area = target_w * target_h
                box_size_factor = target_box_area / avg_player_box_area if avg_player_box_area > 0 else 1.0
                speed_factor *= box_size_factor  # Scale speed based on box size relative to average

            # Calculate target's angle relative to the character
            target_angle = np.arctan2(offset_y, offset_x)  # In radians
            target_angle_degrees = np.degrees(target_angle)
            measurement = np.array([target_x, target_y, target_w, target_h])
            if self.mean is None or self.covariance is None:
                self.mean, self.covariance = self.equations_filter.initiate(measurement)
            self.mean, self.covariance = self.equations_filter.predict(self.mean, self.covariance)
            smoothed_measurement = self.equations_filter.update(self.mean, self.covariance, measurement)[0]
            self.mean[:2] = smoothed_measurement[:2]
            target_x, target_y = smoothed_measurement[:2]

        #if not cfg.AI_mouse_net:
            offset_x = target_x - self.center_x
            offset_y = target_y - self.center_y

            target_vel_x = self.mean[4] # velocity x from equations_filter
            target_vel_y = self.mean[5] # velocity y from equations_filter
            predicted_target_x = target_x + target_vel_x * cfg.prediction_interval
            predicted_target_y = target_y + target_vel_y * cfg.prediction_interval

            offset_x = predicted_target_x - self.center_x
            offset_y = predicted_target_y - self.center_y

            degrees_per_pixel_x = self.fov_x / self.screen_width
            degrees_per_pixel_y = self.fov_y / self.screen_height
            
            mouse_move_x = offset_x * degrees_per_pixel_x
            move_x = (mouse_move_x / 360) * (self.dpi * (1 / self.mouse_sensitivity))

            mouse_move_y = offset_y * degrees_per_pixel_y
            move_y = (mouse_move_y / 360) * (self.dpi * (1 / self.mouse_sensitivity))
        
            # Magnet Pull & Exponential Smoothing (Corrected)
            target_center_x = target_x + target_w / 2
            target_center_y = target_y + target_h / 2
            target_distance = np.linalg.norm([target_center_x - self.center_x, target_center_y - self.center_y])
            
            if self.magnet_pull:
                if target_distance < self.magnet_distance_threshold:
                    # Calculate the scaling factor for magnet pull strength
                    pull_factor = 1 - (target_distance / self.magnet_distance_threshold)
                    
                    # Apply the magnet pull to accelerate movement
                    move_x *= 1 + pull_factor * self.magnet_pull_strength
                    move_y *= 1 + pull_factor * self.magnet_pull_strength
                    
                if target_distance <= self.close_smooth_distance: #Within 2 pixel of the center
                    self.magnet_pull = False
                    smoothing_factor = self.close_smooth_ammount  # You can adjust this factor to control the smoothing strength
                    move_x = smoothing_factor * self.prev_x + (1 - smoothing_factor) * move_x
                    move_y = smoothing_factor * self.prev_y + (1 - smoothing_factor) * move_y
                    

            self.prev_x, self.prev_y = move_x, move_y  # Store for next 
            move_x = move_x * (self.adjust_x * .01)
            move_y = move_y * (self.adjust_y * .01)
            return move_x, move_y
        # Neural Network disabled until update
        #else:
        #    input_data = [
         #       self.screen_width,
          #     self.center_x,
           #     self.center_y,
            #    self.dpi,
             #   self.mouse_sensitivity,
              #  self.fov_x,
               # self.fov_y,
        #        target_x,
         #       target_y  ]
            
    #    input_data = torch.tensor([self.screen_width, self.screen_height, target_x, target_y, target_w, target_h, target_cls, player_box_sizes, head_box_sizes], 
     #                             dtype=torch.float32).to(self.device)
      #  with torch.no_grad():
       #     move_x, move_y = self.model(input_data).cpu().numpy()
        #return move_x, move_y

        
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
        self.magnet_distance_threshold = cfg.magnet_distance_threshold
        self.magnet_pull_strength = cfg.magnet_pull_strength
        self.close_smooth_ammount = cfg.close_smooth_ammount
        self.close_smooth_distance = cfg.close_smooth_distance
                
mouse = MouseThread()
#[/mouse.py]
