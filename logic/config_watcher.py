import configparser
import random

class Config():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.window_name = self.get_random_window_name()
        self.Read(verbose=False)
    
    def Read(self, verbose=False, ):
        self.config.read('./config.ini')
        # Detection window
        self.config_Detection_window = self.config['Detection window']
        self.detection_window_width = int(self.config_Detection_window['detection_window_width'])
        self.detection_window_height = int(self.config_Detection_window['detection_window_height'])
        # Capture Method Bettercam
        self.config_Bettercam_Capture = self.config['Capture Methods']
        self.Bettercam_capture = self.config_Bettercam_Capture.getboolean('Bettercam_capture')
        self.bettercam_capture_fps = int(self.config_Bettercam_Capture['bettercam_capture_fps'])
        self.bettercam_monitor_id = int(self.config_Bettercam_Capture['bettercam_monitor_id'])
        self.bettercam_gpu_id = int(self.config_Bettercam_Capture['bettercam_gpu_id'])
        # Capture Method Obs
        self.config_Obs_capture = self.config['Capture Methods']
        self.Obs_capture = self.config_Obs_capture.getboolean('Obs_capture')
        self.Obs_camera_id = str(self.config_Obs_capture['Obs_camera_id'])
        self.Obs_capture_fps = int(self.config_Obs_capture['Obs_capture_fps'])
        # Aim
        self.config_Aim = self.config['Aim']
        self.body_y_offset = float(self.config_Aim['body_y_offset'])
        self.hideout_targets = self.config_Aim.getboolean('hideout_targets')
        self.disable_headshot = self.config_Aim.getboolean('disable_headshot')
        # Hotkeys
        self.config_Hotkeys_settings = self.config['Hotkeys']
        self.hotkey_targeting = str(self.config_Hotkeys_settings['hotkey_targeting'])
        self.hotkey_targeting_list = self.hotkey_targeting.split(',')
        self.hotkey_exit = str(self.config_Hotkeys_settings['hotkey_exit'])
        self.hotkey_pause = str(self.config_Hotkeys_settings['hotkey_pause'])
        self.hotkey_reload_config = str(self.config_Hotkeys_settings['hotkey_reload_config'])
        # Mouse
        self.config_Mouse = self.config['Mouse']
        self.mouse_dpi = int(self.config_Mouse['mouse_dpi'])
        self.mouse_sensitivity = float(self.config_Mouse['mouse_sensitivity'])
        self.mouse_fov_width = int(self.config_Mouse['mouse_fov_width'])
        self.mouse_fov_height = int(self.config_Mouse['mouse_fov_height'])
        self.mouse_lock_target = self.config_Mouse.getboolean('mouse_lock_target')
        self.mouse_auto_aim = self.config_Mouse.getboolean('mouse_auto_aim')
        self.mouse_ghub = self.config_Mouse.getboolean('mouse_ghub')
        
        # Shooting
        self.config_Shooting = self.config['Shooting']
        self.auto_shoot = self.config_Shooting.getboolean('auto_shoot')
        self.triggerbot = self.config_Shooting.getboolean('triggerbot')
        self.force_click = self.config_Shooting.getboolean('force_click')
        # Arduino
        self.config_Arduino = self.config['Arduino']
        self.arduino_move = self.config_Arduino.getboolean('arduino_move')
        self.arduino_shoot = self.config_Arduino.getboolean('arduino_shoot')
        self.arduino_port = str(self.config_Arduino['arduino_port'])
        self.arduino_baudrate = int(self.config_Arduino['arduino_baudrate'])
        self.arduino_16_bit_mouse = self.config_Arduino.getboolean('arduino_16_bit_mouse')
        # AI
        self.config_AI = self.config['AI']
        self.AI_model_name = str(self.config_AI['AI_model_name'])
        self.ai_model_image_size = int(self.config_AI['ai_model_image_size'])
        self.AI_conf = float(self.config_AI['AI_conf'])
        self.AI_device = str(self.config_AI['AI_device'])
        self.AI_enable_AMD = self.config_AI.getboolean('AI_enable_AMD')
        self.AI_mouse_net = self.config_AI.getboolean('AI_mouse_net')
        self.hard_lock = self.config_Mouse.getboolean('hard_lock')
        # Debug window
        self.config_Debug_window = self.config['Debug window']
        self.show_window = self.config_Debug_window.getboolean('show_window')
        self.show_detection_speed = self.config_Debug_window.getboolean('show_detection_speed')
        self.show_window_fps = self.config_Debug_window.getboolean('show_window_fps')
        self.show_boxes = self.config_Debug_window.getboolean('show_boxes')
        self.show_labels = self.config_Debug_window.getboolean('show_labels')
        self.show_conf = self.config_Debug_window.getboolean('show_conf')
        self.show_target_line = self.config_Debug_window.getboolean('show_target_line')
        self.show_target_prediction_line = self.config_Debug_window.getboolean('show_target_prediction_line')
        self.debug_window_always_on_top = self.config_Debug_window.getboolean('debug_window_always_on_top')
        self.spawn_window_pos_x = int(self.config_Debug_window['spawn_window_pos_x'])
        self.spawn_window_pos_y = int(self.config_Debug_window['spawn_window_pos_y'])
        self.debug_window_scale_percent = int(self.config_Debug_window['debug_window_scale_percent'])
        self.debug_window_name = self.window_name
        
        if verbose:
            print('Config reloaded')
            
    def get_random_window_name(self):
        try:
            with open('window_names.txt', 'r') as file:
                window_names = file.read().splitlines()
            return random.choice(window_names) if window_names else 'Calculator'
        except FileNotFoundError:
            print('window_names.txt file not found, using default window name.')
            return 'Calculator'
            
cfg = Config()