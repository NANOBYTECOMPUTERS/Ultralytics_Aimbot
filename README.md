<div align="center">

# Yolov10_Aimbot was developed from Sunones Project
Due to the large differences in features and functions as well as mine consistantly changing it is branching to its own repository soon

[![Python Version](https://img.shields.io/badge/Python-3.11.6-FFD43B?logo=python)]

[![Sunones Discord server](https://badgen.net/discord/online-members/sunone)](https://discord.gg/sunone)
  <p>
    <a href="https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/releases" target="_blank">
      <img width="75%" src="https://raw.githubusercontent.com/SunOner/sunone_aimbot/main/media/demo.gif"></a>
  </p>
</div>
If you are feeling generous buy me a cofee, spent alot of hours working on stuff.
PayPal:  https://www.paypal.com/donate/?business=TZFWU8AQRYZY6&no_recurring=0&currency_code=USD # Replace with a single Buy Me a Coffee username
CashApp: $DonArrington 
Bitcoin: 3Cysfhcyh7Jejc1bxJN3sVo9unkrnZPrYE
Dogecoin: DGQPgt7n7DgjHygwa67XPExi4wyX7zZTC4
Ethereum: 0x2c8644b5731eF3021b5de7EbE5Df3ec936C11Dc7


## Overview
an AI-powered aim bot for first-person shooter games. It leverages the YOLOv8 and YOLOv10 models, PyTorch, and various other tools to automatically target and aim at enemies within the game. The AI model in repository has been trained on more than 30,000 images from popular first-person shooter games like Warface, Destiny 2, Battlefield (all series), Fortnite, The Finals, CS2 and more.
> [!WARNING]
> Use it at your own risk, we do not guarantee that you may be blocked!

> [!NOTE] 
> This application only works on Nvidia graphics cards. AMD support is testing. See [AI_enable_AMD](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT?tab=readme-ov-file#ai) option.
> The recommended graphics card for starting and more productive and stable operation starts with the rtx 20 series.

## Requirements
Before you get started, make sure you have the following prerequisites installed and pay attention to the versions in [Tested Environment](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT?tab=readme-ov-file#tested-environment) block, this may cause errors in launching the aimbot.

- Information in English:
  - [Auto-Install guide](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/docs/en/helper_en.md)
  - [Self install guide](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/docs/en/install_guide_en.md)
  - [Questions and answers](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/docs/en/questions_en.md)
  - [Arduino setup](https://github.com/SunOner/HID_Arduino)
  - [Discord server](https://discord.gg/sunone)
<br></br>
- Информация на Русском языке:
  - [Инструкция по автоматической установке](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/docs/ru/helper_ru.md)
  - [Инструкция по установке в ручную](https://github.comNANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/docs/ru/install_guide_ru.md)
  - [Вопросы и ответы](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/docs/ru/questions_ru.md)
  - [Установка ардуино](https://github.com/SunOner/HID_Arduino)
  - [Discord сервер](https://discord.gg/sunone)
<br></br>
- To launch the aimbot after all installations, type `py run.py`.

## Tested Environment
### The Sunone Aimbot has been tested on the following environment:
<table>
  <thead><tr><th>Windows</th><td>10 and 11(priority)</td></thead>
  <thead><tr><th>Python:</th><td>3.11.6</td></tr></thead>
  <thead><tr><th>CUDA:</th><td>12.4</td></tr></thead>
  <thead><tr><th>TensorRT:</th><td>10.0.1</td></tr></thead>
  <thead><tr><th>Ultralytics:</th><td>8.2.64</td></tr></thead>
  <thead><tr><th>GitHub AI Model:</th><td>0.4.1 (YOLOv8)</td></tr></thead>
  <thead><tr><th>Boosty AI Model:</th><td>0.6.0 (YOLOv10)</td></tr></thead>
</table>

## Options
The behavior of the aimbot can be configured via the [`config.ini`](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/config.ini) file. Here are the available options:

### Object Search window resolution:
- detection_window_width `int`: Horizontal resolution of the object search window.
- detection_window_height `int`: Vertical resolution of the object search window.

### Bettercam capture method:
- Bettercam_capture `bool`: Use [Bettercam](https://github.com/RootKit-Org/BetterCam) to capture images from the screen.
- bettercam_capture_fps `int`: Specific fps value for screen capture.
- bettercam_monitor_id `int`: Id of the monitor from which the images will be captured.
- bettercam_gpu_id `int`: Id of the GPU to be used for image capture

### Obs capture method:
- Obs_capture `bool`: Use [Obs](https://github.com/obsproject/obs-studio) to capture images from the screen.
- Obs_camera_id `str` or `int`: `auto` or number of Virtual Camera ID.
- Obs_capture_fps `int`: Specific fps value for screen capture.

### Aim:
- body_y_offset `float`: Allows correction of y coordinates inside the body detected box if head is not detected.
- hideout_targets `bool`: Allows shooting at targets on the range (for example in warface on the polygon or in aimlabs).
- disable_headshot `bool`: Disable head targerting.
- disable_prediction `bool`: Disable target position prediction.
- prediction_interval `bool`: The time frame used to estimate an object's future position based on its current motion.
- third_person `bool`: Turn on the third-person game mode. (sunxds_0.5.7+)

### Hot keys:
- The names of all the keys are [here](https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/logic/buttons.py). Type `None` is empty button.
- hotkey_targeting `str`: Aiming at the target. Supports multi keys, for example `hotkey_targeting = RightMouseButton,X2MouseButton`
- hotkey_exit `str`: Exit.
- hotkey_pause `str`: Pause AIM.
- hotkey_reload_config `str`: Reload config.

### Mouse:
- mouse_dpi `int`: Mouse DPI.
- mouse_sensitivity  `float`: Aim sensitivity.
- mouse_fov_width  `int`: The current horizontal value of the viewing angle in the game.
- mouse_fov_height  `int`: The current vertical value of the viewing angle in the game.
- mouse_clamp_movment `bool` True: Activate feature to limit the distance per movement by the bot according to your settings. False: no limitations per movement by the bot are applied.
- mouse_max_movement_x `float`: While clamping is true this setting applies how far the mouse can move left and right per movement by the bot.
- mouse_max_movement_y `float`: While clamping is true this setting applies how far the mouse can move up and down per movement by the bot.
- mouse_on_target_slow_x `float` Maximum movment speed left and right while in the inner 20% of detection window. 10.0 is 100% (full speed) 5.1 is 51% of your speed (adjust to your needs)
- mouse_on_target_slow_y `float` Maximum movment speed while in the inner 20% of detection window. 10.0 is 100% (full speed) 0.5 is 5% of your speed (adjust to your needs 10.0 to 0.1)
- mouse_lock_target `bool`: True: Press once to permanently aim at the target, press again to turn off the aiming. False: Hold down the button to constantly aim at the target.
- mouse_auto_aim `bool`: Automatic targeting.
- mouse_ghub `bool`: Uses Logitech GHUB exploit for mouse movement. If the value is False, native win32 library is used for movement.

### Shooting:
- auto_shoot `bool`: Automatic shooting. (For some games need [arduino](https://github.com/SunOner/HID_Arduino)).
- triggerbot `bool`: Automatic shooting at a target if it is in the scope, requires the `mouse_auto_shoot` option enabled, and aiming will also be automatically turned off.
- force_click `bool`: Shooting will be performed even if the sight is not located within the object.
- bScope_multiplier `float`: The multiplier of the target trigger size.

### Arduino:
- arduino_move `bool`: Sends a command to the arduino to move the mouse.
- arduino_shoot `bool`: Sends a command to the arduino to fire with the mouse.
- arduino_port `str`: Arduino COM port. Use `COM1` or `COM2` ... or `auto`.
- arduino_baudrate `int`: Custom Arduino baudrate.
- arduino_16_bit_mouse `bool`: Send 16 bit data to the arduino port to move the mouse.

### AI:
- AI_model_name `str`: AI model name.
- AI_model_image_size `int`: AI model image size.
- AI_conf `float`: How many percent is AI sure that this is the right goal.
- AI_device `int` or `str`: Device to run on, `0`, `1`... or `cpu`.
- AI_enable_AMD `bool`: Enable support Amd GPUs. Install ROCm, [Zluda](https://github.com/vosen/ZLUDA) and PyTorch. See [AMD docs](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html).
- AI_mouse_net `bool`: Use a neural network to calculate mouse movements. See [this repository](https://github.com/SunOner/mouse_net).

### Overlay:
- show_overlay `bool`: Enables the overlay. It is not recommended for gameplay, only for debugging.
- overlay_show_borders `bool`: Displaying the borders of the overlay.
- overlay_show_boxes `bool`: Display of boxes.
- overlay_show_target_line `bool`: Displaying the line to the target.
- overlay_show_target_prediction_line `bool`: Displaying the predictive line to the target.
- overlay_show_labels `bool`: Displaying label names.
- overlay_show_conf `bool`: Displaying the label names as well as the confidence level.

### Debug window:
- show_window `bool`: Shows the OpenCV2 window for visual feedback.
- show_detection_speed `bool`: Displays speed information inside the debug window.
- show_window_fps `bool`: Displays FPS in the corner.
- show_boxes `bool`: Displays detectable objects.
- show_labels `bool`: Displays the name of the detected object.
- show_conf `bool`: Displays object confidence threshold for detection.
- show_target_line `bool`: Shows the mouse finishing line.
- show_target_prediction_line `bool`: Show mouse prediction line.
- show_bScope_box  `bool`: Show the trigger box for auto shooting.
- show_history_points `bool`: Show history points.
- debug_window_always_on_top `bool`: The debug window will always be on top of other windows.
- spawn_window_pos_x `int`: When the debugging window starts, it takes the x position.
- spawn_window_pos_y `int`: When the debugging window starts, it takes the y position.
- debug_window_scale_percent `int`: Adjusts the size of the debug window.
- The names of the debugging window can be written in the file window_names.txt they will be randomly selected.

## AI Models
- *.pt: Default AI model.
- *.onnx: The model is optimized to run on processors.
- *.engine: Final exported model, which is faster than the previous two.

## Export .pt model to .engine
1. All commands are executed in the console window:
2. First, go to the aimbot directory using the command:
```cmd
cd C:\Users\your_username\downloads\sunone_aimbot-main
```
3. Then export the model from the .pt format in .engine format.
```cmd
yolo export model="models/sunxds_6.pt" format=engine device=0 imgsz=640 half=True
```
  - `model="model_path/model_name.pt"`: Path to model.
  - `format=engine`: TensorRT model format.
  - `half=true`: Use Half-precision floating-point format.
  - `device=0`: GPU id.
  - `workspace=8`: GPU max video memory.
  - `verbose=False`: Debug stuff. Convenient function, can show errors when exporting.

4. Each model has its own image size with which it was trained, export only with the image size with which it was published.

## Notes / Recommendations
- Limit the maximum value of frames per second in the game in which you will use it. And also do not set the screen resolution to high. Do not overload the graphics card.
- Do not set high graphics settings in games.
- Limit the browser (try not to watch YouTube while playing and working AI at the same time, for example (of course if you don't have a super duper graphics card)) and so on, which loads the video card.
- Try to use TensorRT for acceleration. `.pt` model is good, but does not have as much speed as `.engine`.
- Turn off the cv2 debug window, this saves system resources.
- Do not increase the object search window resolution, this may affect your search speed.
- If you have started the application and nothing happens, it may be working, close it with the F2 key and change the `show_window` option to `True` in the file [config.ini](https://github.com/https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/logic/buttons.py/blob/main/config.ini) to make sure that the application is working.

## Support the project
sunones models are great [here](https://boosty.to/sunone).

## License
This project is licensed under the MIT License. See **[LICENSE](https://github.com/https://github.com/NANOBYTECOMPUTERS/YOLOV10_AIMBOT/blob/main/logic/buttons.py/blob/main/LICENSE)** for details
