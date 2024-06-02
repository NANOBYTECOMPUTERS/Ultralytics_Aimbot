<div align="center">

# YOLOv8 Aimbot
[![Python Version](https://img.shields.io/badge/Python-3.11.6-FFD43B?logo=python)]
[![License MIT]

</div>

## Overview
 AI-powered aim bot for first-person shooter games.
 
 Use Train_AI_Mouse.py to train your mousenet.pth file scroll close to the bottom to change your settings to match in game settings before training
 they will be below
 if __name__ == "__main__":
    ###################### Options ######################

 
 I cannot stress enough you should be familier with how python works for using this as there are currently many settings within the hard code you will need to tinker with to make this work well for you.
 this is still in development
 

> [!WARNING]
> Use it at your own risk, I cannot guarentee you wont get banned

> [!NOTE] 
> This application only works on Nvidia graphics cards. AMD support is in the works *sunones project
> The recommended graphics card for starting and more productive and stable operation starts with the rtx 20 series.

## Requirements
Before you get started, make sure you have the following prerequisites installed and pay attention





<br></br>
- To launch the aimbot after all installations, type `py run.py` or `py launcher.py` to open gui settings.

## Tested Environment
### The YOLOv8 Aimbot has been tested on the following environment:
<table>
  <thead><tr><th>Windows</th><td>11</td></thead>
  <thead><tr><th>Python:</th><td>3.11.6</td></tr></thead>
  <thead><tr><th>CUDA:</th><td>12.4</td></tr></thead>
  <thead><tr><th>TensorRT:</th><td>10.0</td></tr></thead>
  <thead><tr><th>Ultralytics:</th><td>8.2</td></tr></thead>
  <thead><tr><th>AI Model:</th><td>model1</td></tr></thead>
</table>

## Options

### within mouse.py
you should be familiar with how python operated to change these settings 
E.G # Acceleration based on target distance (example)
        accel_x = 5.0 if target_distance_x > 100 else 6.0
        accel_y = 5.0 if target_distance_y > 100 else 6.0
        move_x *= accel_x
        move_y *= accel_y
        this changes the acceleration speed based on distance, for instance it is 5.0 unless it is within 100 pixels than changes to 6.0 (this may be way too fast for you)
        
        there are several things like scaling alpha smoothing factors etc which may be modified. this is a work in progress,


The behavior of the aim bot can be configured via the [`config.ini`](https://github.com/SunOner/yolov8_aimbot/blob/main/config.ini) file and furteher within /logic/mouse.py. Here are the available options:

### Object Search window resolution:
- detection_window_width `int`: Horizontal resolution of the object search window.
- detection_window_height `int`: Vertical resolution of the object search window.

### Bettercam capture method:
- Bettercam_capture `bool`: Use [Bettercam](https://github.com/RootKit-Org/BetterCam) to capture images from the screen.
- bettercam_capture_fps `int`: Specific fps value for screen capture.
- bettercam_monitor_id `int`: Id of the monitor from which the images will be captured.
- bettercam_gpu_id `int`: Id of the GPU to be used for image capture


### Aim:
- body_y_offset `float`: Allows correction of y coordinates inside the body detected box if head is not detected.
- hideout_targets `bool`: Allows shooting at targets on the range (for example in warface on the polygon or in aimlabs).
- disable_headshot `bool`: Disable head targerting.


### Mouse:
- mouse_dpi `int`: Mouse DPI.
- mouse_sensitivity  `float`: Aim sensitivity.
- mouse_fov_width  `int`: The current horizontal value of the viewing angle in the game.
- mouse_fov_height  `int`: The current vertical value of the viewing angle in the game.
- mouse_lock_target `bool`: True: Press once to permanently aim at the target, press again to turn off the aiming. False: Hold down the button to constantly aim at the target.
- mouse_auto_aim `bool`: Automatic targeting.


### Shooting:
- auto_shoot `bool`: Automatic shooting. 
- triggerbot `bool`: Automatic shooting at a target if it is in the scope, requires the `mouse_auto_shoot` option enabled, and aiming will also be automatically turned off.
- force_click `bool`: Shooting will be performed even if the sight is not located within the object.

### AI:
- AI_model_name `str`: AI model name.
- AI_model_image_size `int`: AI model image size.
- AI_conf `float`: How many percent is AI sure that this is the right goal.
- AI_device `int` or `str`: Device to run on, `0`, `1`... or `cpu`.
- AI_enable_AMD `bool`: Enable support Amd GPUs. Install ROCm, [Zluda](https://github.com/vosen/ZLUDA) and PyTorch. See [AMD docs](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html).
- AI_mouse_net `bool`: Use trained .pth file to calculate mouse movements.

### Debug window:
- show_window `bool`: Shows the OpenCV2 window for visual feedback.
- show_detection_speed `bool`: Displays speed information inside the debug window.
- show_window_fps `bool`: Displays FPS in the corner.
- show_boxes `bool`: Displays detectable objects.
- show_labels `bool`: Displays the name of the detected object.
- show_conf `bool`: Displays object confidence threshold for detection.
- show_target_line `bool`: Shows the mouse finishing line.
- show_target_prediction_line `bool`: Show mouse prediction line.
- debug_window_always_on_top `bool`: The debug window will always be on top of other windows.
- spawn_window_pos_x `int`: When the debugging window starts, it takes the x position.
- spawn_window_pos_y `int`: When the debugging window starts, it takes the y position.
- debug_window_scale_percent `int`: Adjusts the size of the debug window.
- The names of the debugging window can be written in the file window_names.txt they will be randomly selected.

## AI Models
- *.pt: Default AI model.
- *.onnx: The model is optimized to run on processors.
- *.engine: Final exported model, which is faster than the previous two.


## Export .pt model to .engine (
1. All commands are executed in the console window:
2. First, go to the aimbot directory using the command:
```cmd
cd C:\Users\your_username\downloads\yolov8_aimbot-main
```
3. Then export the model from the .pt format in .engine format.
```cmd
yolo export model="models/model1.pt" format=engine device=0 imgsz=640 half=True
```
  - `model="model_path/model_name.pt"`: Path to model.
  - `format=engine`: TensorRT model format.
  - `half=true`: Use Half-precision floating-point format.
  - `device=0`: GPU id.
  - `workspace=8`: GPU max video memory.
  - `verbose=False`: Debug stuff. Convenient function, can show errors when exporting.

## Notes / Recommendations
- Limit the maximum value of frames per second in the game in which you will use it. And also do not set the screen resolution to high. Do not overload the graphics card.
- Do not set high graphics settings in games.
- Limit the browser (try not to watch YouTube while playing and working AI at the same time, for example (of course if you don't have a super duper graphics card)) and so on, which loads the video card.
- Try to use TensorRT for acceleration. `.pt` model is good, but does not have as much speed as `.engine`.
- Turn off the cv2 debug window, this saves system resources.
- Do not increase the object search window resolution, this may affect your search speed.
- If you have started the application and nothing happens, it may be working, close it with the F2 key and change the `show_window` option to `True` in the config.ini

## if you are looking for new pt models
sunone makes great models highly reccomended you can subscribe here for the latest releases [here](https://boosty.to/sunone).

## License
This project is licensed read the notices files
