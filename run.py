import torch
import threading
import ctypes
import ctypes.util
import ctypes.wintypes
import ctypes
from ctypes import byref
from ctypes.wintypes import MSG
from ultralytics import YOLO
from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.frame_parser import frameParser
from logic.hotkeys_watcher import hotkeys_watcher
from logic.checks import run_checks
from logic.sys_mouse import select_mouse, mouse, capture_and_retransmit_mouse_input
from logic.mouse import MouseThread
from ctypes.wintypes import MSG
import queue  # Add this line to import the queue module
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


@torch.inference_mode()
def perform_detection(model, image):
    return model.predict(
        source=image,
        cfg='logic/game.yaml',
        imgsz=cfg.ai_model_image_size,
        stream=True,
        conf=cfg.AI_conf,
        iou=0.5,
        device=cfg.AI_device,
        half=False if 'cpu' in cfg.AI_device else True,
        max_det=20,
        agnostic_nms=False,
        augment=False,
        vid_stride=False,
        visualize=False,
        verbose=False,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        save=False,
        show=False)
        
def init():
    run_checks()

    try:
        model = YOLO(f'models/{cfg.AI_model_name}', task='detect')
    except Exception as e:
        print('An error occurred when loading the AI model:\n', e)
        quit(0)

    selected_mouse_id = select_mouse()

    if selected_mouse_id:
        mouse_queue = queue.Queue() 
        mouse = MouseThread()  # Create MouseThread instance
        
        # Start the mouse capture thread
        output_method = "win32" # or "ghub" or "arduino"
        mouse_capture_thread = threading.Thread(
            target=capture_and_retransmit_mouse_input, 
            args=(selected_mouse_id, mouse_queue, output_method)
        )
        mouse_capture_thread.daemon = True
        mouse_capture_thread.start()

        while True:
            image = capture.get_new_frame()

            if cfg.show_window or cfg.show_overlay:
                visuals.queue.put(image)

            if image is not None:
                result = perform_detection(model, image)

                if hotkeys_watcher.app_pause == 0:
                    frameParser.parse(result)

                    # Process mouse input from the queue
                    while not mouse_queue.empty():
                        mouse_x, mouse_y, is_movement = mouse_queue.get()
                        if is_movement:  
                            mouse.process_mouse_input(mouse_x, mouse_y)  

    else:
        print("No mouse selected or no mice detected.")

if __name__ == "__main__":
    init()