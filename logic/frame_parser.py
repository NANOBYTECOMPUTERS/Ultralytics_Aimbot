#[frame_parser.py]
import torch
import numpy as np

from logic.hotkeys_watcher import hotkeys_watcher
from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.mouse import mouse
from logic.shooting import shooting

class Target:
    def __init__(self, x, y, w, h, cls):
        self.x = x
        self.y = y if cls == 7 else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h
        self.cls = cls
        
class FrameParser:
    def __init__(self):
        self.arch = self.get_arch()
        
    def parse(self, result):
        for frame in result:
            if frame.boxes:  # If there are any detections
                # Find the closest target using the new function
                target, player_box_sizes, head_box_sizes = self.find_closest_target(frame)

                if target:
                    if hotkeys_watcher.clss is None:
                        hotkeys_watcher.active_classes()
                    if target.cls in hotkeys_watcher.clss:
                        mouse.process_data((target.x, target.y, target.w, target.h, target.cls, player_box_sizes, head_box_sizes))

                if cfg.show_window or cfg.show_overlay:
                    if cfg.show_boxes or cfg.overlay_show_boxes:
                        visuals.draw_helpers(frame.boxes)
            else:
                # no detections
                if cfg.auto_shoot or cfg.triggerbot:
                    shooting.shoot(False, False)
                if cfg.show_window or cfg.show_overlay:
                    visuals.clear()

            if cfg.show_window and cfg.show_detection_speed:
                visuals.draw_speed(frame.speed['preprocess'], frame.speed['inference'], frame.speed['postprocess'])

    def find_closest_target(self, frame):
        boxes_array = frame.boxes.xywh.cpu().numpy()
        center = (capture.screen_x_center, capture.screen_y_center)

        distances_sq = np.sum((boxes_array[:, :2] - center) ** 2, axis=1)
        classes_array = frame.boxes.cls.cpu().numpy()

        # Find sizes of player and head boxes
        player_box_sizes = []
        head_box_sizes = []
        for box, cls in zip(boxes_array, classes_array):
            w, h = box[2:]
            if cls == 7:  # Assuming class 7 is for heads
                head_box_sizes.append((w, h))
            else:
                player_box_sizes.append((w, h))

        if not cfg.disable_headshot:
            # Prioritize heads closest to the center
            head_indices = np.where(classes_array == 7)[0]
            if len(head_indices) > 0:
                nearest_head_idx = head_indices[np.argmin(distances_sq[head_indices])]
                return Target(*boxes_array[nearest_head_idx], classes_array[nearest_head_idx]), player_box_sizes, head_box_sizes  # Return 3 values

        # If no heads or headshot is disabled, use the nearest target
        nearest_idx = np.argmin(distances_sq)
        return Target(*boxes_array[nearest_idx], classes_array[nearest_idx]), player_box_sizes, head_box_sizes
                
    def sort_targets(self, frame):
        boxes_array = frame.boxes.xywh.to(self.arch)
        center = torch.tensor([capture.screen_x_center, capture.screen_y_center], device=self.arch)
        
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)
        
        classes_tensor = frame.boxes.cls.to(self.arch)
        
        if not classes_tensor.numel():
            return None

        if not cfg.disable_headshot:
            head_indices = torch.nonzero(classes_tensor == 7, as_tuple=False).squeeze(1)
            if head_indices.numel():
                head_distances_sq = distances_sq[head_indices]
                nearest_head_idx = head_indices[torch.argmin(head_distances_sq)]
                return Target(*boxes_array[nearest_head_idx, :4].cpu().numpy(), classes_tensor[nearest_head_idx].item())
        
        nearest_idx = torch.argmin(distances_sq)
        return Target(*boxes_array[nearest_idx, :4].cpu().numpy(), classes_tensor[nearest_idx].item())
    
    def get_arch(self):
        if cfg.AI_enable_AMD:
            arch = f'hip:{cfg.AI_device}'
        else:
            arch = f'cuda:{cfg.AI_device}'
        if 'cpu' in cfg.AI_device:
            arch = 'cpu'
        return arch
    
frameParser = FrameParser()
#[/frame_parser.py]