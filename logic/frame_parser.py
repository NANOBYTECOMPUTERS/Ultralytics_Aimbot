import torch

from logic.config_watcher import cfg
from logic.capture import capture
from logic.visual import visuals
from logic.mouse import mouse

class Target():
    def __init__(self, x, y, w, h, cls):
        self.x = x
        self.y = y if cls == 7 else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h
        self.cls = cls
        
class FrameParser():
    def __init__(self):
        self.arch = self.get_arch()
            
    def parse(self, result):
        for frame in result:
            if len(frame.boxes):
                target = self.sort_targets(frame, cfg)
                
                mouse.process_data((target.x, target.y, target.w, target.h))

                if cfg.show_window:
                    if cfg.show_boxes:
                        visuals.draw_helpers(frame.boxes)
                    if cfg.show_target_line:
                        visuals.draw_target_line(target.x, target.y)
                    if cfg.show_target_prediction_line:
                        visuals.draw_predicted_position(target.x, target.y)
            else:
                if cfg.auto_shoot or cfg.triggerbot:
                    mouse.shoot(False) # release shooting buttons
                pass
            
            if cfg.show_window and cfg.show_detection_speed == True:
                visuals.draw_speed(frame.speed['preprocess'], frame.speed['inference'], frame.speed['postprocess'])
                
    def sort_targets(self, frame, cfg) -> Target:
        boxes_array = frame.boxes.xywh.to(self.arch)
        distances_sq = torch.sum((boxes_array[:, :2] - torch.tensor([capture.screen_x_center, capture.screen_y_center], device=self.arch)) ** 2, dim=1)
        classes_tensor = frame.boxes.cls.to(self.arch)

        if not cfg.disable_headshot:
            head_indices = torch.nonzero(classes_tensor == 7, as_tuple=False).squeeze(1)
            if len(head_indices) > 0:
                head_distances_sq = distances_sq[head_indices]
                nearest_head_idx = head_indices[torch.argmin(head_distances_sq)]
                return Target(*boxes_array[nearest_head_idx, :4].cpu().numpy(), classes_tensor[nearest_head_idx].item())
            else:
                nearest_idx = torch.argmin(distances_sq)
                return Target(*boxes_array[nearest_idx, :4].cpu().numpy(), classes_tensor[nearest_idx].item())
        else:
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