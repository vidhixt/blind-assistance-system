
import torch

def preprocess_frame(frame, detections):

    if len(detections) == 0:
        return torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    frame_width = frame.shape[1]
    frame_center_x = frame_width / 2

    # Use first detected object
    box = detections[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    object_center_x = (x1 + x2) / 2

    # Normalize position: -1 (left) to +1 (right)
    object_position = (object_center_x - frame_center_x) / frame_center_x

    # Distance: 1 = far, 0 = close
    object_width = x2 - x1
    object_distance = 1 - (object_width / frame_width)

    # Space availability
    left_space = 1.0 if x1 > 50 else 0.0
    right_space = 1.0 if x2 < (frame_width - 50) else 0.0

    return torch.tensor([object_position, object_distance, left_space, right_space], dtype=torch.float32)