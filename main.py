import cv2
import torch
print(torch.__version__)
import time
import threading
import pyttsx3
from model import DQN
from utils import preprocess_frame
from ultralytics import YOLO

# Constants
INPUT_DIM = 4  # [object_position, object_distance, left_space, right_space]
OUTPUT_DIM = 3  # [Move Left, Move Right, Stay]
EPSILON = 0.05  # Exploration rate
ANNOUNCE_DELAY = 5  # Seconds between repeated announcements

# Accuracy tracking
correct_decisions = 0
total_decisions = 0

# Text-to-speech
def speak(text):
    tts_engine = pyttsx3.init()
    tts_engine.say(text)
    tts_engine.runAndWait()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained DQN model
policy_net = DQN(INPUT_DIM, OUTPUT_DIM)
policy_net.load_state_dict(torch.load("dodge_model.pth", map_location=device))
policy_net.to(device)
policy_net.eval()

# Load YOLO model
model = YOLO("yolov5s.pt")

# Start video capture
cap = cv2.VideoCapture(0)

last_announcement = ""
last_announcement_time = 0

print("Starting system... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Run YOLO detection
    results = model(frame)
    detections = results[0].boxes
    names = model.names

    # Default label
    detected_object_name = "nothing"

    if detections is not None and len(detections) > 0:
        cls_id = int(detections[0].cls.item())
        detected_object_name = names[cls_id]

    # Preprocess frame for model input
    state = preprocess_frame(frame, detections).unsqueeze(0).to(device)

    # Choose action using DQN (ε-greedy)
    if torch.rand(1).item() < EPSILON:
        action = torch.randint(0, OUTPUT_DIM, (1,)).item()
    else:
        with torch.no_grad():
            q_values = policy_net(state)
            action = torch.argmax(q_values).item()

    # Convert action to readable text
    action_text = ["Move Left", "Move Right", "Stay"][action]
    announcement = f"{detected_object_name}: {action_text}"

    # Say the decision aloud if it's new or delayed
    current_time = time.time()
    if announcement != last_announcement or (current_time - last_announcement_time) > ANNOUNCE_DELAY:
        print(f"Detected: {detected_object_name}, Decision: {action_text}")
        threading.Thread(target=speak, args=(f"{detected_object_name}. {action_text}",)).start()
        last_announcement = announcement
        last_announcement_time = current_time

    # -------- Accuracy Tracking --------
    if detections is not None and len(detections) > 0:
        # Get bounding box of first detection
        x1, y1, x2, y2 = detections[0].xyxy[0]
        frame_width = frame.shape[1]

        # Unflip the box coordinates to match real-world position
        flipped_x1 = frame_width - x2.item()
        flipped_x2 = frame_width - x1.item()
        object_center_x = (flipped_x1 + flipped_x2) / 2

        # Normalize position: -0.5 (left) to +0.5 (right)
        obj_position = (object_center_x - frame_width / 2) / frame_width

        # Define correct action
        if obj_position < -0.1:
            correct_action = 0  # Move Left
        elif obj_position > 0.1:
            correct_action = 1  # Move Right
        else:
            correct_action = 2  # Stay

        # Accuracy count
        if action == correct_action:
            correct_decisions += 1
        total_decisions += 1

        # Debug info
        print(f"[DEBUG] Obj_Pos: {obj_position:.2f}, Model: {action}, Correct: {correct_action}, Acc: {correct_decisions}/{total_decisions}")

    # Accuracy update every 100 decisions
    if total_decisions > 0 and total_decisions % 100 == 0:
        accuracy = (correct_decisions / total_decisions) * 100
        print(f"[Accuracy] Correct: {correct_decisions}/{total_decisions} — Accuracy: {accuracy:.2f}%")

    # Show frame
    cv2.imshow("Object Detection + Decision Making", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Final result
if total_decisions > 0:
    final_accuracy = (correct_decisions / total_decisions) * 100
    print(f"\n[FINAL RESULT] Overall Accuracy: {final_accuracy:.2f}%")