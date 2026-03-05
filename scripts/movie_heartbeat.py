import cv2
import numpy as np
import json 

def extract_transition_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_metadata = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. MOTION METADATA: Farneback Optical Flow
        # Tracks how much 'energy' moves from Frame A to Frame B
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_energy = np.mean(mag)
        
        # 2. COLOR PULSE: Mean Brightness Change
        brightness = np.mean(gray)
        brightness_diff = abs(brightness - np.mean(prev_gray))
        
        frame_metadata.append({
            "motion_energy": float(round(motion_energy, 4)),
            "brightness_shift": float(round(brightness_diff, 4))
        })
        
        prev_gray = gray
        
    cap.release()
    return frame_metadata

data = extract_transition_metadata('videoplayback.mp4')

print("Total frames analyzed:", len(data))

# Print first 10 frames metadata
for i in range(10):
    print(data[i])

with open("video_metadata.json", "w") as f:
    json.dump(data, f, indent=4)

print("Metadata saved to video_metadata.json")