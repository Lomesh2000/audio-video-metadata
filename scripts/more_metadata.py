import cv2
import numpy as np
import librosa
import json
import time
import os
from scenedetect import detect, ContentDetector
from moviepy import VideoFileClip
from transformers import CLIPProcessor, CLIPModel
import torch

# -----------------------------
# VISUAL METADATA
# -----------------------------
def compute_visual_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    color_variance = float(np.var(frame))
    contrast = float(gray.std())

    brightness = float(np.mean(gray))
    lighting_key = "low-key" if brightness < 80 else "high-key"

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)

    return {
        "color_variance": color_variance,
        "contrast": contrast,
        "lighting_key": lighting_key,
        "edge_density": edge_density
    }

# -----------------------------
# AUDIO METADATA
# -----------------------------
def compute_audio_features(audio_path):
    print("🔊 [Checkpoint] Loading audio...")
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"⚠️ [Error] Could not load audio: {e}")
        return {}

    print("🔊 [Checkpoint] Extracting audio features...")
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle Librosa 0.10+ tempo array return type safely
    tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    onset_strength = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
    print("✅ [Checkpoint] Audio feature extraction complete")

    return {
        "spectral_centroid": spectral_centroid,
        "zero_crossing_rate": zcr,
        "tempo_bpm": tempo_val,
        "onset_strength": onset_strength
    }

# -----------------------------
# SCENE / PACING METADATA
# -----------------------------
def compute_scene_features(video_path):
    print("🎬 [Checkpoint] Running scene detection...")
    scenes = detect(video_path, ContentDetector(threshold=27.0))

    durations = [end.get_seconds() - start.get_seconds() for start, end in scenes]
    print(f"🎬 [Checkpoint] Scene detection complete ({len(scenes)} scenes found)")

    return {
        "num_scenes": len(scenes),
        "shot_density": len(scenes),
        "avg_scene_duration": float(np.mean(durations)) if durations else 0.0
    }

# -----------------------------
# SEMANTIC DRIFT (CLIP)
# -----------------------------
print("🧠 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("🧠 CLIP model loaded")

def compute_semantic_drift(frames):
    print(f"🧠 [Checkpoint] Computing semantic drift across {len(frames)} frames...")
    embeddings = []

    for i, frame in enumerate(frames):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        inputs = clip_processor(images=[rgb_frame], return_tensors="pt")
        
        with torch.no_grad():
            output = clip_model.get_image_features(**inputs)
        
        # FOOLPROOF EXTRACTION: Unwrap the tensor from the HuggingFace object
        if hasattr(output, "pooler_output"):
            emb = output.pooler_output
        elif hasattr(output, "image_embeds"):
            emb = output.image_embeds
        elif isinstance(output, tuple):
            emb = output[0]
        else:
            emb = output # Fallback if it is already a pure PyTorch tensor

        if i == 0:
            print(f"DEBUG: Unwrapped emb shape: {emb.shape}")
            
        embeddings.append(emb[0])

    drift = []
    for i in range(1, len(embeddings)):
        vec1 = embeddings[i-1].unsqueeze(0)
        vec2 = embeddings[i].unsqueeze(0)

        sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)
        
        if i == 1:
            print(f"DEBUG: sim shape: {sim.shape}, sim numel: {sim.numel()}")

        drift.append(float(1 - sim.item()))

    print("✅ [Checkpoint] Semantic drift computed")
    return float(np.mean(drift)) if drift else 0.0

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def extract_metadata(video_path):
    start_time = time.time()
    print("\n🚀 Starting metadata extraction pipeline\n")

    # ---------------- VISUAL ----------------
    cap = cv2.VideoCapture(video_path)
    sampled_frames = []
    visual_data = []
    frame_idx = 0
    sample_rate = 10 

    print("🎥 [Checkpoint] Processing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            features = compute_visual_features(frame)
            visual_data.append(features)
            sampled_frames.append(frame)

            if frame_idx % 300 == 0:
                print(f"   Processed frame {frame_idx}")

        frame_idx += 1
    cap.release()
    print(f"✅ [Checkpoint] Video frame processing complete ({len(sampled_frames)} sampled frames)")

    # ---------------- AUDIO ----------------
    print("\n🎧 [Checkpoint] Extracting audio from video...")
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    audio_features = {}

    # FIX: Safety check for videos without an audio track
    if video.audio is not None:
        video.audio.write_audiofile(audio_path, logger=None)
        audio_features = compute_audio_features(audio_path)
        if os.path.exists(audio_path):
            os.remove(audio_path) # Clean up temp file
    else:
        print("⚠️ [Warning] No audio track found in video.")
        audio_features = {"error": "No audio track"}
    
    video.close() # FIX: Prevent resource leaks
    print("✅ [Checkpoint] Audio processing step complete")

    # ---------------- SCENE ----------------
    scene_features = compute_scene_features(video_path)

    # ---------------- SEMANTIC ----------------
    # FIX: Evenly sample 20 frames across the whole video, not just the first 20
    if len(sampled_frames) > 20:
        step = len(sampled_frames) // 20
        drift_frames = sampled_frames[::step][:20]
    else:
        drift_frames = sampled_frames
        
    semantic_drift = compute_semantic_drift(drift_frames)

    # ---------------- ASSEMBLE ----------------
    metadata = {
        "visual_features": visual_data,
        "audio_features": audio_features,
        "scene_features": scene_features,
        "semantic_drift": semantic_drift
    }

    end_time = time.time()
    print(f"\n⏱ Total pipeline runtime: {round(end_time - start_time, 2)} seconds")
    return metadata

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    video_file = "videoplayback.mp4"
    
    if not os.path.exists(video_file):
        print(f"❌ Error: File '{video_file}' not found.")
    else:
        metadata = extract_metadata(video_file)

        print("\n💾 Saving metadata to JSON...")
        with open("cinematic_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print("✅ Metadata extraction complete")