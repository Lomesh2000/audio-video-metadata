import librosa
import numpy as np
from scenedetect import detect, ContentDetector
from moviepy import VideoFileClip
import os
import json

def calculate_stimulation(video_path):
    print(f"--- Analyzing: {video_path} ---")
    
    # 1. VISUAL STIMULATION: Shot Density
    scenes = detect(video_path, ContentDetector(threshold=27.0))
    num_scenes = len(scenes)
    
    # 2. Extract audio using moviepy
    temp_audio = "temp_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio, logger=None)
    
    # Load audio with librosa
    y, sr = librosa.load(temp_audio, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    shot_density = num_scenes / (duration / 60)
    
    # 3. AUDIO STIMULATION: RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)

    # Sub-band Energy Extraction
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Define bands
    low_mask = (freqs <= 250)
    mid_mask = (freqs > 250) & (freqs <= 4000)
    high_mask = (freqs > 4000)

    # Calculate Mean Energy (magnitude squared) for each band
    low_energy_mean = np.mean(np.sum(S[low_mask]**2, axis=0))
    mid_energy_mean = np.mean(np.sum(S[mid_mask]**2, axis=0))
    high_energy_mean = np.mean(np.sum(S[high_mask]**2, axis=0))

    total_energy = low_energy_mean + mid_energy_mean + high_energy_mean

    # Normalize to percentage
    if total_energy > 0:
        low_pct = low_energy_mean / total_energy
        mid_pct = mid_energy_mean / total_energy
        high_pct = high_energy_mean / total_energy
    else:
        low_pct = mid_pct = high_pct = 0.0
    
    # 4. NORMALIZED STIMULATION SCORE
    v_score = min((shot_density / 30) * 100, 100)
    a_score = min((avg_rms / 0.1) * 100, 100)
    
    final_score = (0.6 * v_score) + (0.4 * a_score)
    
    # Clean up temp file
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    return {
        "Final Stimulation Score": float(round(final_score, 2)),
        "Cuts Per Minute": float(round(shot_density, 2)),
        "Audio Energy": float(round(avg_rms, 4)),
        "low_freq_energy": float(round(low_pct, 4)),
        "mid_freq_energy": float(round(mid_pct, 4)),
        "high_freq_energy": float(round(high_pct, 4))
    }


result = calculate_stimulation("videoplayback.mp4")
print(result)

with open("stimulation_score.json", "w") as f:
    json.dump(result, f, indent=4)