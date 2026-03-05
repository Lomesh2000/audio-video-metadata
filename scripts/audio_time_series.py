import librosa
import numpy as np
import json
import os
from moviepy import VideoFileClip

def extract_audio_time_series(file_path, fps=30):
    print(f"--- Extracting Temporal Stimuli for: {file_path} ---")
    
    temp_audio = "temp_proc.wav"
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(temp_audio, logger=None)
        audio_path = temp_audio
    else:
        audio_path = file_path
        
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Align audio 'hop' with video FPS
    # This ensures one audio data point per video frame
    hop_length = int(sr / fps)
    
    # 1. TIME-SERIES FEATURE EXTRACTION
    # RMS (Raw Energy/Intensity)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Spectral Centroid (Sharpness/Brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Spectral Flux (Transition/Change intensity)
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Zero Crossing Rate (Noisiness/Distortion)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

    # 2. CALCULATING HIKES & DOWNSIDES (The Derivative)
    # We calculate the percentage change between subsequent windows
    def calculate_hike(arr):
        # Add small epsilon to avoid division by zero
        diff = np.diff(arr)
        hike = (diff / (arr[:-1] + 1e-6)) * 100
        return np.insert(hike, 0, 0).tolist() # Pad first element

    rms_hike = calculate_hike(rms)
    centroid_hike = calculate_hike(centroid)

    # 3. SUB-BAND TEMPORAL ANALYSIS
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    
    bands = {
        "low_end": (freqs <= 250),
        "mid_range": (freqs > 250) & (freqs <= 4000),
        "high_end": (freqs > 4000)
    }

    temporal_pulse = []
    for i in range(len(rms)):
        # Calculate energy for this specific window
        total_spec_energy = np.sum(S[:, i]**2) + 1e-10
        
        frame_data = {
            "second": round(i / fps, 3),
            "intensity_rms": float(rms[i]),
            "intensity_hike_pct": round(float(rms_hike[i]), 2),
            "brightness_hz": float(centroid[i]),
            "brightness_hike_pct": round(float(centroid_hike[i]), 2),
            "transition_flux": float(flux[i]),
            "noise_zcr": float(zcr[i]),
            "bands": {
                "low": float(np.sum(S[bands["low_end"], i]**2) / total_spec_energy),
                "mid": float(np.sum(S[bands["mid_range"], i]**2) / total_spec_energy),
                "high": float(np.sum(S[bands["high_end"], i]**2) / total_spec_energy)
            }
        }
        temporal_pulse.append(frame_data)

    # Clean up
    if os.path.exists(temp_audio):
        os.remove(temp_audio)

    metadata = {
        "file": os.path.basename(file_path),
        "global_tempo_bpm": float(np.squeeze(librosa.beat.beat_track(y=y, sr=sr)[0])),
        "temporal_pulse": temporal_pulse
    }
    
    return metadata

if __name__ == "__main__":
    test_file = "videoplayback.mp4" 
    if os.path.exists(test_file):
        data = extract_audio_time_series(test_file)
        with open("audio_time_series.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Done. Temporal stimuli saved to audio_time_series.json")