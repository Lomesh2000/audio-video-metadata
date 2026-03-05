import librosa
import numpy as np
import json
import os
from moviepy import VideoFileClip

def extract_timeseries_metadata(file_path):
    print(f"--- Extracting Time-Series Audio Metadata: {file_path} ---")
    
    temp_audio = None
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        temp_audio = "temp_audio_ts.wav"
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(temp_audio, logger=None)
        audio_path = temp_audio
    else:
        audio_path = file_path
        
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Audio loaded: {duration:.2f} seconds | SR: {sr} Hz")
    
    # Define hop length for frame extraction (e.g., ~11.6ms chunks at 44.1kHz)
    hop_length = 512
    
    # 1. Compute frame-level features
    # RMS Energy
    rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Spectral Centroid
    centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Spectral Flux (difference in magnitude between consecutive frames to find transitions)
    S, _ = librosa.magphase(librosa.stft(y, hop_length=hop_length))
    # Euclidean distance between consecutive spectral frames
    flux_frames = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    # Prepend 0.0 to make it the same length as other frame arrays
    flux_frames = np.concatenate(([0.0], flux_frames))
    
    # 2. Aggregate into 1-second windows
    times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_length)
    num_seconds = int(np.ceil(duration))
    
    temporal_pulse = []
    
    for sec in range(num_seconds):
        window_mask = (times >= sec) & (times < sec + 1)
        if not np.any(window_mask):
            continue
            
        sec_rms = float(np.mean(rms_frames[window_mask]))
        sec_centroid = float(np.mean(centroid_frames[window_mask]))
        sec_flux = float(np.mean(flux_frames[window_mask]))
        
        temporal_pulse.append({
            "window_start_sec": sec,
            "window_end_sec": sec + 1,
            "rms_energy": round(sec_rms, 4),
            "spectral_centroid": round(sec_centroid, 2),
            "spectral_flux": round(sec_flux, 4)
        })
        
    # 3. Calculate Deltas (Percentage Change / First Derivative)
    for i in range(len(temporal_pulse)):
        if i == 0:
            temporal_pulse[i]["rms_delta_pct"] = 0.0
            temporal_pulse[i]["centroid_delta_pct"] = 0.0
        else:
            prev_rms = temporal_pulse[i-1]["rms_energy"]
            curr_rms = temporal_pulse[i]["rms_energy"]
            
            if prev_rms > 0:
                rms_delta = ((curr_rms - prev_rms) / prev_rms) * 100
            else:
                rms_delta = 0.0 if curr_rms == 0 else 100.0
                
            prev_cent = temporal_pulse[i-1]["spectral_centroid"]
            curr_cent = temporal_pulse[i]["spectral_centroid"]
            
            if prev_cent > 0:
                cent_delta = ((curr_cent - prev_cent) / prev_cent) * 100
            else:
                cent_delta = 0.0 if curr_cent == 0 else 100.0
                
            temporal_pulse[i]["rms_delta_pct"] = round(float(rms_delta), 2)
            temporal_pulse[i]["centroid_delta_pct"] = round(float(cent_delta), 2)
            
    # Clean up temp audio
    if temp_audio and os.path.exists(temp_audio):
        try:
            os.remove(temp_audio)
        except Exception as e:
            print(f"Warning: Could not remove temp file: {e}")
            
    return {
        "file": os.path.basename(file_path),
        "duration_sec": round(float(duration), 2),
        "windows_analyzed": len(temporal_pulse),
        "temporal_pulse": temporal_pulse
    }

if __name__ == "__main__":
    test_file = "videoplayback.mp4"
    output_file = "audio_timeseries.json"
    
    if os.path.exists(test_file):
        metadata = extract_timeseries_metadata(test_file)
        
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"\n✅ Extraction Complete!")
        print(f"Saved time-series data for {metadata['windows_analyzed']} windows to {output_file}")
    else:
        print(f"File '{test_file}' not found. Please provide a valid video or audio file.")