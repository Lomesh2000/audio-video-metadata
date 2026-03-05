import librosa
import numpy as np
import json
import os
from moviepy import VideoFileClip

def extract_audio_details(file_path):
    print(f"--- Analyzing Audio Details for: {file_path} ---")
    
    temp_audio = None
    # Check if video, extract audio if so
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        temp_audio = "temp_audio_ext.wav"
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(temp_audio, logger=None)
        audio_path = temp_audio
    else:
        audio_path = file_path
        
    # Load audio preserving its native sample rate
    y, sr = librosa.load(audio_path, sr=None) 
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Loaded audio: {duration:.2f} seconds | Sample Rate: {sr} Hz")
    
    # 1. Standard Audio Features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # MFCCs (Timbre descriptors, useful for classifying sound types)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
    
    # 2. Detailed Frequency Sub-band Analysis (Dependent on the Sample Rate)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Define fine-grained bands based on human hearing constraints
    bands = {
        "sub_bass_energy (0-60Hz)": (freqs <= 60),
        "bass_energy (60-250Hz)": (freqs > 60) & (freqs <= 250),
        "low_mid_energy (250-2000Hz)": (freqs > 250) & (freqs <= 2000),
        "high_mid_energy (2000-4000Hz)": (freqs > 2000) & (freqs <= 4000),
        "presence_energy (4000-6000Hz)": (freqs > 4000) & (freqs <= 6000),
        "brilliance_energy (6000Hz+)": (freqs > 6000)
    }
    
    band_energies = {}
    total_energy = max(float(np.sum(S**2)), 1e-10) # Avoid division by zero
    
    for band_name, mask in bands.items():
        band_energy = float(np.sum(S[mask]**2))
        band_energies[band_name] = round(band_energy / total_energy, 4)

    # Clean up temp file
    if temp_audio and os.path.exists(temp_audio):
        try:
            os.remove(temp_audio)
        except Exception as e:
            print(f"Could not remove temp audio file: {e}")
        
    return {
        "File": os.path.basename(file_path),
        "Duration_sec": round(float(duration), 2),
        "Sample_Rate_Hz": int(sr),
        "Tempo_BPM": round(float(tempo), 2),
        "Mean_RMS_Energy": round(float(np.mean(rms)), 4),
        "Mean_Zero_Crossing_Rate": round(float(np.mean(zcr)), 4),
        "Mean_Spectral_Centroid_Hz": round(float(np.mean(spectral_centroid)), 2),
        "Mean_Spectral_Bandwidth_Hz": round(float(np.mean(spectral_bandwidth)), 2),
        "Mean_Spectral_Rolloff_Hz": round(float(np.mean(spectral_rolloff)), 2),
        "MFCC_Mean_Vector": [round(float(m), 4) for m in np.mean(mfccs, axis=1)],
        "Sub_Band_Energy_Distribution": band_energies
    }

if __name__ == "__main__":
    # Test file path, fallback to videoplayback.mp4 if present in the workspace
    test_file = "videoplayback.mp4"
    
    if os.path.exists(test_file):
        details = extract_audio_details(test_file)
        
        with open("audio_specific_metadata.json", "w") as f:
            json.dump(details, f, indent=4)
            
        print("\n--- Extraction Complete ---")
        print(json.dumps(details, indent=4))
        print("\nDetailed audio metadata saved to audio_specific_metadata.json")
    else:
        print(f"File '{test_file}' not found. Please provide a valid video or audio file.")