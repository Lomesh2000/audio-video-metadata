"""
temporal_features.py
====================
Sequential, per-frame/per-second feature extraction for sequence-aware RecSys
models (Transformers, LSTMs, attention layers) and engagement prediction.

Consolidates (zero data loss):
  - movie_heartbeat.py    → per-frame optical flow motion energy + brightness shift
  - audio_time_series.py  → FPS-aligned per-frame audio (rms, centroid, flux,
                             zcr, sub-band energy, hike/derivative signals)
  - audio_timeseries.py   → 1-second windowed audio (rms, centroid, flux, deltas)

Output schema:
{
  "file":             str,
  "duration_sec":     float,
  "sample_rate_hz":   int,
  "video_fps":        float,
  "global_tempo_bpm": float,

  # ── Per-frame (one entry per video frame, aligned by frame index) ──
  "per_frame": [
    {
      "frame_idx":           int,
      "timestamp_sec":       float,

      # from movie_heartbeat.py
      "motion_energy":       float,   # mean optical-flow magnitude
      "brightness_shift":    float,   # abs brightness diff from prev frame

      # from audio_time_series.py (FPS-aligned)
      "intensity_rms":       float,
      "intensity_hike_pct":  float,   # % change in rms vs previous frame
      "brightness_hz":       float,   # spectral centroid
      "brightness_hike_pct": float,   # % change in centroid vs previous frame
      "transition_flux":     float,   # onset strength
      "noise_zcr":           float,
      "audio_bands": {
          "low":  float,              # <= 250 Hz fraction
          "mid":  float,              # 250–4000 Hz fraction
          "high": float               # > 4000 Hz fraction
      }
    },
    ...
  ],

  # ── Per-second windows (one entry per second, from audio_timeseries.py) ──
  "per_second": [
    {
      "window_start_sec":   int,
      "window_end_sec":     int,
      "rms_energy":         float,
      "spectral_centroid":  float,
      "spectral_flux":      float,
      "rms_delta_pct":      float,    # % change vs previous second
      "centroid_delta_pct": float
    },
    ...
  ]
}
"""

import os
import json
import time

import cv2
import numpy as np
import librosa
from moviepy import VideoFileClip


# ---------------------------------------------------------------------------
# Audio extraction helper
# ---------------------------------------------------------------------------
def _extract_audio(video_path: str, temp_name: str = "temp_temporal_audio.wav") -> tuple:
    """
    Extract audio to WAV if needed; returns (y, sr, temp_path_or_None).
    """
    if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            raise ValueError(f"No audio track found in '{video_path}'")
        video.audio.write_audiofile(temp_name, logger=None)
        video.close()
        audio_path = temp_name
    else:
        audio_path = video_path
        temp_name  = None

    y, sr = librosa.load(audio_path, sr=None)
    return y, sr, temp_name


# ---------------------------------------------------------------------------
# Per-frame video features (movie_heartbeat.py)
# ---------------------------------------------------------------------------
def _extract_video_frame_features(video_path: str) -> tuple:
    """
    Compute per-frame motion energy and brightness shift via Farneback optical flow.
    Source: movie_heartbeat.py → extract_transition_metadata()

    Returns
    -------
    (frame_data_list, fps)
      frame_data_list : list of {frame_idx, motion_energy, brightness_shift}
      fps             : video frame rate
    """
    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return [], fps

    prev_gray   = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_data  = []
    frame_idx   = 1   # frame 0 has no previous; skip it (same as original)

    print("  [VideoFrames] Computing optical flow per frame...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _         = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_energy  = float(round(np.mean(mag), 4))
        brightness_shift = float(round(abs(np.mean(gray) - np.mean(prev_gray)), 4))

        frame_data.append({
            "frame_idx":       frame_idx,
            "motion_energy":   motion_energy,
            "brightness_shift": brightness_shift
        })

        prev_gray = gray
        frame_idx += 1

    cap.release()
    print(f"  [VideoFrames] Done — {len(frame_data)} frames processed.")
    return frame_data, fps


# ---------------------------------------------------------------------------
# FPS-aligned per-frame audio features (audio_time_series.py)
# ---------------------------------------------------------------------------
def _extract_fps_audio_features(y: np.ndarray, sr: int, fps: float) -> list:
    """
    Extract per-frame audio features aligned to video FPS.
    Source: audio_time_series.py → extract_audio_time_series()

    One audio data point per video frame.
    Fields: intensity_rms, intensity_hike_pct, brightness_hz,
            brightness_hike_pct, transition_flux, noise_zcr, audio_bands
    """
    hop_length = int(sr / fps)

    rms      = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    flux     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    zcr      = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

    # Sub-band energies per frame
    S     = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    low_mask  = freqs <= 250
    mid_mask  = (freqs > 250) & (freqs <= 4000)
    high_mask = freqs > 4000

    # Percentage change helper (from audio_time_series.py → calculate_hike)
    def _hike(arr):
        diff = np.diff(arr)
        h    = (diff / (arr[:-1] + 1e-6)) * 100
        return np.insert(h, 0, 0).tolist()

    rms_hike      = _hike(rms)
    centroid_hike = _hike(centroid)

    n_frames   = len(rms)
    frame_list = []

    for i in range(n_frames):
        total_spec = np.sum(S[:, i] ** 2) + 1e-10
        frame_list.append({
            "intensity_rms":       float(rms[i]),
            "intensity_hike_pct":  round(float(rms_hike[i]), 2),
            "brightness_hz":       float(centroid[i]),
            "brightness_hike_pct": round(float(centroid_hike[i]), 2),
            "transition_flux":     float(flux[i]),
            "noise_zcr":           float(zcr[i]),
            "audio_bands": {
                "low":  float(np.sum(S[low_mask,  i] ** 2) / total_spec),
                "mid":  float(np.sum(S[mid_mask,  i] ** 2) / total_spec),
                "high": float(np.sum(S[high_mask, i] ** 2) / total_spec)
            }
        })

    return frame_list


# ---------------------------------------------------------------------------
# Per-second windowed audio features (audio_timeseries.py)
# ---------------------------------------------------------------------------
def _extract_per_second_audio(y: np.ndarray, sr: int, duration: float) -> list:
    """
    Aggregate audio into 1-second windows with delta (% change) signals.
    Source: audio_timeseries.py → extract_timeseries_metadata()

    Fields: window_start_sec, window_end_sec, rms_energy, spectral_centroid,
            spectral_flux, rms_delta_pct, centroid_delta_pct
    """
    hop_length = 512

    rms_frames      = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # Spectral flux: Euclidean distance between consecutive magnitude frames
    S, _         = librosa.magphase(librosa.stft(y, hop_length=hop_length))
    flux_frames  = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux_frames  = np.concatenate(([0.0], flux_frames))  # pad first frame

    times       = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop_length)
    num_seconds = int(np.ceil(duration))

    per_second = []
    for sec in range(num_seconds):
        mask = (times >= sec) & (times < sec + 1)
        if not np.any(mask):
            continue
        per_second.append({
            "window_start_sec":  sec,
            "window_end_sec":    sec + 1,
            "rms_energy":        round(float(np.mean(rms_frames[mask])),      4),
            "spectral_centroid": round(float(np.mean(centroid_frames[mask])), 2),
            "spectral_flux":     round(float(np.mean(flux_frames[mask])),     4),
            "rms_delta_pct":     0.0,
            "centroid_delta_pct": 0.0
        })

    # Compute deltas (% change, first derivative) — audio_timeseries.py logic
    for i in range(1, len(per_second)):
        prev_rms  = per_second[i - 1]["rms_energy"]
        curr_rms  = per_second[i]["rms_energy"]
        rms_delta = ((curr_rms - prev_rms) / prev_rms * 100) if prev_rms > 0 \
                    else (0.0 if curr_rms == 0 else 100.0)

        prev_cent  = per_second[i - 1]["spectral_centroid"]
        curr_cent  = per_second[i]["spectral_centroid"]
        cent_delta = ((curr_cent - prev_cent) / prev_cent * 100) if prev_cent > 0 \
                     else (0.0 if curr_cent == 0 else 100.0)

        per_second[i]["rms_delta_pct"]      = round(float(rms_delta),  2)
        per_second[i]["centroid_delta_pct"] = round(float(cent_delta), 2)

    return per_second


# ---------------------------------------------------------------------------
# Merge per-frame video + audio (align by frame index)
# ---------------------------------------------------------------------------
def _merge_per_frame(video_frames: list, audio_frames: list, fps: float) -> list:
    """
    Zip video motion features and FPS-aligned audio features by frame index.
    Shortest list sets length (handles minor off-by-one from codec rounding).
    """
    n = min(len(video_frames), len(audio_frames))
    merged = []

    for i in range(n):
        vf = video_frames[i]
        af = audio_frames[i]
        merged.append({
            "frame_idx":           vf["frame_idx"],
            "timestamp_sec":       round(vf["frame_idx"] / fps, 4),

            # from movie_heartbeat.py
            "motion_energy":       vf["motion_energy"],
            "brightness_shift":    vf["brightness_shift"],

            # from audio_time_series.py
            "intensity_rms":       af["intensity_rms"],
            "intensity_hike_pct":  af["intensity_hike_pct"],
            "brightness_hz":       af["brightness_hz"],
            "brightness_hike_pct": af["brightness_hike_pct"],
            "transition_flux":     af["transition_flux"],
            "noise_zcr":           af["noise_zcr"],
            "audio_bands":         af["audio_bands"]
        })

    return merged


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def extract_temporal_features(video_path: str) -> dict:
    """
    Extract all sequential/temporal features for a movie.

    Parameters
    ----------
    video_path : path to video (or audio-only) file

    Returns
    -------
    Full feature dict (schema at top of file).
    """
    start = time.time()
    print(f"\n[TemporalFeatures] Starting: {video_path}")

    # ── Audio (shared y/sr across all audio extractors) ──────────────────
    print("  [Audio] Extracting audio track...")
    y, sr, temp_audio = _extract_audio(video_path)
    duration          = librosa.get_duration(y=y, sr=sr)
    global_tempo, _   = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(global_tempo, np.ndarray):
        global_tempo = global_tempo[0]
    print(f"  [Audio] Done — {duration:.2f}s @ {sr} Hz | Tempo: {float(global_tempo):.1f} BPM")

    # ── Video per-frame (optical flow) ───────────────────────────────────
    video_frame_data, fps = _extract_video_frame_features(video_path)

    # ── FPS-aligned per-frame audio ──────────────────────────────────────
    print("  [AudioFrames] Extracting FPS-aligned audio features...")
    audio_frame_data = _extract_fps_audio_features(y, sr, fps)
    print(f"  [AudioFrames] Done — {len(audio_frame_data)} audio frames.")

    # ── 1-second windowed audio ──────────────────────────────────────────
    print("  [AudioSeconds] Extracting 1-second windowed audio features...")
    per_second_data = _extract_per_second_audio(y, sr, duration)
    print(f"  [AudioSeconds] Done — {len(per_second_data)} second windows.")

    # ── Merge per-frame streams ───────────────────────────────────────────
    per_frame_merged = _merge_per_frame(video_frame_data, audio_frame_data, fps)

    # ── Cleanup ──────────────────────────────────────────────────────────
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)

    elapsed = round(time.time() - start, 2)
    print(f"[TemporalFeatures] Complete in {elapsed}s\n")

    return {
        "file":             os.path.basename(video_path),
        "duration_sec":     round(float(duration), 2),
        "sample_rate_hz":   int(sr),
        "video_fps":        round(float(fps), 4),
        "global_tempo_bpm": round(float(global_tempo), 2),
        "per_frame":        per_frame_merged,
        "per_second":       per_second_data
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Look for the video file in the parent folder (the root workspace)
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "videoplayback.mp4"))

    if not os.path.exists(test_file):
        print(f"File '{test_file}' not found.")
    else:
        features = extract_temporal_features(test_file)
        
        # Make sure output directory exists
        out_dir = os.path.join(os.path.dirname(__file__), "metadata")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "temporal_features.json")
        
        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)
        print(f"\nTotal frames:   {len(features['per_frame'])}")
        print(f"Total seconds:  {len(features['per_second'])}")
        print(f"Saved to {out_path}")
