"""
global_features.py
==================
Static, per-movie feature extraction for RecSys item embeddings.

Consolidates (zero data loss):
  - audio_metadata_extractor.py  → detailed audio + 6-band sub-band energy
  - extract_movie_metadata.py    → stimulation score, 3-band energy ratios, shot density
  - more_metadata.py             → visual frame features, onset strength,
                                   scene pacing, CLIP semantic drift

Output schema:
{
  "file":           str,
  "duration_sec":   float,
  "sample_rate_hz": int,

  # --- Audio: Core ---
  "tempo_bpm":                  float,
  "mean_rms_energy":            float,
  "mean_zero_crossing_rate":    float,
  "mean_spectral_centroid_hz":  float,
  "mean_spectral_bandwidth_hz": float,
  "mean_spectral_rolloff_hz":   float,
  "mean_onset_strength":        float,         # from more_metadata.py
  "mfcc_mean_vector":           List[float],   # 13 coefficients

  # --- Audio: Sub-band energy (6-band proportional) ---
  "sub_band_energy": {
      "sub_bass_0_60hz":        float,
      "bass_60_250hz":          float,
      "low_mid_250_2000hz":     float,
      "high_mid_2000_4000hz":   float,
      "presence_4000_6000hz":   float,
      "brilliance_6000hz_plus": float
  },

  # --- Audio: 3-band energy (normalised %, from extract_movie_metadata.py) ---
  "three_band_energy": {
      "low_pct":  float,
      "mid_pct":  float,
      "high_pct": float
  },

  # --- Visual: Per-frame averages (sampled every visual_sample_rate frames) ---
  "visual_features_avg": {
      "color_variance": float,
      "contrast":       float,
      "edge_density":   float
  },
  "dominant_lighting_key": str,  # "low-key" | "high-key"

  # --- Scene / Pacing ---
  "num_scenes":             int,
  "cuts_per_minute":        float,
  "avg_scene_duration_sec": float,

  # --- Stimulation Score (formula from extract_movie_metadata.py) ---
  "stimulation": {
      "visual_score": float,   # 0-100
      "audio_score":  float,   # 0-100
      "final_score":  float    # 0.6*visual + 0.4*audio, capped at 100
  },

  # --- Semantic ---
  "semantic_drift": float      # mean cosine distance across CLIP frame embeddings
}
"""

import os
import json
import time

import cv2
import numpy as np
import librosa
import torch
from moviepy import VideoFileClip
from scenedetect import detect, ContentDetector
from transformers import CLIPProcessor, CLIPModel


# ---------------------------------------------------------------------------
# Lazy CLIP loader — loaded once and reused across calls in the same process
# ---------------------------------------------------------------------------
_clip_model     = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("  [CLIP] Loading model (first call only)...")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("  [CLIP] Model loaded.")
    return _clip_model, _clip_processor


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------
def _compute_frame_features(frame: np.ndarray) -> dict:
    """
    Per-frame visual features.
    Source: more_metadata.py → compute_visual_features()
    Returns: color_variance, contrast, lighting_key, edge_density
    """
    gray           = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color_variance = float(np.var(frame))
    contrast       = float(gray.std())
    brightness     = float(np.mean(gray))
    lighting_key   = "low-key" if brightness < 80 else "high-key"
    edges          = cv2.Canny(gray, 100, 200)
    edge_density   = float(np.sum(edges > 0) / edges.size)
    return {
        "color_variance": color_variance,
        "contrast":       contrast,
        "lighting_key":   lighting_key,
        "edge_density":   edge_density
    }


def _extract_visual_features(video_path: str, sample_rate: int = 10) -> tuple:
    """
    Sample every `sample_rate` frames, compute and aggregate visual features.
    Returns (avg_features_dict, sampled_frames_list, dominant_lighting_key).
    """
    cap            = cv2.VideoCapture(video_path)
    sampled_frames = []
    frame_features = []
    frame_idx      = 0

    print("  [Visual] Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            feats = _compute_frame_features(frame)
            frame_features.append(feats)
            sampled_frames.append(frame)
            if frame_idx % 300 == 0 and frame_idx > 0:
                print(f"    Processed frame {frame_idx}")
        frame_idx += 1
    cap.release()
    print(f"  [Visual] Done — {len(sampled_frames)} sampled frames from {frame_idx} total.")

    # Aggregate averages
    avg_feats = {
        "color_variance": round(float(np.mean([f["color_variance"] for f in frame_features])), 4),
        "contrast":       round(float(np.mean([f["contrast"]       for f in frame_features])), 4),
        "edge_density":   round(float(np.mean([f["edge_density"]   for f in frame_features])), 4)
    }

    # Dominant lighting: majority vote across all sampled frames
    low_key_count = sum(1 for f in frame_features if f["lighting_key"] == "low-key")
    dominant_key  = "low-key" if low_key_count > len(frame_features) / 2 else "high-key"

    return avg_feats, sampled_frames, dominant_key


# ---------------------------------------------------------------------------
# CLIP semantic drift
# ---------------------------------------------------------------------------
def _compute_semantic_drift(frames: list) -> float:
    """
    Mean cosine distance between consecutive CLIP frame embeddings.
    Source: more_metadata.py → compute_semantic_drift()
    """
    model, processor = _get_clip()
    embeddings = []

    for i, frame in enumerate(frames):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=[rgb], return_tensors="pt")
        with torch.no_grad():
            output = model.get_image_features(**inputs)

        # Robust tensor unwrapping — handles all HuggingFace output variants
        if hasattr(output, "pooler_output"):
            emb = output.pooler_output
        elif hasattr(output, "image_embeds"):
            emb = output.image_embeds
        elif isinstance(output, tuple):
            emb = output[0]
        else:
            emb = output

        embeddings.append(emb[0])

    drift = []
    for i in range(1, len(embeddings)):
        sim = torch.nn.functional.cosine_similarity(
            embeddings[i - 1].unsqueeze(0),
            embeddings[i].unsqueeze(0),
            dim=1
        )
        drift.append(float(1 - sim.item()))

    return float(np.mean(drift)) if drift else 0.0


# ---------------------------------------------------------------------------
# Scene / pacing
# ---------------------------------------------------------------------------
def _compute_scene_features(video_path: str, duration_sec: float) -> dict:
    """
    Shot count, cuts-per-minute, avg scene duration.
    Source: more_metadata.py → compute_scene_features() + extract_movie_metadata.py
    """
    print("  [Scene] Running scene detection...")
    scenes    = detect(video_path, ContentDetector(threshold=27.0))
    durations = [end.get_seconds() - start.get_seconds() for start, end in scenes]
    print(f"  [Scene] {len(scenes)} scenes detected.")

    num_scenes   = len(scenes)
    shot_density = num_scenes / (duration_sec / 60) if duration_sec > 0 else 0.0
    avg_dur      = float(np.mean(durations)) if durations else 0.0

    return {
        "num_scenes":             num_scenes,
        "cuts_per_minute":        round(float(shot_density), 4),
        "avg_scene_duration_sec": round(avg_dur, 4)
    }


# ---------------------------------------------------------------------------
# Audio extraction + features
# ---------------------------------------------------------------------------
def _extract_audio(video_path: str) -> tuple:
    """
    Extract audio track to a temp WAV if video; load directly if audio file.
    Returns (y, sr, temp_path_or_None).
    """
    if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        temp_audio = "temp_global_audio.wav"
        video      = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            raise ValueError(f"No audio track found in '{video_path}'")
        video.audio.write_audiofile(temp_audio, logger=None)
        video.close()
        audio_path = temp_audio
    else:
        audio_path = video_path
        temp_audio = None

    y, sr = librosa.load(audio_path, sr=None)
    return y, sr, temp_audio


def _compute_audio_features(y: np.ndarray, sr: int) -> dict:
    """
    All audio features preserved from all three source files.

    Sources:
      audio_metadata_extractor.py → rms, zcr, centroid, bandwidth, rolloff,
                                     tempo, mfccs, 6-band sub-band energy
      extract_movie_metadata.py   → 3-band normalised % energy
      more_metadata.py            → onset_strength
    """
    # Core time-domain / spectral descriptors
    rms                = librosa.feature.rms(y=y)[0]
    zcr                = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    onset_strength     = librosa.onset.onset_strength(y=y, sr=sr)
    mfccs              = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    tempo, _           = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]

    # STFT for band energies
    S     = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # ── 6-band energy (audio_metadata_extractor.py) ──────────────────────
    six_band_masks = {
        "sub_bass_0_60hz":        freqs <= 60,
        "bass_60_250hz":          (freqs > 60)   & (freqs <= 250),
        "low_mid_250_2000hz":     (freqs > 250)  & (freqs <= 2000),
        "high_mid_2000_4000hz":   (freqs > 2000) & (freqs <= 4000),
        "presence_4000_6000hz":   (freqs > 4000) & (freqs <= 6000),
        "brilliance_6000hz_plus": freqs > 6000
    }
    total_energy    = max(float(np.sum(S ** 2)), 1e-10)
    sub_band_energy = {
        k: round(float(np.sum(S[mask] ** 2)) / total_energy, 4)
        for k, mask in six_band_masks.items()
    }

    # ── 3-band energy (extract_movie_metadata.py) ────────────────────────
    low_e  = float(np.mean(np.sum(S[freqs <= 250]                           ** 2, axis=0)))
    mid_e  = float(np.mean(np.sum(S[(freqs > 250) & (freqs <= 4000)]        ** 2, axis=0)))
    high_e = float(np.mean(np.sum(S[freqs > 4000]                           ** 2, axis=0)))
    tot3   = low_e + mid_e + high_e
    if tot3 > 0:
        three_band_energy = {
            "low_pct":  round(low_e  / tot3, 4),
            "mid_pct":  round(mid_e  / tot3, 4),
            "high_pct": round(high_e / tot3, 4)
        }
    else:
        three_band_energy = {"low_pct": 0.0, "mid_pct": 0.0, "high_pct": 0.0}

    return {
        "tempo_bpm":                   round(float(tempo), 2),
        "mean_rms_energy":             round(float(np.mean(rms)), 4),
        "mean_zero_crossing_rate":     round(float(np.mean(zcr)), 4),
        "mean_spectral_centroid_hz":   round(float(np.mean(spectral_centroid)), 2),
        "mean_spectral_bandwidth_hz":  round(float(np.mean(spectral_bandwidth)), 2),
        "mean_spectral_rolloff_hz":    round(float(np.mean(spectral_rolloff)), 2),
        "mean_onset_strength":         round(float(np.mean(onset_strength)), 4),
        "mfcc_mean_vector":            [round(float(m), 4) for m in np.mean(mfccs, axis=1)],
        "sub_band_energy":             sub_band_energy,
        "three_band_energy":           three_band_energy
    }


def _compute_stimulation_score(cuts_per_minute: float, mean_rms: float) -> dict:
    """
    Stimulation score.
    Source: extract_movie_metadata.py → calculate_stimulation()
    visual = min((cuts_per_min / 30) * 100, 100)
    audio  = min((mean_rms / 0.1)   * 100, 100)
    final  = 0.6 * visual + 0.4 * audio
    """
    v_score = min((cuts_per_minute / 30) * 100, 100)
    a_score = min((mean_rms / 0.1)       * 100, 100)
    return {
        "visual_score": round(float(v_score), 2),
        "audio_score":  round(float(a_score), 2),
        "final_score":  round(float(0.6 * v_score + 0.4 * a_score), 2)
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def extract_global_features(
    video_path: str,
    visual_sample_rate: int  = 10,
    semantic_frame_count: int = 20
) -> dict:
    """
    Extract all static/global features for a movie.

    Parameters
    ----------
    video_path            : path to video (or audio) file
    visual_sample_rate    : sample 1 frame every N frames for visual analysis
    semantic_frame_count  : number of evenly-spaced frames used for CLIP drift

    Returns
    -------
    Full feature dict (schema at top of file).
    """
    start = time.time()
    print(f"\n[GlobalFeatures] Starting: {video_path}")

    # ── Audio ────────────────────────────────────────────────────────────
    print("  [Audio] Extracting...")
    y, sr, temp_audio = _extract_audio(video_path)
    duration          = librosa.get_duration(y=y, sr=sr)
    audio_feats       = _compute_audio_features(y, sr)
    if temp_audio and os.path.exists(temp_audio):
        os.remove(temp_audio)
    print(f"  [Audio] Done — {duration:.2f}s @ {sr} Hz")

    # ── Visual ───────────────────────────────────────────────────────────
    visual_avg, sampled_frames, dominant_key = _extract_visual_features(
        video_path, sample_rate=visual_sample_rate
    )

    # ── Scene ────────────────────────────────────────────────────────────
    scene_feats = _compute_scene_features(video_path, duration)

    # ── Stimulation ──────────────────────────────────────────────────────
    stimulation = _compute_stimulation_score(
        scene_feats["cuts_per_minute"],
        audio_feats["mean_rms_energy"]
    )

    # ── Semantic drift (CLIP) ────────────────────────────────────────────
    print("  [CLIP] Computing semantic drift...")
    if len(sampled_frames) > semantic_frame_count:
        step         = len(sampled_frames) // semantic_frame_count
        drift_frames = sampled_frames[::step][:semantic_frame_count]
    else:
        drift_frames = sampled_frames
    semantic_drift = _compute_semantic_drift(drift_frames)
    print(f"  [CLIP] Semantic drift = {semantic_drift:.4f}")

    elapsed = round(time.time() - start, 2)
    print(f"[GlobalFeatures] Complete in {elapsed}s\n")

    return {
        "file":           os.path.basename(video_path),
        "duration_sec":   round(float(duration), 2),
        "sample_rate_hz": int(sr),

        # Audio
        "tempo_bpm":                  audio_feats["tempo_bpm"],
        "mean_rms_energy":            audio_feats["mean_rms_energy"],
        "mean_zero_crossing_rate":    audio_feats["mean_zero_crossing_rate"],
        "mean_spectral_centroid_hz":  audio_feats["mean_spectral_centroid_hz"],
        "mean_spectral_bandwidth_hz": audio_feats["mean_spectral_bandwidth_hz"],
        "mean_spectral_rolloff_hz":   audio_feats["mean_spectral_rolloff_hz"],
        "mean_onset_strength":        audio_feats["mean_onset_strength"],
        "mfcc_mean_vector":           audio_feats["mfcc_mean_vector"],
        "sub_band_energy":            audio_feats["sub_band_energy"],
        "three_band_energy":          audio_feats["three_band_energy"],

        # Visual
        "visual_features_avg":   visual_avg,
        "dominant_lighting_key": dominant_key,

        # Scene
        "num_scenes":             scene_feats["num_scenes"],
        "cuts_per_minute":        scene_feats["cuts_per_minute"],
        "avg_scene_duration_sec": scene_feats["avg_scene_duration_sec"],

        # Stimulation
        "stimulation": stimulation,

        # Semantic
        "semantic_drift": round(semantic_drift, 6)
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
        features = extract_global_features(test_file)
        # out_path = os.path.join(os.path.dirname(__file__), "global_features.json")
        # Make sure output directory exists
        out_dir = os.path.join(os.path.dirname(__file__), "metadata")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "global_features.json")
        with open(out_path, "w") as f:
            json.dump(features, f, indent=4)
        print(json.dumps(features, indent=4))
        print(f"\nSaved to {out_path}")
