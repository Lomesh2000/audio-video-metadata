# RS exp - Video Feature Extraction Pipeline

This repository contains a comprehensive pipeline for extracting multimodal features from video content. It is designed to generate rich metadata for Recommendation Systems (RecSys), specifically for creating item embeddings and training sequence-aware models.

The pipeline is divided into two main components: **Global Features** (static summaries) and **Temporal Features** (time-series data).

## 📂 Project Structure

```
.
├── gloabal_features/           # Static, per-movie feature extraction
│   ├── global_features.py      # Main script for global summaries
│   └── metadata/               # Output JSONs for global features
├── temporal_features/          # Sequential, per-frame/second extraction
│   ├── temporal_features.py    # Main script for time-series data
│   └── metadata/               # Output JSONs for temporal features
├── metadata/                   # Legacy/Intermediate metadata storage
└── scripts/                    # Utility scripts (consolidated into main modules)
```

## 🚀 Key Modules

### 1. Global Features
**Script**: `gloabal_features/global_features.py`

Extracts a single static vector representing the entire video. Ideal for lightweight item embeddings.

#### Audio Analysis
- **Core Metrics**: BPM (Tempo), Mean RMS Energy, Zero Crossing Rate.
- **Spectral Features**: Spectral Centroid, Bandwidth, Rolloff, MFCCs (Mean Vector).
- **Sub-band Energy**: 6-band analysis (Sub-bass, Bass, Low-mid, High-mid, Presence, Brilliance).
- **3-Band Ratio**: Normalized energy distribution (Low/Mid/High).

#### Visual & Cinematic Analysis
- **Frame Stats**: Color variance, Contrast, Edge Density (Canny).
- **Lighting**: Dominant lighting key classification (Low-key vs. High-key).
- **Editing**: Shot density, Cuts per minute, Average scene duration (using `PySceneDetect`).
- **Semantic**: **CLIP**-based semantic drift (measuring visual consistency over time).

#### Computed Metrics
- **Stimulation Score**: A composite score (0-100) combining visual pacing and audio intensity.

---

### 2. Temporal Features
**Script**: `temporal_features/temporal_features.py`

Extracts sequential data aligned with video frames or time intervals. Ideal for LSTMs, Transformers, and attention-based models.

#### Per-Frame Features (FPS Aligned)
- **Visual**:
  - Motion Energy (Optical Flow magnitude).
  - Brightness Shift (Delta between frames).
- **Audio**:
  - Intensity & Hike % (RMS).
  - Brightness & Hike % (Spectral Centroid).
  - Transition Flux (Onset strength).
  - Band-wise energy (Low/Mid/High).

#### Per-Second Features (Windowed)
- Aggregated stats (RMS, Flux, Centroid) per second.
- Delta percentage changes to track dynamics over longer windows.

## 🛠 Dependencies

- `numpy`
- `librosa` (Audio analysis)
- `opencv-python` (Visual analysis)
- `moviepy` (Video processing)
- `scenedetect` (Shot detection)
- `transformers`, `torch` (CLIP model)

## 📊 Output Schema

Data is saved as JSON files in the respective `metadata/` directories.

**Global Output Example:**
```json
{
  "file": "movie.mp4",
  "tempo_bpm": 120.5,
  "stimulation": { "final_score": 75.4 },
  "visual_features_avg": { "contrast": 0.45, "edge_density": 0.12 },
  "sub_band_energy": { ... }
}
```

**Temporal Output Example:**
```json
{
  "per_frame": [
    { "frame_idx": 0, "motion_energy": 0.5, "intensity_rms": 0.1 },
    ...
  ]
}
```