# Rayleigh-Bénard Convection — CV + ML Pipeline

Computer vision pipeline for studying **Rayleigh-Bénard convection** (RBC) by tracking mica particles in a convection cell. The long-term goal is a CNN-Transformer model that can infer flow fields from raw video.

---

## What it does

A fluid heated from below develops convection rolls. This project tracks reflective mica particles seeded into the fluid to measure that flow. The pipeline goes:

```
raw video → preprocessing → particle detection → tracking → (PIV / ML model)
```

## Current state

| Module | File | Status |
|---|---|---|
| Config & preprocessing | `data/preprocessor.py` | Done |
| Particle detector | `data/detector.py` | Done |
| Particle tracker | — | Planned |
| PIV velocity field | — | Planned |
| CNN-Transformer model | — | Planned |

## Pipeline overview

### 1. Preprocessing (`FramePreprocessor`)

Each raw frame goes through:
1. Grayscale conversion
2. Gaussian blur — reduces sensor noise
3. CLAHE — adaptive contrast enhancement for uneven illumination
4. MOG2 background subtraction — isolates moving particles from the static background
5. Morphological close + open — fills holes and removes speckle in the mask

Returns `(enhanced_image, binary_foreground_mask)`.

The background model needs a warm-up pass (~50 frames) before it's reliable. Call `preprocessor.warm_up(opening_frames)` before the main loop.

### 2. Particle detection (`MicaDetector`)

Detects mica particles (flat, reflective, non-circular) in the preprocessed frame.

Primary method: OpenCV `SimpleBlobDetector` with filters on area, brightness, circularity, inertia, and convexity.

Fallback: if fewer than `min_particles_per_frame` blobs are found, switches to contour analysis with centroid moments for sub-pixel accuracy.

Returns a list of `Particle(x, y, area, frame_idx)`.

### 3. Configuration (`RBConfig`)

All physical and vision parameters live in one dataclass:

```python
@dataclass
class RBConfig:
    px_per_mm: float = 10.0      # camera calibration
    fps: float = 30.0
    gaussian_ksize: int = 5
    blob_min_area: float = 8.0
    blob_max_area: float = 300.0
    piv_window_size: int = 32    # for future PIV
    ...
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

- `opencv-python` + `opencv-contrib-python` — blob detection, background subtraction, morphology
- `numpy`, `scipy` — numerical ops
- `scikit-image` — supplementary image processing
- `matplotlib`, `seaborn`, `plotly` — visualization

## Reference

See `resources/paper.pdf` for the underlying physics and experimental setup.
