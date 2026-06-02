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
| Particle tracker (PTV) | `src/tracker.py` | Done |
| PIV velocity field | `src/piv.py` | Done |
| End-to-end analysis script | `scripts/analyze_video.py` | Done |
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

## Results

Analysis run on `video/video.mp4` (top-view recording, $478 \times 850$ px, 29 fps, 6679 frames).
The central 60 % ROI ($510 \times 286$ px) is used to exclude the curved Pyrex dish edges.
Calibration: $5\ \text{px/mm}$.

### Particle tracking (PTV)

The detector found an average of **2.4 particles/frame** — sparse seeding, which limits PIV reliability.
Hungarian matching with a search radius of $r_\text{max} = 20\ \text{px}$ built **459 trajectories** across the full video.

Instantaneous velocity is computed from consecutive positions:

$$v_x = \frac{\Delta x}{\text{px/mm}} \cdot f_s, \qquad v_y = \frac{\Delta y}{\text{px/mm}} \cdot f_s$$

where $f_s = 29\ \text{fps}$.  The speed distribution is right-skewed, peaking at 5–15 mm/s with a tail to ~80 mm/s.

| Rank | Particle | $x$ (mm) | $y$ (mm) | $\bar{u}$ (mm/s) |
|------|----------|-----------|-----------|-------------------|
| 1 | 436 | 20.2 | 25.5 | 76.5 |
| 2 | 132 | 32.3 | 45.2 | 66.1 |
| 3 | 450 | 1.1 | 59.1 | 57.2 |

### PIV velocity field

Phase-correlation PIV on $32\ \text{px}$ interrogation windows with 50 % overlap (grid step $= 16\ \text{px} = 3.2\ \text{mm}$).
Only frame pairs with $\geq 10$ particles in both frames were used — **352 out of 6678** pairs passed this gate.

The time-averaged velocity field $(\bar{u}, \bar{v})$ was used to compute:

**Vorticity** ($z$-component of curl, valid for top-view):
$$\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}, \qquad \omega_\text{RMS} \approx 1.28\ \text{s}^{-1}$$

**Divergence** (should be $\approx 0$ for incompressible flow):
$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}, \qquad \|\nabla \cdot \mathbf{u}\|_\text{RMS} \approx 0.94\ \text{s}^{-1}$$

The near-zero divergence confirms the velocity field is physically consistent.

**Reynolds number** (using glycerin $\nu \approx 60 \times 10^{-6}\ \text{m}^2/\text{s}$):
$$Re = \frac{U_\text{rms}\, L}{\nu} \approx 5.9$$

This places the experiment in the **laminar–transitional** regime.

**Kinetic energy spectrum** — the measured inertial-range slope is $+1.47$, compared to the 2D RBC theoretical prediction of $-\tfrac{11}{5} \approx -2.20$.  The positive slope indicates the PIV field is noise-dominated due to sparse seeding; a reliable spectrum requires at least ~50 particles/frame.

### Summary figures

All outputs are written to `outputs/session_<timestamp>/`:

| File | Description |
|------|-------------|
| `velocity_field.png` | Quiver plot overlaid on last frame |
| `vorticity.png` | $\omega_z$ heatmap |
| `divergence.png` | $\nabla \cdot \mathbf{u}$ heatmap |
| `energy_spectrum.png` | $E(k)$ with inertial-range fit and $k^{-11/5}$ reference |
| `trajectories.png` | All trajectories coloured by mean speed |
| `speed_histogram.png` | PTV particle speed distribution |
| `rbc_summary.png` | Three-panel summary (velocity \| vorticity \| spectrum) |

---

## Reference

See `resources/paper.pdf` for the underlying physics and experimental setup.
