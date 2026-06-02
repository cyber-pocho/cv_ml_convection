# Rayleigh-Bénard Convection — CV + ML Pipeline

Experimental fluid dynamics, computer vision, and physics-informed machine learning applied to Rayleigh-Bénard convection imaged from above.

---

## Table of Contents

1. [What is Rayleigh-Bénard Convection?](#1-what-is-rayleigh-bénard-convection)
2. [Experiment](#2-experiment)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Methods](#4-methods)
5. [Results — Session 1](#5-results--session-1-single-video-sparse-seeding)
6. [Results — Session 2](#6-results--session-2-batch-analysis-improved-seeding)
7. [Known Limitations and Next Steps](#7-known-limitations-and-next-steps)
8. [How to Run](#8-how-to-run)
9. [Project Structure](#9-project-structure)
10. [Technical Stack](#10-technical-stack)
11. [References](#11-references)

---

## 1. What is Rayleigh-Bénard Convection?

Rayleigh-Bénard convection (RBC) is the buoyancy-driven flow that arises when a fluid layer is heated from below and cooled from above. Past a critical temperature difference, the fluid spontaneously organizes into convection rolls — rising hot plumes and sinking cold columns — visible through reflective tracer particles.

### Governing equations (Boussinesq approximation)

Momentum:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{\nabla p}{\rho_0} + \nu \nabla^2 \mathbf{u} + \alpha g \Delta T \,\hat{z}$$

Incompressibility:

$$\nabla \cdot \mathbf{u} = 0$$

Energy:

$$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \kappa \nabla^2 T$$

### Key dimensionless numbers

$$Ra = \frac{g \,\alpha \,\Delta T \,H^3}{\nu \,\kappa}, \qquad Re = \frac{U_\mathrm{rms}\, L}{\nu}, \qquad Pr = \frac{\nu}{\kappa}, \qquad Nu = \frac{q H}{\lambda \Delta T}$$

**Onset condition:** $Ra > 1708$ — below this, conduction dominates and no rolls form.

**Theoretical inertial-range slope for 2D RBC** (Kooloth et al. 2021):

$$E(k) \sim k^{-11/5} \approx k^{-2.20}$$

---

## 2. Experiment

```
         [Camera — top view]
                  |
                  v
    +----------------------------------+  <- fluid surface (cooled by ambient air, ~25 °C)
    |                                  |
    |    glycerin 80 % + water 20 %    |  <- fluid layer (~20 mm deep)
    |    + mica powder tracer          |
    |                                  |
    |      (↻)(↺)(↻)(↺)(↻)            |  <- convection rolls visible from above
    |                                  |
    +----------------------------------+  <- hot plate (Haceb, controlled temperature)
             [thermocouple probes]
```

- **Fluid:** 80 % glycerin / 20 % water, seeded with mica powder (flat, reflective tracer)
- **Camera:** fixed top-view camera
- **Heating:** Haceb laboratory hot plate
- **Cooling:** passive ambient air ($T_\mathrm{ambient} \approx 25$ °C)
- **Container:** circular Pyrex dish

See `resources/setup.jpg` for a photograph of the experimental setup.

### Video manifest

| Video | $T_\mathrm{plate}$ (°C) | $\Delta T$ (°C) | Notes |
|-------|:-----------------------:|:---------------:|-------|
| video2.mp4 | 130 | 105 | Session 2, used |
| ~~video3.mp4~~ | ~~290~~ | ~~265~~ | **Excluded** — thermocouple reading exceeds glycerin decomposition point (~180 °C); sensor malfunction suspected |
| video4.mp4 | 120 | 95 | Session 2, used |
| video5.mp4 | 100 | 75 | Session 2, used |
| video6.mp4 | 100 | 75 | Session 2, used |
| video7.mp4 | 90 | 65 | Session 2, used |

---

## 3. Pipeline Overview

```
raw video
    │
    ▼
data/preprocessor.py     Gaussian blur · CLAHE · MOG2 background subtraction
    │
    ▼
data/detector.py          SimpleBlobDetector + contour fallback → Particle list
    │
    ▼
src/tracker.py            Hungarian matching → trajectories, (vx, vy) per particle
    │
    ▼
src/piv.py                Phase-correlation PIV → velocity field (u, v) on grid
    │
    ▼
scripts/batch_analyze.py  Vorticity · divergence · E(k) · Re · Ra · Nu_proxy
    │
    ▼
outputs/batch_TIMESTAMP/  Per-video figures + cross-video comparisons
```

---

## 4. Methods

### 4.1 Preprocessing

Each frame is processed in three steps:

1. **Gaussian blur** ($5 \times 5$ kernel) — suppresses pixel noise before contrast enhancement.
2. **CLAHE** (clip = 2.0, tile = $8 \times 8$) — local contrast equalization so mica reflections do not crush surrounding detail.
3. **MOG2 background subtraction** — isolates moving particles from the static background; warmed up on the first 50 frames so the background model is stable before particles are tracked.

### 4.2 Particle Detection

`SimpleBlobDetector` with the following filters:

| Filter | Value | Rationale |
|--------|-------|-----------|
| Area | 8–300 px² | Rejects sub-pixel speckle and particle aggregates |
| Circularity | ≥ 0.2 | Loose threshold — mica flakes are flat and irregular |
| Inertia ratio | ≥ 0.05 | Rejects streak-like noise |
| Convexity | ≥ 0.4 | Rejects deeply concave artifacts |

If fewer than 10 blobs are detected, a contour-based fallback is used; image moments provide sub-pixel centroid accuracy.

### 4.3 Particle Tracking Velocimetry (PTV)

Consecutive frames are matched with the Hungarian algorithm (minimum-cost bipartite matching). Displacements larger than 20 px are rejected as physically implausible.

$$v_x = \frac{\Delta x \cdot f_s}{\mathrm{px/mm}}, \qquad v_y = \frac{\Delta y \cdot f_s}{\mathrm{px/mm}}$$

where $f_s$ is the frame rate (fps).

### 4.4 Particle Image Velocimetry (PIV)

Phase correlation on $32 \times 32$ px interrogation windows with 50 % overlap (16 px step). `cv2.phaseCorrelate` returns sub-pixel displacement $(dx, dy)$ per window. Only frame pairs with $\geq 10$ detected particles are used.

### 4.5 Derived Physical Quantities

**Vorticity** ($z$-component, meaningful in top view):

$$\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

**Divergence** (incompressibility check — should be $\approx 0$):

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

**Kinetic energy spectrum** — azimuthal average of the 2D FFT power:

$$E(k) = \oint \frac{1}{2}\!\left(|\hat{u}|^2 + |\hat{v}|^2\right) d\theta$$

**Rayleigh number** (glycerin: $\alpha = 5 \times 10^{-4}$ K$^{-1}$, $\nu = 60 \times 10^{-6}$ m²/s, $\kappa = 10^{-7}$ m²/s, $H = 20$ mm):

$$Ra = \frac{g \,\alpha \,\Delta T \,H^3}{\nu \,\kappa}$$

**Reynolds number:**

$$Re = \frac{U_\mathrm{rms} \,L}{\nu}, \qquad U_\mathrm{rms} = \sqrt{\langle u^2 + v^2 \rangle}\ [\mathrm{m/s}]$$

**Nusselt proxy** (top-view estimate; no temperature-field measurement available):

$$Nu_\mathrm{proxy} = 1 + \frac{\bar{U}}{\kappa^*}, \qquad \kappa^* = \sqrt{\frac{16}{Ra \cdot Pr}}$$

> **Note:** $Nu_\mathrm{proxy}$ is a kinematic surrogate only. A true Nusselt number requires a vertical temperature-sensor array or schlieren imaging.

### 4.6 Physics-Informed ML (Planned)

A super-resolution CNN will enforce incompressibility via a PDE penalty:

$$\mathcal{L} = \mathcal{L}_\mathrm{pixel} + \gamma\, \mathcal{L}_\mathrm{PDE}, \qquad \gamma = 0.05$$

$$\mathcal{L}_\mathrm{PDE} = \left\langle \left| \frac{\partial \hat{u}}{\partial x} + \frac{\partial \hat{v}}{\partial y} \right| \right\rangle$$

This enforces $\nabla \cdot \hat{\mathbf{u}} \approx 0$ in the predicted field without requiring temperature measurements (adapted from Salim et al. 2024).

---

## 5. Results — Session 1 (single video, sparse seeding)

**Video:** `video/video.mp4` · $478 \times 850$ px · 29 fps · 6 679 frames · ROI: central 60 % ($510 \times 286$ px)

| Metric | Value |
|--------|-------|
| Particles/frame | 2.4 |
| $Re$ | 5.9 |
| $E(k)$ slope | +1.47 (noise-dominated; theory: −2.20) |

**Diagnosis:** at 2.4 particles/frame, the cross-correlation is essentially noise-driven, producing a spuriously positive $E(k)$ slope. The fix for Session 2 was to increase mica concentration by approximately $20\times$.

---

## 6. Results — Session 2 (batch analysis, improved seeding)

Batch analysis of five videos using `scripts/batch_analyze.py`.
Output directory: `outputs/batch_20260602_154329/`.

### Summary table

| Video | $\Delta T$ (°C) | $Ra$ | $Re$ | $Nu_\mathrm{proxy}$ | $\omega_\mathrm{RMS}$ (s$^{-1}$) | $E(k)$ slope | Particles/frame |
|-------|:--------------:|:----:|:----:|:-------------------:|:---------------------------------:|:------------:|:---------------:|
| video2 | 105 | $6.87 \times 10^5$ | 0.1 | 343.7 | 0.0230 | +0.90 | 3.2 |
| video4 | 95  | $6.21 \times 10^5$ | 0.1 | 366.5 | 0.0256 | +0.48 | 5.6 |
| video5 | 75  | $4.91 \times 10^5$ | 0.1 | 230.1 | 0.0136 | **−0.59** | 4.0 |
| video6 | 75  | $4.91 \times 10^5$ | 0.1 | 263.0 | 0.0172 | +0.19 | 7.1 |
| video7 | 65  | $4.25 \times 10^5$ | 0.1 | 239.7 | 0.0148 | +1.04 | 4.1 |

### Key observations

**Ra scales with $\Delta T$ as expected.** $Ra$ ranges from $4.25 \times 10^5$ (video7) to $6.87 \times 10^5$ (video2), consistent with $Ra \propto \Delta T$. All values are well above the onset threshold of 1 708, confirming active convection in every video.

**The flow is deeply laminar.** $Re \approx 0.1$ across all videos. Glycerin's high kinematic viscosity ($\nu \approx 600\times$ that of water) keeps the flow strongly viscosity-dominated. The inertial-range cascade predicted by Kooloth et al. ($k^{-11/5}$) only develops in the turbulent regime, so these $E(k)$ slopes are not expected to match theory.

**video5 gives the closest approach to theoretical scaling.** It is the only video with a negative $E(k)$ slope ($-0.59$), suggesting the onset of a weak inertial range. The remaining videos show positive slopes driven by sparse-seeding noise.

**Divergence residuals are small and physically consistent.** $|\nabla \cdot \mathbf{u}|_\mathrm{RMS} \approx 0.012$–$0.023$ s$^{-1}$ — non-zero but within the noise expected at 3–7 particles/frame, and small enough to confirm that the velocity fields are physically reasonable.

### Comparison with theory

**Kooloth et al. 2021** — The $k^{-11/5}$ prediction applies to 2D RBC in the turbulent regime. At $Re \sim 0.1$ the inertial cascade has not developed, so measured slopes of $-0.59$ to $+1.04$ are expected.

**Salim et al. 2024** — Their DNS spans $Ra = 10^6$–$10^{10}$ (fully turbulent). Our experimental $Ra \sim 5$–$7 \times 10^5$ sits just below this range, and the high Prandtl number of glycerin ($Pr = \nu/\kappa = 600$) further suppresses inertial effects. The physics-informed incompressibility loss ($\nabla \cdot \mathbf{u} = 0$) from that work is directly applicable here and will be adopted in the planned CNN.

### Comparison figures

**Figure 1 — $Ra$, $Re$, $Nu_\mathrm{proxy}$, and $E(k)$ slope vs. $\Delta T$:**
![Parameters vs ΔT](outputs/batch_20260602_154329/comparison/parameters_vs_deltaT.png)

**Figure 2 — Kinetic energy spectra, all five videos:**
![All spectra](outputs/batch_20260602_154329/comparison/all_spectra.png)

**Figure 3 — Vorticity fields at increasing $\Delta T$ (shared colorscale):**
![All vorticity fields](outputs/batch_20260602_154329/comparison/all_vorticity.png)

**Figure 4 — Velocity fields at increasing $\Delta T$ (shared speed scale):**
![All velocity fields](outputs/batch_20260602_154329/comparison/all_velocity_fields.png)

**Figure 5 — Summary table:**
![Summary table](outputs/batch_20260602_154329/comparison/summary_table.png)

---

## 7. Known Limitations and Next Steps

### Current limitations

| Issue | Impact | Root cause |
|-------|--------|------------|
| Sparse seeding (3–7 particles/frame; need ≥ 50) | PIV cross-correlation unreliable; $E(k)$ cannot resolve inertial range | Low mica concentration |
| Circular Pyrex dish | Optical distortion near edges (mitigated by 60 % ROI crop) | Container geometry |
| No temperature field | $Nu$ is a kinematic proxy only | No vertical sensor array or schlieren |
| Thermocouple error (video3) | One session excluded entirely | Sensor malfunction at high temperature |

### Session 3 plan

- **Container:** rectangular refractaria dish → eliminates edge distortion; enables lateral (side) view
- **Cooling:** Peltier module → controlled, stable $\Delta T$
- **Fluid:** silicone oil 50 cSt → lower $Pr$, closer to turbulent transition
- **ML:** CNN super-resolution with incompressibility PDE loss
- **Classifier:** laminar vs. turbulent regime from velocity-field snapshots

---

## 8. How to Run

```bash
git clone https://github.com/cyber-pocho/cv_ml_convection
cd cv_ml_convection
pip install -r requirements.txt

# Batch analysis — all five Session 2 videos
python scripts/batch_analyze.py

# Single-video analysis (Session 1)
python scripts/analyze_video.py --video video/video.mp4
```

---

## 9. Project Structure

```
cv_ml_convection/
├── data/
│   ├── preprocessor.py      Gaussian blur · CLAHE · MOG2 background subtraction
│   └── detector.py          SimpleBlobDetector + contour fallback · Particle dataclass
│
├── src/
│   ├── config.py            RBConfig dataclass — all tunable parameters
│   ├── tracker.py           Hungarian PTV: match_particles · compute_velocities
│   ├── piv.py               Phase-correlation PIV on interrogation windows
│   └── __init__.py
│
├── scripts/
│   ├── analyze_video.py     Single-video end-to-end pipeline (Session 1)
│   └── batch_analyze.py     Multi-video batch pipeline with comparison figures
│
├── video/                   Raw experiment videos (video2–7; video3 excluded)
├── outputs/                 Session and batch output directories
│   └── batch_TIMESTAMP/     Per-video figures · comparison/ · all_results.npy
├── resources/               Setup photos · calibration images
├── notebook/                Exploratory Jupyter notebooks
├── requirements.txt
└── CLAUDE.md                Coding style guide
```

---

## 10. Technical Stack

| Component | Tool |
|-----------|------|
| Particle detection | OpenCV `SimpleBlobDetector` + contour analysis |
| Particle tracking | Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) |
| PIV velocity field | Phase correlation (`cv2.phaseCorrelate`) |
| Physical analysis | NumPy · SciPy |
| Visualization | Matplotlib + SciencePlots (`science`, `no-latex`) |
| Super-resolution CNN | PyTorch · U-Net · PDE loss *(planned)* |
| Regime classifier | PyTorch *(planned)* |
| Language | Python 3.14 |

---

## 11. References

Kooloth, P., Sondak, D., & Smith, L. M. (2021).
Coherent solutions and transition to turbulence in two-dimensional Rayleigh-Bénard convection.
*Physical Review Fluids*, 6(1), 013501.

Salim, D. M., Burkhart, B., & Sondak, D. (2024).
Extending a physics-informed machine learning network for superresolution studies of Rayleigh-Bénard convection.
*arXiv:2307.02674.*

---

*Portfolio project combining experimental physics, classical computer vision, and physics-informed deep learning.*
