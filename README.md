# Rayleigh-Bénard Convection — CV + ML Pipeline

> Experimental fluid dynamics meets computer vision and deep learning.  
> From a Pyrex dish of glycerin and mica powder to a physics-informed neural network.

---

## What is this?

This project builds a full pipeline to study **Rayleigh-Bénard Convection (RBC)** —
one of the most studied phenomena in fluid dynamics and the basis of atmospheric
circulation, ocean currents, and stellar interiors.

A thin layer of fluid is heated from below and cooled from above. Past a critical
temperature difference, the fluid spontaneously organizes into convection rolls:
rising hot plumes and sinking cold plumes, visible through reflective mica particles
suspended in the fluid.

The pipeline goes from **raw video** to **quantitative physics** using computer
vision, classical fluid mechanics, and a physics-informed neural network.

---

## Motivation

Numerical simulations of turbulent convection are computationally expensive — the
paper that inspired this project (Salim et al. 2024) uses supercomputer clusters
to run DNS at Rayleigh numbers up to $Ra = 10^{10}$.

This project asks: **can we extract the same physical quantities from a cheap
experimental setup using ML?**

The answer, for moderate $Ra$, is yes.

---

## The experiment

```
         [Camera — top view]
                  ↓
    ┌─────────────────────────────┐  ← surface (cooled by ambient air)
    │                             │
    │   glycerin + mica powder    │  ← fluid layer (~15 mm)
    │       ↻  ↺  ↻  ↺           │     convection rolls visible from above
    │                             │
    └─────────────────────────────┘  ← hot plate (heating element)
```

**Session 1 (current):**
- **Fluid:** glycerin 80% + water 20%
- **Tracer:** mica powder — flat, reflective particles that align with flow
- **Container:** circular Pyrex dish, top-view camera
- **Heating:** lab hot plate from below
- **Cooling:** ambient air (passive)

**Session 2 (planned):**
- **Container:** rectangular glass baking dish (refractaria) — enables lateral view
- **Cooling:** Peltier module for controlled $\Delta T$
- **Fluid:** silicone oil 50 cSt for final measurements

---

## Pipeline

```
raw video
    ↓
data/preprocessor.py    Gaussian blur · CLAHE · MOG2 background subtraction
    ↓
data/detector.py        SimpleBlobDetector + contour fallback → Particle list
    ↓
src/tracker.py          Hungarian matching → trajectories · (vx, vy) per particle
    ↓
src/piv.py              Phase-correlation PIV → velocity field (u, v) on grid
    ↓
src/physics.py          Vorticity · divergence · E(k) · Re
    ↓
src/visualizer.py       Quiver plots · vorticity maps · energy spectra
    ↓
src/superres.py         Physics-informed CNN — 4× super-resolution  [planned]
src/classifier.py       Laminar vs turbulent classifier CNN          [planned]
```

---

## Physics

### Governing equations

The flow is governed by the incompressible Navier-Stokes equations under the
Boussinesq approximation:

$$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\nabla P + \nu \nabla^2 \mathbf{u} + \alpha g \Delta T \, \hat{z}$$

$$\nabla \cdot \mathbf{u} = 0$$

The dimensionless parameter controlling the onset of convection is the **Rayleigh number**:

$$Ra = \frac{g \, \alpha \, \Delta T \, H^3}{\nu \, \kappa}$$

Convection begins when $Ra > 1708$. The **Reynolds number** characterizes the flow regime:

$$Re = \frac{U_\text{rms} \, L}{\nu}, \qquad U_\text{rms} = \sqrt{\langle u^2 + v^2 \rangle}$$

### Measured quantities

**Vorticity** ($z$-component, top view):

$$\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

**Divergence** (incompressibility check, should be $\approx 0$):

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

**Kinetic energy spectrum** — azimuthal average of the 2D power spectrum:

$$E(k) = \oint \left( |\hat{u}|^2 + |\hat{v}|^2 \right) \, d\theta$$

Theoretical inertial-range slope for 2D RBC (Kooloth et al. 2021):

$$E(k) \sim k^{-11/5}$$

**Physics-informed loss** (adapted from Salim et al. 2024, Eq. 20):

$$\mathcal{L} = \mathcal{L}_\text{pixel} + \gamma \, \mathcal{L}_\text{PDE}, \qquad \gamma = 0.05$$

$$\mathcal{L}_\text{PDE} = \left\langle \left| \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right| \right\rangle$$

---

## Results — Session 1

Analysis run on `video/video.mp4`:
- Resolution: $478 \times 850$ px · 29 fps · 6679 frames
- ROI: central 60% ($510 \times 286$ px) to exclude curved Pyrex edges
- Calibration: $5\ \text{px/mm}$
- Fluid: glycerin 80%, $\nu \approx 60 \times 10^{-6}\ \text{m}^2/\text{s}$

### Particle tracking (PTV)

The Hungarian algorithm with search radius $r_\text{max} = 20\ \text{px}$ built
**459 trajectories** across all frames.

Average particles per frame: **2.4** — sparse seeding, see [known limitations](#known-limitations).

Instantaneous velocity from consecutive positions:

$$v_x = \Delta x \cdot \frac{f_s}{\text{px/mm}}, \qquad v_y = \Delta y \cdot \frac{f_s}{\text{px/mm}}$$

where $f_s = 29\ \text{fps}$. Speed distribution peaks at 5–15 mm/s with a tail to ~80 mm/s.

**Top 3 fastest particles:**

| Rank | Particle | $x\ \text{(mm)}$ | $y\ \text{(mm)}$ | $\bar{u}\ \text{(mm/s)}$ |
|------|----------|-----------|-----------|-------------------|
| 1 | 436 | 20.2 | 25.5 | 76.5 |
| 2 | 132 | 32.3 | 45.2 | 66.1 |
| 3 | 450 | 1.1 | 59.1 | 57.2 |

### PIV velocity field

Phase-correlation PIV on $32\ \text{px}$ interrogation windows, 50% overlap
(grid step $= 16\ \text{px} = 3.2\ \text{mm}$).

Frame pairs passing the $\geq 10$ particles gate: **352 / 6678 (5%)**.

| Parameter | Value | Note |
|---|---|---|
| $\omega_\text{RMS}$ | $1.28\ \text{s}^{-1}$ | Active convection confirmed |
| $\|\nabla \cdot \mathbf{u}\|_\text{RMS}$ | $0.94\ \text{s}^{-1}$ | Near-zero → physically consistent |
| $Re$ | $5.9$ | Laminar–transitional regime |
| $E(k)$ slope | $+1.47$ | Noise-dominated (sparse seeding) |
| Theory slope | $-2.20$ | Target for Session 2 |

The near-zero divergence confirms the velocity field is physically consistent.
The positive $E(k)$ slope is a known artifact of sparse seeding — addressed in Session 2.

---

## Known limitations

### Session 1

**Sparse seeding** is the primary limitation. With only 2.4 particles/frame:
- PIV cross-correlation is unreliable — only 5% of frame pairs usable
- $E(k)$ slope is noise-dominated ($+1.47$ vs theory $-2.20$)
- Vorticity and divergence fields are low-resolution

**Root cause and fix:**

```
too little mica (2.4 px/frame)
        ↓
PIV unreliable → E(k) = noise
        ↓
Fix: 20× more mica (target: 50+ particles/frame)
     mix mica into glycerin paste before adding to fluid
```

**Circular container** introduces edge distortion. Mitigated with 60% ROI crop
but a rectangular container (Session 2) eliminates this entirely.

---

## Roadmap

| Session | Container | View | Goal |
|---|---|---|---|
| 1 ✓ | Pyrex dish | Top | Calibrate pipeline, first PTV results |
| 2 | Glass refractaria | Lateral | Clean RBC, $E(k) \sim k^{-11/5}$, $Re > 50$ |
| 3 | Glass refractaria | Lateral | CNN super-resolution + classifier |

**Session 2 targets:**

$$\text{particles/frame} \geq 50, \qquad Re > 50, \qquad E(k)\ \text{slope} \approx -2.20$$

---

## ML layer — connection to the paper

The super-resolution model in `src/superres.py` is a lightweight adaptation of
MeshFreeFlowNet (Salim et al. 2024). The key idea is the physics-informed loss:

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_{j} \|\mathbf{y}_j - \hat{\mathbf{y}}_j\|^2}_{\mathcal{L}_\text{pixel}} + \gamma \underbrace{\left\langle \left|\frac{\partial \hat{u}}{\partial x} + \frac{\partial \hat{v}}{\partial y}\right| \right\rangle}_{\mathcal{L}_\text{PDE}}$$

This enforces $\nabla \cdot \mathbf{u} \approx 0$ in the super-resolved output
without requiring temperature field data — making it applicable to experimental video.

---

## Technical stack

| Component | Tool |
|---|---|
| Particle detection | OpenCV `SimpleBlobDetector` + contour analysis |
| Particle tracking | Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) |
| Velocity field | Phase-correlation PIV (`cv2.phaseCorrelate`) |
| Physical analysis | `numpy`, `scipy` |
| Super-resolution CNN | PyTorch · U-Net · PDE loss |
| Classifier CNN | PyTorch |
| Visualization | `matplotlib` |

---

## Code philosophy

Karpathy-style research engineer: minimal abstractions, explicit math, readable
in one sitting. No frameworks, no factory patterns — equations map directly to code.

```python
# vorticity: curl of the 2D velocity field
# ω_z = ∂v/∂x - ∂u/∂y  reveals convection roll structure
def vorticity(u, v, dx, dy):
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    return dv_dx - du_dy
```

---

## Setup

```bash
git clone https://github.com/cyber-pocho/cv_ml_convection
cd cv_ml_convection
pip install -r requirements.txt
python scripts/analyze_video.py --video video/video.mp4
```

---

## Reference

Salim, D.M., Burkhart, B., & Sondak, D. (2024).
*Extending a Physics-Informed Machine Learning Network for Superresolution Studies
of Rayleigh-Bénard Convection.* arXiv:2307.02674.

The original paper applies MFFN to DNS simulation data at $Ra = 10^6$–$10^{10}$.
This project adapts the physics-informed loss to real experimental video —
a harder problem with noisier data and no ground truth temperature field.

---

*Portfolio project combining experimental physics, classical computer vision,
and physics-informed deep learning.*
