#!/usr/bin/env python3
"""
analyze_video.py — end-to-end RBC particle analysis (PTV + PIV)

Runs like a research notebook: sequential, self-documenting, no classes.
Each step is wrapped in try/except so the script always reaches the end
and saves whatever it managed to compute.

Usage:
    python scripts/analyze_video.py --video video/video.mp4
"""

import sys
import os

# allow imports from project root regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # write to file; no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

from src.config import RBConfig, CameraView
from src.piv import compute_piv
from src.tracker import match_particles, update_trajectories, compute_velocities
from data.preprocessor import FramePreprocessor
from data.detector import MicaDetector, Particle


# ── Physics helpers ────────────────────────────────────────────────────────────
# These are pure math functions that mirror the physics equations directly.
# Defined at the top so the step blocks below stay lean and readable.

def vorticity(u, v, dx, dy):
    # omega_z = dv/dx - du/dy  (z-component of curl; only meaningful in top view)
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    return dvdx - dudy


def divergence(u, v, dx, dy):
    # div = du/dx + dv/dy; should be ≈ 0 for incompressible flow
    dudx = np.gradient(u, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    div = dudx + dvdy
    return div, float(np.sqrt(np.mean(div ** 2)))


def kinetic_energy_spectrum(u, v):
    # Radially-binned 2D FFT energy: collapses the 2D spectrum to E(k)
    # for isotropic turbulence analysis without needing k_x, k_y separately
    ny, nx = u.shape
    U = np.fft.fft2(u)
    V = np.fft.fft2(v)
    # energy density per grid point (normalise by N so amplitude is grid-size-independent)
    E2d = 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2) / (nx * ny) ** 2

    kx = np.fft.fftfreq(nx) * nx  # wavenumber in units of fundamental mode
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_max = int(min(nx, ny) // 2)
    k_bins = np.arange(1, k_max)
    E_k = np.zeros(len(k_bins))
    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        E_k[i] = E2d[mask].sum()

    # drop bins with zero energy (empty annuli at high k due to grid geometry)
    valid = E_k > 0
    return k_bins[valid].astype(float), E_k[valid]


def fit_inertial_range(k, E_k):
    # log-log slope over the middle half of resolved scales
    # (avoids the injection scale at low k and dissipation roll-off at high k)
    n = len(k)
    if n < 4:
        return 0.0, 0.0
    log_k = np.log(k)
    log_E = np.log(E_k)
    i0, i1 = max(1, n // 4), max(2, 3 * n // 4)
    slope, intercept = np.polyfit(log_k[i0:i1], log_E[i0:i1], 1)
    return float(slope), float(intercept)


def reynolds_estimate(u, v, L_mm, nu):
    # Re = U_rms * L / nu;  u, v in mm/s → convert to SI (m/s, m)
    U_rms = float(np.sqrt(np.mean(u ** 2 + v ** 2))) * 1e-3
    L = L_mm * 1e-3
    return U_rms * L / nu


# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="RBC video analysis — PTV + PIV")
parser.add_argument("--video", required=True, help="path to input video file")
args = parser.parse_args()


# ── Step 0: Setup ──────────────────────────────────────────────────────────────
try:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"session_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RBConfig(
        camera_view=CameraView.TOP,
        px_per_mm=5.0,    # rough estimate — calibrate with a known object in frame
        fps=30.0,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # prefer the actual header fps over our default; may differ (e.g. slow-motion clips)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 0:
        cfg.fps = actual_fps
        cfg.dt = 1.0 / cfg.fps

    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video:  {args.video}")
    print(f"        {frame_w}x{frame_h} px  |  {actual_fps:.1f} fps  |  {n_frames_total} frames")
    print(f"Config: px_per_mm={cfg.px_per_mm}  roi_fraction={cfg.roi_fraction}  view={cfg.camera_view.value}")
    print(f"Output: {out_dir}/")

except Exception as e:
    print(f"WARNING: Step 0 failed — {e}")
    sys.exit(1)


# ── Step 1: Load frames and warm up ───────────────────────────────────────────
try:
    preprocessor = FramePreprocessor(cfg)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        # crop to central ROI to exclude curved pyrex dish edges
        y0 = int(h * (1 - cfg.roi_fraction) / 2)
        x0 = int(w * (1 - cfg.roi_fraction) / 2)
        frame = frame[y0 : y0 + int(h * cfg.roi_fraction),
                      x0 : x0 + int(w * cfg.roi_fraction)]
        frames.append(frame)

    cap.release()

    # prime MOG2 before experiment frames so particle-bearing frames aren't swallowed into background
    preprocessor.warm_up(frames[: cfg.bg_history])

    roi_h, roi_w = frames[0].shape[:2]
    print(f"Loaded {len(frames)} frames, ROI shape: {roi_h} x {roi_w}")

except Exception as e:
    print(f"WARNING: Step 1 failed — {e}")
    sys.exit(1)


# ── Step 2: Preprocess all frames ─────────────────────────────────────────────
try:
    enhanced_frames = []
    masks = []

    for frame in frames:
        enhanced, mask = preprocessor.process(frame)
        enhanced_frames.append(enhanced)
        masks.append(mask)

    print(f"Preprocessed {len(enhanced_frames)} frames")

except Exception as e:
    print(f"WARNING: Step 2 failed — {e}")
    sys.exit(1)


# ── Step 3: PTV — detect and track particles ───────────────────────────────────
# Defaults that downstream steps can safely reference even if this step fails
trajectories = {}
all_particles = [[] for _ in frames]
velocity_records = []
n_trajectories = 0

try:
    detector = MicaDetector(cfg)
    all_particles = []

    for i, (enhanced, mask) in enumerate(zip(enhanced_frames, masks)):
        particles = detector.detect(enhanced, mask, frame_idx=i)
        all_particles.append(particles)

    mean_n = np.mean([len(p) for p in all_particles])
    print(f"Detected avg {mean_n:.1f} particles/frame")

    # seed trajectories from frame 0 — all particles are unmatched → each births a new trajectory
    trajectories = update_trajectories({}, [], all_particles[0], frame_idx=0)

    for i in range(len(all_particles) - 1):
        nxt = all_particles[i + 1]
        active_ids = sorted(trajectories.keys())

        # project each active trajectory's last known position as a virtual particle
        virtual_prev = [
            Particle(x=trajectories[tid][-1][0],
                     y=trajectories[tid][-1][1],
                     area=0.0,
                     frame_idx=i)
            for tid in active_ids
        ]

        raw_matches = match_particles(virtual_prev, nxt, cfg)
        # translate (idx_a, idx_b) → (traj_id, idx_b) as update_trajectories expects
        matches = [(active_ids[ia], ib) for ia, ib in raw_matches]
        trajectories = update_trajectories(trajectories, matches, nxt, frame_idx=i + 1)

    velocity_records = compute_velocities(trajectories, cfg)
    n_trajectories = len(trajectories)

    np.save(out_dir / "trajectories.npy", trajectories, allow_pickle=True)
    print(f"Built {n_trajectories} trajectories")

except Exception as e:
    print(f"WARNING: Step 3 failed — {e}")


# ── Step 4: PIV — velocity field ───────────────────────────────────────────────
u_stack = []
v_stack = []
piv_x = piv_y = None
piv_available = False

try:
    for i in range(len(enhanced_frames) - 1):
        n1 = len(all_particles[i])
        n2 = len(all_particles[i + 1])
        # PIV cross-correlation degrades below a minimum seeding density
        if n1 < cfg.min_particles_per_frame or n2 < cfg.min_particles_per_frame:
            continue

        result = compute_piv(enhanced_frames[i], enhanced_frames[i + 1], cfg)
        u_stack.append(result["u"])
        v_stack.append(result["v"])
        piv_x = result["x"]
        piv_y = result["y"]

    if not u_stack:
        print("WARNING: Step 4 — no valid PIV frame pairs (seeding too sparse), skipping Steps 5-6")
    else:
        u_stack = np.stack(u_stack, axis=0)   # (N_pairs, H_grid, W_grid)
        v_stack = np.stack(v_stack, axis=0)
        piv_available = True

        np.save(out_dir / "piv_fields.npy",
                {"u": u_stack, "v": v_stack, "x": piv_x, "y": piv_y},
                allow_pickle=True)

        n_pairs, H_grid, W_grid = u_stack.shape
        print(f"PIV computed for {n_pairs} frame pairs, grid: {H_grid} x {W_grid}")

except Exception as e:
    print(f"WARNING: Step 4 failed — {e}")


# ── Step 5: Physics ────────────────────────────────────────────────────────────
omega = div = div_rms = k_spec = E_k = slope = intercept = Re = None
omega_rms = 0.0
u_mean = v_mean = None

if piv_available:
    try:
        u_mean = u_stack.mean(axis=0)
        v_mean = v_stack.mean(axis=0)

        # spacing between adjacent PIV grid points in mm
        dx = float(piv_x[1] - piv_x[0]) if len(piv_x) > 1 else 1.0 / cfg.px_per_mm
        dy = float(piv_y[1] - piv_y[0]) if len(piv_y) > 1 else 1.0 / cfg.px_per_mm

        omega = vorticity(u_mean, v_mean, dx, dy)
        div, div_rms = divergence(u_mean, v_mean, dx, dy)
        k_spec, E_k = kinetic_energy_spectrum(u_mean, v_mean)
        slope, intercept = fit_inertial_range(k_spec, E_k)
        # nu = 60e-6 m²/s for ~80% glycerin at ~25 °C — measure fluid temperature and adjust
        Re = reynolds_estimate(u_mean, v_mean, cfg.container_width_mm, nu=60e-6)
        omega_rms = float(np.sqrt(np.mean(omega ** 2)))

        print("=== Physical Parameters ===")
        print(f"Reynolds number:       {Re:.1f}")
        print(f"Divergence RMS:        {div_rms:.6f}  (≈ 0 = incompressible)")
        print(f"Inertial range slope:  {slope:.3f}    (theory 2D RBC: -2.20)")
        print(f"Vorticity RMS:         {omega_rms:.4f}")

    except Exception as e:
        print(f"WARNING: Step 5 failed — {e}")


# ── Step 6: Save figures ───────────────────────────────────────────────────────
try:
    last_frame = enhanced_frames[-1] if enhanced_frames else None

    def _quiver_gradient(ax, XX, YY, U, V, cmap_name="cool"):
        """Quiver arrows coloured by local speed magnitude."""
        speed_2d = np.sqrt(U ** 2 + V ** 2)
        speed_max = max(np.percentile(speed_2d, 95), 1e-3)
        norm_q = Normalize(vmin=0, vmax=speed_max)
        q = ax.quiver(XX, YY, U, -V, speed_2d,
                      cmap=cmap_name, norm=norm_q,
                      scale=speed_max * 20, width=0.003, alpha=0.9)
        return q, speed_max

    def _spectrum_gradient(ax, k, E, cmap_name="plasma"):
        """Draw the E(k) curve as a gradient line (low-k → high-k)."""
        lk, lE = np.log10(k), np.log10(E)
        pts = np.array([lk, lE]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0, 1, len(segs))
        lc = LineCollection(segs, array=t, cmap=cmap_name, lw=2)
        ax.add_collection(lc)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(k.min(), k.max()); ax.set_ylim(E.min() * 0.5, E.max() * 2)
        return lc

    # --- velocity field: quiver overlaid on last enhanced frame
    if piv_available and last_frame is not None and u_mean is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(last_frame, cmap="gray", origin="upper")
        X_px = piv_x * cfg.px_per_mm
        Y_px = piv_y * cfg.px_per_mm
        XX, YY = np.meshgrid(X_px, Y_px)
        q, smax = _quiver_gradient(ax, XX, YY, u_mean, v_mean)
        plt.colorbar(q, ax=ax, label="speed  (mm/s)")
        ax.set_title("Time-averaged velocity field (PIV)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "velocity_field.png", dpi=150)
        plt.close(fig)

    # --- vorticity map
    if omega is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        vmax = np.percentile(np.abs(omega), 98)
        im = ax.imshow(omega, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        plt.colorbar(im, ax=ax, label="ω  [s⁻¹]")
        ax.set_title("Vorticity  ω = ∂v/∂x − ∂u/∂y")
        fig.tight_layout()
        fig.savefig(out_dir / "vorticity.png", dpi=150)
        plt.close(fig)

    # --- kinetic energy spectrum: gradient line from low-k (purple) to high-k (yellow)
    if k_spec is not None and E_k is not None:
        k_mid = k_spec[len(k_spec) // 2]
        E_mid = E_k[len(E_k) // 2]
        ref = E_mid / k_mid ** (-11 / 5) * k_spec ** (-11 / 5)

        fig, ax = plt.subplots(figsize=(6, 5))
        lc = _spectrum_gradient(ax, k_spec, E_k, "plasma")
        sm_spec = ScalarMappable(cmap="plasma", norm=Normalize(0, 1))
        sm_spec.set_array([])
        plt.colorbar(sm_spec, ax=ax, label="low k → high k")
        ax.loglog(k_spec, np.exp(intercept) * k_spec ** slope, "--",
                  color="tomato", lw=1.2, label=f"fit  $k^{{{slope:.2f}}}$")
        ax.loglog(k_spec, ref, "k:", lw=1.0, label=r"$k^{-11/5}$ ref")
        ax.set_xlabel("wavenumber $k$")
        ax.set_ylabel("$E(k)$")
        ax.set_title("Kinetic energy spectrum")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "energy_spectrum.png", dpi=150)
        plt.close(fig)

    # --- particle trajectories: each segment coloured by instantaneous speed
    if last_frame is not None and velocity_records:
        speed_by_id = defaultdict(list)
        for rec in velocity_records:
            speed_by_id[rec["particle_id"]].append(
                np.sqrt(rec["vx"] ** 2 + rec["vy"] ** 2))
        mean_speed = {pid: float(np.mean(s)) for pid, s in speed_by_id.items()}

        all_speeds = list(mean_speed.values())
        traj_norm = Normalize(vmin=np.percentile(all_speeds, 5),
                              vmax=np.percentile(all_speeds, 95))
        traj_cmap = plt.cm.plasma

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(last_frame, cmap="gray", origin="upper")
        for tid, positions in trajectories.items():
            if len(positions) < 2:
                continue
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])
            # colour each segment by instantaneous speed along the trajectory
            spd = speed_by_id.get(tid, [mean_speed.get(tid, 0.0)])
            seg_speeds = np.interp(
                np.linspace(0, 1, len(xs) - 1),
                np.linspace(0, 1, max(len(spd), 1)),
                spd[:len(xs) - 1] if len(spd) >= len(xs) - 1 else spd,
            )
            pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc_t = LineCollection(segs, array=seg_speeds, cmap="plasma",
                                  norm=traj_norm, lw=0.8, alpha=0.75)
            ax.add_collection(lc_t)

        sm = ScalarMappable(norm=traj_norm, cmap=traj_cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="speed  (mm/s)")
        ax.set_title("Particle trajectories  (colour = instantaneous speed)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "trajectories.png", dpi=150)
        plt.close(fig)

    # --- divergence heatmap
    if div is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        dmax = np.percentile(np.abs(div), 98)
        im = ax.imshow(div, cmap="PiYG", vmin=-dmax, vmax=dmax, origin="upper")
        plt.colorbar(im, ax=ax, label="∇·u  [s⁻¹]")
        ax.set_title(f"Divergence  (RMS = {div_rms:.4f} s⁻¹)")
        fig.tight_layout()
        fig.savefig(out_dir / "divergence.png", dpi=150)
        plt.close(fig)

    # --- summary panel: quiver | vorticity | spectrum
    if piv_available and omega is not None and k_spec is not None and u_mean is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(last_frame, cmap="gray", origin="upper")
        X_px = piv_x * cfg.px_per_mm
        Y_px = piv_y * cfg.px_per_mm
        XX, YY = np.meshgrid(X_px, Y_px)
        q0, _ = _quiver_gradient(axes[0], XX, YY, u_mean, v_mean)
        fig.colorbar(q0, ax=axes[0], fraction=0.046, pad=0.04, label="mm/s")
        axes[0].set_title("Velocity field")
        axes[0].axis("off")

        vmax = np.percentile(np.abs(omega), 98)
        im2 = axes[1].imshow(omega, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="s⁻¹")
        axes[1].set_title("Vorticity")
        axes[1].axis("off")

        _spectrum_gradient(axes[2], k_spec, E_k, "plasma")
        ref = E_mid / k_mid ** (-11 / 5) * k_spec ** (-11 / 5)
        axes[2].loglog(k_spec, np.exp(intercept) * k_spec ** slope, "--",
                       color="tomato", lw=1.2, label=f"$k^{{{slope:.2f}}}$")
        axes[2].loglog(k_spec, ref, "k:", lw=1.0, label=r"$k^{-11/5}$")
        axes[2].set_xlabel("$k$"); axes[2].set_ylabel("$E(k)$")
        axes[2].set_title("Energy spectrum")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "rbc_summary.png", dpi=150)
        plt.close(fig)

    # --- speed histogram: bars coloured by speed value (viridis gradient)
    if velocity_records:
        speeds = np.array([np.sqrt(r["vx"] ** 2 + r["vy"] ** 2) for r in velocity_records])
        fig, ax = plt.subplots(figsize=(6, 4))
        counts, bin_edges, patches = ax.hist(speeds, bins=50, edgecolor="white", linewidth=0.3)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_norm = Normalize(vmin=bin_centres.min(), vmax=bin_centres.max())
        hist_cmap = plt.cm.viridis
        for patch, bc in zip(patches, bin_centres):
            patch.set_facecolor(hist_cmap(hist_norm(bc)))
        sm_h = ScalarMappable(norm=hist_norm, cmap=hist_cmap)
        sm_h.set_array([])
        plt.colorbar(sm_h, ax=ax, label="speed bin  (mm/s)")
        ax.set_xlabel("Speed  (mm/s)")
        ax.set_ylabel("Count")
        ax.set_title("PTV particle speed distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "speed_histogram.png", dpi=150)
        plt.close(fig)

    print(f"Figures saved to {out_dir}/")

except Exception as e:
    import traceback; traceback.print_exc()
    print(f"WARNING: Step 6 failed — {e}")


# ── Step 7: PTV speed report ───────────────────────────────────────────────────
try:
    if not velocity_records:
        print("WARNING: Step 7 — no velocity records available")
    else:
        # aggregate per-step (frame-to-frame) velocities → mean speed per trajectory
        speed_by_id = defaultdict(list)
        last_pos_by_id = {}
        for rec in velocity_records:
            pid = rec["particle_id"]
            speed_by_id[pid].append(np.sqrt(rec["vx"] ** 2 + rec["vy"] ** 2))
            last_pos_by_id[pid] = (rec["x"], rec["y"])

        summary = []
        for pid, speeds in speed_by_id.items():
            x_px, y_px = last_pos_by_id[pid]
            summary.append((pid, x_px / cfg.px_per_mm, y_px / cfg.px_per_mm,
                            float(np.mean(speeds))))

        summary.sort(key=lambda r: r[3], reverse=True)

        print("=== Top 10 Fastest Particles ===")
        print(f"  {'particle_id':>12}  {'x_mm':>8}  {'y_mm':>8}  {'speed_mm_s':>12}")
        for pid, x_mm, y_mm, spd in summary[:10]:
            print(f"  {pid:>12}  {x_mm:>8.2f}  {y_mm:>8.2f}  {spd:>12.3f}")

except Exception as e:
    print(f"WARNING: Step 7 failed — {e}")
