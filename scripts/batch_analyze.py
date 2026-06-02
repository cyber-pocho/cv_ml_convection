#!/usr/bin/env python3
"""
batch_analyze.py — RBC batch pipeline: all 5 videos, per-video figures,
cross-video comparisons, and a printed summary table.

Sequential research-notebook style — no classes, equations map to code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots                         # noqa: F401 — registers 'science' style
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401 — registers 3d projection

plt.style.use(["science", "no-latex"])

from src.config import RBConfig, CameraView
from src.piv import compute_piv
from src.tracker import match_particles, update_trajectories, compute_velocities
from data.preprocessor import FramePreprocessor
from data.detector import MicaDetector, Particle


# ── Config ────────────────────────────────────────────────────────────────────────

# video3 excluded — thermocouple read 290°C, above glycerin decomposition point
VIDEOS = {
    "video2.mp4": {"T_plate": 130, "T_ambient": 25, "delta_T": 105},
    "video4.mp4": {"T_plate": 120, "T_ambient": 25, "delta_T":  95},
    "video5.mp4": {"T_plate": 100, "T_ambient": 25, "delta_T":  75},
    "video6.mp4": {"T_plate": 100, "T_ambient": 25, "delta_T":  75},
    "video7.mp4": {"T_plate":  90, "T_ambient": 25, "delta_T":  65},
}

# glycerin 80% + water 20% at ~50°C bulk average
FLUID = {
    "nu":    60e-6,   # kinematic viscosity [m²/s]
    "alpha":  5e-4,   # volumetric expansion [K⁻¹]
    "kappa":  1e-7,   # thermal diffusivity [m²/s]
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BATCH_DIR = Path("outputs") / f"batch_{timestamp}"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

print(f"Batch output: {BATCH_DIR}/")


# ── Physics helpers ───────────────────────────────────────────────────────────────

def vorticity(u, v, dx, dy):
    # omega_z = dv/dx - du/dy: z-component of curl, meaningful in top view
    return np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)


def divergence(u, v, dx, dy):
    # div u = du/dx + dv/dy; near-zero confirms the flow is incompressible
    div = np.gradient(u, dx, axis=1) + np.gradient(v, dy, axis=0)
    return div, float(np.sqrt(np.mean(div**2)))


def kinetic_energy_spectrum(u, v):
    # azimuthally-averaged 2D FFT energy density → 1D E(k)
    # assumes isotropy, which is reasonable for developed 2D turbulence
    ny, nx = u.shape
    U = np.fft.fft2(u)
    V = np.fft.fft2(v)
    E2d = 0.5 * (np.abs(U)**2 + np.abs(V)**2) / (nx * ny)**2
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    k_max = int(min(nx, ny) // 2)
    k_bins = np.arange(1, k_max)
    E_k = np.array([E2d[(K >= k - 0.5) & (K < k + 0.5)].sum() for k in k_bins])
    valid = E_k > 0
    return k_bins[valid].astype(float), E_k[valid]


def fit_inertial_range(k, E_k):
    # log-log slope over middle half: skips injection scale (low k) and dissipation (high k)
    n = len(k)
    if n < 4:
        return 0.0, 0.0
    i0, i1 = max(1, n // 4), max(2, 3 * n // 4)
    slope, intercept = np.polyfit(np.log(k[i0:i1]), np.log(E_k[i0:i1]), 1)
    return float(slope), float(intercept)


def reynolds_estimate(u, v, L_mm, nu):
    # Re = U_rms * L / nu; u, v in mm/s → convert to SI before dividing by nu [m²/s]
    U_rms = float(np.sqrt(np.mean(u**2 + v**2))) * 1e-3
    return U_rms * (L_mm * 1e-3) / nu


def rayleigh_estimate(delta_T, cfg, alpha, nu, kappa):
    # Ra = g * alpha * dT * H^3 / (nu * kappa)
    # Ra > 1708 → convection onset;  Ra > 1e5 → turbulent regime
    g = 9.81
    H = cfg.container_height_mm / 1e3
    return g * alpha * delta_T * H**3 / (nu * kappa)


def nusselt_proxy(u, v, kappa_star, cfg):
    # top-view proxy: Nu ~ 1 + U_rms / kappa_star
    # kappa_star = sqrt(16 / (Ra * Pr)) is a dimensionless convective velocity scale
    mean_speed = float(np.mean(np.sqrt(u**2 + v**2)))
    denom = kappa_star if kappa_star > 1e-12 else 1.0
    return 1.0 + mean_speed / denom


# ── Per-video loop ────────────────────────────────────────────────────────────────

results_list = []

for video_name, meta in VIDEOS.items():
    delta_T = meta["delta_T"]
    video_path = Path("video") / video_name

    print(f"\n{'='*60}")
    print(f"Processing {video_name}  |  ΔT = {delta_T}°C")
    print(f"{'='*60}\n")

    video_stem = Path(video_name).stem
    vid_dir = BATCH_DIR / video_stem
    vid_dir.mkdir(parents=True, exist_ok=True)

    try:

        # ── Step 1: Load frames, ROI crop, warm up preprocessor ──────────────
        print("  Step 1: Loading frames...")
        cfg = RBConfig(
            camera_view=CameraView.TOP,
            px_per_mm=5.0,
            fps=30.0,
            roi_fraction=0.6,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            cfg.fps = actual_fps
            cfg.dt = 1.0 / cfg.fps

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            y0 = int(h * (1 - cfg.roi_fraction) / 2)
            x0 = int(w * (1 - cfg.roi_fraction) / 2)
            frame = frame[y0 : y0 + int(h * cfg.roi_fraction),
                          x0 : x0 + int(w * cfg.roi_fraction)]
            frames.append(frame)
        cap.release()

        preprocessor = FramePreprocessor(cfg)
        # prime MOG2 on initial frames so the background model is stable by experiment start
        preprocessor.warm_up(frames[: cfg.bg_history])

        roi_h, roi_w = frames[0].shape[:2]
        print(f"    {len(frames)} frames  |  ROI {roi_h}×{roi_w} px  |  {cfg.fps:.1f} fps")

        # ── Step 2: Preprocess all frames ─────────────────────────────────────
        print("  Step 2: Preprocessing...")
        enhanced_frames, masks = [], []
        for frame in frames:
            enhanced, mask = preprocessor.process(frame)
            enhanced_frames.append(enhanced)
            masks.append(mask)
        print(f"    {len(enhanced_frames)} frames preprocessed")

        # ── Step 3: PTV — detect + track + velocities ─────────────────────────
        print("  Step 3: PTV (detect, track, velocities)...")
        detector = MicaDetector(cfg)
        all_particles = [
            detector.detect(enh, msk, frame_idx=i)
            for i, (enh, msk) in enumerate(zip(enhanced_frames, masks))
        ]
        mean_n = float(np.mean([len(p) for p in all_particles]))

        trajectories = update_trajectories({}, [], all_particles[0], frame_idx=0)
        for i in range(len(all_particles) - 1):
            active_ids = sorted(trajectories.keys())
            virtual_prev = [
                Particle(x=trajectories[tid][-1][0],
                         y=trajectories[tid][-1][1],
                         area=0.0, frame_idx=i)
                for tid in active_ids
            ]
            raw = match_particles(virtual_prev, all_particles[i + 1], cfg)
            matches = [(active_ids[ia], ib) for ia, ib in raw]
            trajectories = update_trajectories(trajectories, matches, all_particles[i + 1], i + 1)

        velocity_records = compute_velocities(trajectories, cfg)
        n_trajectories = len(trajectories)
        print(f"    {mean_n:.1f} particles/frame  |  {n_trajectories} trajectories  |  {len(velocity_records)} velocity records")

        # ── Step 4: PIV — velocity field for all frame pairs ──────────────────
        print("  Step 4: PIV (velocity field)...")
        u_stack, v_stack = [], []
        piv_x = piv_y = None

        for i in range(len(enhanced_frames) - 1):
            n1 = len(all_particles[i])
            n2 = len(all_particles[i + 1])
            # cross-correlation degrades below minimum seeding density — skip sparse pairs
            if n1 < cfg.min_particles_per_frame or n2 < cfg.min_particles_per_frame:
                continue
            result = compute_piv(enhanced_frames[i], enhanced_frames[i + 1], cfg)
            u_stack.append(result["u"])
            v_stack.append(result["v"])
            piv_x = result["x"]
            piv_y = result["y"]

        if not u_stack:
            raise RuntimeError("No valid PIV pairs — seeding too sparse for this video")

        u_stack = np.stack(u_stack, axis=0)   # (N_pairs, H_grid, W_grid)
        v_stack = np.stack(v_stack, axis=0)
        u_mean = u_stack.mean(axis=0)
        v_mean = v_stack.mean(axis=0)
        n_pairs, H_grid, W_grid = u_stack.shape
        print(f"    {n_pairs} frame pairs  |  grid {H_grid}×{W_grid}")

        # ── Step 5: Physics ───────────────────────────────────────────────────
        print("  Step 5: Physics...")
        nu = FLUID["nu"]
        alpha = FLUID["alpha"]
        kappa = FLUID["kappa"]
        Pr = nu / kappa

        dx = float(piv_x[1] - piv_x[0]) if len(piv_x) > 1 else 1.0 / cfg.px_per_mm
        dy = float(piv_y[1] - piv_y[0]) if len(piv_y) > 1 else 1.0 / cfg.px_per_mm

        omega = vorticity(u_mean, v_mean, dx, dy)
        div, div_rms = divergence(u_mean, v_mean, dx, dy)
        k, E_k = kinetic_energy_spectrum(u_mean, v_mean)
        slope, intercept = fit_inertial_range(k, E_k)
        Re = reynolds_estimate(u_mean, v_mean, cfg.container_width_mm, nu)
        omega_rms = float(np.sqrt(np.mean(omega**2)))

        Ra = rayleigh_estimate(delta_T, cfg, alpha, nu, kappa)
        kappa_star = np.sqrt(16.0 / (Ra * Pr))
        Nu = nusselt_proxy(u_mean, v_mean, kappa_star, cfg)

        print(f"    Ra = {Ra:.3e}  |  Re = {Re:.1f}  |  Nu_proxy = {Nu:.1f}")
        print(f"    E(k) slope = {slope:.3f}  (theory: −2.20)")
        print(f"    ω_rms = {omega_rms:.4f}  |  div_rms = {div_rms:.6f}")

        # ── Step 6: Per-video figures ─────────────────────────────────────────
        print("  Step 6: Saving figures...")
        last_frame = enhanced_frames[-1]
        speed_2d = np.sqrt(u_mean**2 + v_mean**2)

        # pixel-space grid for quiver overlaid on image
        X_px = piv_x * cfg.px_per_mm
        Y_px = piv_y * cfg.px_per_mm
        XX_px, YY_px = np.meshgrid(X_px, Y_px)

        # mm-space grid for 3D surface plots (physical axes)
        X_mm, Y_mm = np.meshgrid(piv_x, piv_y)

        # ·· velocity_field.png — quiver coloured by magnitude ················
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(last_frame, cmap="gray", origin="upper")
        spd_max = max(float(np.percentile(speed_2d, 95)), 1e-3)
        norm_q = Normalize(vmin=0, vmax=spd_max)
        q = ax.quiver(XX_px, YY_px, u_mean, -v_mean, speed_2d,
                      cmap="plasma", norm=norm_q,
                      scale=spd_max * 20, width=0.003, alpha=0.9)
        plt.colorbar(q, ax=ax, label="speed (mm/s)")
        ax.set_title(f"{video_stem}  |  ΔT = {delta_T}°C")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(vid_dir / "velocity_field.png", dpi=200)
        plt.close(fig)

        # ·· velocity_3d.png — 3D surface of speed magnitude ·················
        fig = plt.figure(figsize=(9, 6))
        ax3 = fig.add_subplot(111, projection="3d")
        surf = ax3.plot_surface(X_mm, Y_mm, speed_2d,
                                cmap="plasma",
                                norm=Normalize(vmin=0, vmax=spd_max),
                                linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax3, shrink=0.5, label="speed (mm/s)")
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("y (mm)")
        ax3.set_zlabel("speed (mm/s)")
        ax3.set_title(f"Speed surface  |  ΔT = {delta_T}°C")
        ax3.view_init(elev=35, azim=225)
        fig.tight_layout()
        fig.savefig(vid_dir / "velocity_3d.png", dpi=200)
        plt.close(fig)

        # ·· vorticity.png — diverging colormap centred at 0 ·················
        vmax_om = max(float(np.percentile(np.abs(omega), 98)), 1e-6)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(omega, cmap="RdBu_r",
                       vmin=-vmax_om, vmax=vmax_om, origin="upper")
        plt.colorbar(im, ax=ax, label=r"$\omega_z$ (s$^{-1}$)")
        ax.set_title(f"Vorticity  |  ΔT = {delta_T}°C")
        fig.tight_layout()
        fig.savefig(vid_dir / "vorticity.png", dpi=200)
        plt.close(fig)

        # ·· vorticity_3d.png — 3D surface of vorticity ······················
        fig = plt.figure(figsize=(9, 6))
        ax3 = fig.add_subplot(111, projection="3d")
        surf_om = ax3.plot_surface(X_mm, Y_mm, omega,
                                   cmap="RdBu_r",
                                   norm=Normalize(vmin=-vmax_om, vmax=vmax_om),
                                   linewidth=0, antialiased=True)
        fig.colorbar(surf_om, ax=ax3, shrink=0.5, label="vorticity (s$^{-1}$)")
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("y (mm)")
        ax3.set_zlabel("vorticity (s$^{-1}$)")
        ax3.set_title(f"Vorticity surface  |  ΔT = {delta_T}°C")
        ax3.view_init(elev=35, azim=225)
        fig.tight_layout()
        fig.savefig(vid_dir / "vorticity_3d.png", dpi=200)
        plt.close(fig)

        # ·· energy_spectrum.png — log-log with fit and k^(-11/5) reference ··
        k_mid = k[len(k) // 2]
        E_mid = E_k[len(E_k) // 2]
        k_ref_scale = E_mid / (k_mid ** (-11.0 / 5.0))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(k, E_k, color="steelblue", lw=1.5, label=r"$E(k)$")
        ax.loglog(k, np.exp(intercept) * k ** slope, "--",
                  color="tomato", lw=1.2, label=f"fit $k^{{{slope:.2f}}}$")
        ax.loglog(k, k_ref_scale * k ** (-11.0 / 5.0), "k:", lw=1.0, label=r"$k^{-11/5}$ ref")
        ax.annotate(f"slope = {slope:.2f}\n$Ra$ = {Ra:.1e}",
                    xy=(0.05, 0.08), xycoords="axes fraction", fontsize=8)
        ax.set_xlabel("wavenumber $k$")
        ax.set_ylabel("$E(k)$")
        ax.set_title(f"Energy spectrum  |  ΔT = {delta_T}°C")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(vid_dir / "energy_spectrum.png", dpi=200)
        plt.close(fig)

        # ·· trajectories.png — 100 most active, colour = speed ··············
        if velocity_records:
            speed_by_id = defaultdict(list)
            for rec in velocity_records:
                speed_by_id[rec["particle_id"]].append(
                    np.sqrt(rec["vx"]**2 + rec["vy"]**2))
            mean_spd_id = {pid: float(np.mean(s)) for pid, s in speed_by_id.items()}

            # rank by number of velocity records (active frames in trajectory)
            top_ids = sorted(speed_by_id.keys(),
                             key=lambda pid: len(speed_by_id[pid]), reverse=True)[:100]

            all_spd_vals = list(mean_spd_id.values())
            traj_norm = Normalize(vmin=np.percentile(all_spd_vals, 5),
                                  vmax=np.percentile(all_spd_vals, 95))

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(last_frame, cmap="gray", origin="upper")
            for tid in top_ids:
                if tid not in trajectories or len(trajectories[tid]) < 2:
                    continue
                pos = trajectories[tid]
                xs = np.array([p[0] for p in pos])
                ys = np.array([p[1] for p in pos])
                spd = speed_by_id[tid]
                n_seg = len(xs) - 1
                seg_spd = np.interp(np.linspace(0, 1, n_seg),
                                    np.linspace(0, 1, max(len(spd), 1)),
                                    spd)
                pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc_t = LineCollection(segs, array=seg_spd, cmap="plasma",
                                      norm=traj_norm, lw=0.8, alpha=0.75)
                ax.add_collection(lc_t)
            sm_t = ScalarMappable(norm=traj_norm, cmap=plt.cm.plasma)
            sm_t.set_array([])
            plt.colorbar(sm_t, ax=ax, label="speed (mm/s)")
            ax.set_title(f"Trajectories (top 100)  |  ΔT = {delta_T}°C")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(vid_dir / "trajectories.png", dpi=200)
            plt.close(fig)

        # ·· divergence.png — 2D heatmap with RMS annotated ··················
        dmax = max(float(np.percentile(np.abs(div), 98)), 1e-6)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(div, cmap="RdBu_r", vmin=-dmax, vmax=dmax, origin="upper")
        plt.colorbar(im, ax=ax, label=r"$\nabla \cdot \mathbf{u}$ (s$^{-1}$)")
        ax.set_title(f"Divergence  (RMS = {div_rms:.4f} s$^{{-1}}$)  |  ΔT = {delta_T}°C")
        fig.tight_layout()
        fig.savefig(vid_dir / "divergence.png", dpi=200)
        plt.close(fig)

        # ·· speed_histogram.png — PTV particle speeds ·······················
        if velocity_records:
            speeds = np.array([np.sqrt(r["vx"]**2 + r["vy"]**2) for r in velocity_records])
            s_mean = float(np.mean(speeds))
            s_std = float(np.std(speeds))

            fig, ax = plt.subplots(figsize=(6, 4))
            counts, bin_edges, patches = ax.hist(speeds, bins=50,
                                                 edgecolor="white", linewidth=0.3)
            bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            hn = Normalize(bin_centres.min(), bin_centres.max())
            for patch, bc in zip(patches, bin_centres):
                patch.set_facecolor(plt.cm.plasma(hn(bc)))
            ax.axvline(s_mean, color="white", lw=1.5, linestyle="--", alpha=0.8)
            ax.annotate(f"mean = {s_mean:.1f} mm/s\nstd = {s_std:.1f} mm/s",
                        xy=(0.62, 0.82), xycoords="axes fraction", fontsize=8)
            ax.set_xlabel("speed (mm/s)")
            ax.set_ylabel("count")
            ax.set_title(f"Speed distribution  |  ΔT = {delta_T}°C")
            fig.tight_layout()
            fig.savefig(vid_dir / "speed_histogram.png", dpi=200)
            plt.close(fig)

        print(f"    Figures → {vid_dir}/")

        # ── Step 7: Collect per-video summary ─────────────────────────────────
        results_list.append({
            "name":              video_name,
            "delta_T":           delta_T,
            "Ra":                Ra,
            "Re":                Re,
            "Nu":                Nu,
            "omega_rms":         omega_rms,
            "div_rms":           div_rms,
            "E_k_slope":         slope,
            "n_particles_avg":   mean_n,
            "n_trajectories":    n_trajectories,
            "k":                 k,
            "E_k":               E_k,
            "u_mean":            u_mean,
            "v_mean":            v_mean,
            "omega":             omega,
            "intercept":         intercept,
        })

    except Exception as e:
        print(f"\n  WARNING: {video_name} failed — {e}")
        traceback.print_exc()
        print("  Continuing to next video...\n")
        continue


# ── Comparison figures ────────────────────────────────────────────────────────────

if not results_list:
    print("\nWARNING: No videos processed successfully. Skipping comparison plots.")
else:
    comp_dir = BATCH_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Building comparison figures...")

    # sort by delta_T ascending so physical trend (cool → hot) reads left to right
    results_list.sort(key=lambda r: r["delta_T"])

    delta_Ts  = [r["delta_T"]    for r in results_list]
    Ras       = [r["Ra"]         for r in results_list]
    Res       = [r["Re"]         for r in results_list]
    Nus       = [r["Nu"]         for r in results_list]
    slopes    = [r["E_k_slope"]  for r in results_list]
    names     = [r["name"]       for r in results_list]
    stems     = [Path(n).stem    for n in names]

    # ── parameters_vs_deltaT.png — 2×2 grid ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].semilogy(delta_Ts, Ras, "o-", color="royalblue", ms=5)
    for dT, Ra, s in zip(delta_Ts, Ras, stems):
        axes[0, 0].annotate(s, (dT, Ra), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
    axes[0, 0].set_xlabel(r"$\Delta T$ (°C)")
    axes[0, 0].set_ylabel(r"$Ra$")
    axes[0, 0].set_title(r"Rayleigh number vs $\Delta T$")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    axes[0, 1].plot(delta_Ts, Res, "o-", color="darkorange", ms=5)
    for dT, Re, s in zip(delta_Ts, Res, stems):
        axes[0, 1].annotate(s, (dT, Re), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
    axes[0, 1].set_xlabel(r"$\Delta T$ (°C)")
    axes[0, 1].set_ylabel(r"$Re$")
    axes[0, 1].set_title(r"Reynolds number vs $\Delta T$")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(delta_Ts, Nus, "o-", color="forestgreen", ms=5)
    for dT, Nu, s in zip(delta_Ts, Nus, stems):
        axes[1, 0].annotate(s, (dT, Nu), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
    axes[1, 0].set_xlabel(r"$\Delta T$ (°C)")
    axes[1, 0].set_ylabel(r"$Nu_\mathrm{proxy}$")
    axes[1, 0].set_title(r"Nusselt proxy vs $\Delta T$")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(delta_Ts, slopes, "o-", color="crimson", ms=5)
    axes[1, 1].axhline(-2.20, color="gray", lw=1.2, linestyle="--", label="theory −2.20")
    for dT, sl, s in zip(delta_Ts, slopes, stems):
        axes[1, 1].annotate(s, (dT, sl), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
    axes[1, 1].set_xlabel(r"$\Delta T$ (°C)")
    axes[1, 1].set_ylabel(r"$E(k)$ slope")
    axes[1, 1].set_title(r"Inertial range slope vs $\Delta T$")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(comp_dir / "parameters_vs_deltaT.png", dpi=200)
    plt.close(fig)

    # ── all_spectra.png — overlaid E(k), sequential colormap by delta_T ──────
    fig, ax = plt.subplots(figsize=(7, 5))
    dT_min, dT_max = min(delta_Ts), max(delta_Ts)
    cmap_seq = plt.cm.cool_r       # cool (low ΔT) → warm (high ΔT)

    for r in results_list:
        t = (r["delta_T"] - dT_min) / max(dT_max - dT_min, 1)
        ax.loglog(r["k"], r["E_k"], lw=1.4, color=cmap_seq(t),
                  label=f"$\\Delta T = {r['delta_T']}$°C")

    # theoretical reference scaled to sit visually near the data
    r0 = results_list[len(results_list) // 2]
    k_mid0 = r0["k"][len(r0["k"]) // 2]
    E_mid0 = r0["E_k"][len(r0["E_k"]) // 2]
    k_ref = r0["k"]
    E_ref = E_mid0 / (k_mid0 ** (-11.0 / 5.0)) * k_ref ** (-11.0 / 5.0)
    ax.loglog(k_ref, E_ref, "k--", lw=1.0, alpha=0.5, label=r"$k^{-11/5}$ theory")

    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("Kinetic energy spectra — all videos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(comp_dir / "all_spectra.png", dpi=200)
    plt.close(fig)

    # ── all_vorticity.png — 1×N strip with shared colorscale ─────────────────
    omegas_all = [r["omega"] for r in results_list]
    vmax_global = max(float(np.percentile(np.abs(om), 98)) for om in omegas_all)
    vmax_global = max(vmax_global, 1e-6)

    n_v = len(results_list)
    fig, axes = plt.subplots(1, n_v, figsize=(4 * n_v, 4))
    axes = [axes] if n_v == 1 else list(axes)
    last_im = None
    for ax, r, om in zip(axes, results_list, omegas_all):
        last_im = ax.imshow(om, cmap="RdBu_r",
                            vmin=-vmax_global, vmax=vmax_global, origin="upper")
        ax.set_title(f"$\\Delta T = {r['delta_T']}$°C", fontsize=9)
        ax.axis("off")
    fig.colorbar(last_im, ax=axes[-1], label=r"$\omega_z$ (s$^{-1}$)", shrink=0.8)
    fig.suptitle("Vorticity fields", y=1.01)
    fig.tight_layout()
    fig.savefig(comp_dir / "all_vorticity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── all_velocity_fields.png — 1×N quiver strip with shared speed scale ───
    speeds_all = [np.sqrt(r["u_mean"]**2 + r["v_mean"]**2) for r in results_list]
    spd_global = max(float(np.percentile(spd, 95)) for spd in speeds_all)
    spd_global = max(spd_global, 1e-3)
    norm_glob = Normalize(vmin=0, vmax=spd_global)

    fig, axes = plt.subplots(1, n_v, figsize=(4 * n_v, 4))
    axes = [axes] if n_v == 1 else list(axes)
    for ax, r, spd in zip(axes, results_list, speeds_all):
        u, v = r["u_mean"], r["v_mean"]
        h, w = u.shape
        step_q = max(1, min(h, w) // 10)    # ~10 arrows across the shorter axis
        u_q = u[::step_q, ::step_q]
        v_q = v[::step_q, ::step_q]
        spd_q = spd[::step_q, ::step_q]
        yq = np.arange(0, h, step_q)
        xq = np.arange(0, w, step_q)
        XXq, YYq = np.meshgrid(xq, yq)
        ax.quiver(XXq, YYq, u_q, -v_q, spd_q,
                  cmap="plasma", norm=norm_glob,
                  scale=spd_global * 20, width=0.006)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f"$\\Delta T = {r['delta_T']}$°C", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    sm_g = ScalarMappable(cmap="plasma", norm=norm_glob)
    sm_g.set_array([])
    fig.colorbar(sm_g, ax=axes[-1], label="speed (mm/s)", shrink=0.8)
    fig.suptitle("Velocity fields", y=1.01)
    fig.tight_layout()
    fig.savefig(comp_dir / "all_velocity_fields.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── summary_table.png — rendered table figure ─────────────────────────────
    # highlight the row(s) whose E(k) slope is closest to the theoretical -2.20
    sorted_by_dT = sorted(results_list, key=lambda r: r["delta_T"], reverse=True)
    min_dist = min(abs(r["E_k_slope"] - (-2.20)) for r in sorted_by_dT)

    col_labels = ["Video", "ΔT (°C)", "Ra", "Re", "Nu", "ω_RMS (s⁻¹)", "E(k) slope", "Particles/frame"]
    table_data = []
    highlight_rows = []
    for i, r in enumerate(sorted_by_dT):
        if abs(r["E_k_slope"] - (-2.20)) <= min_dist + 1e-9:
            highlight_rows.append(i)
        table_data.append([
            Path(r["name"]).stem,
            str(r["delta_T"]),
            f"{r['Ra']:.2e}",
            f"{r['Re']:.1f}",
            f"{r['Nu']:.1f}",
            f"{r['omega_rms']:.4f}",
            f"{r['E_k_slope']:.2f}",
            f"{r['n_particles_avg']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.6 * len(sorted_by_dT)))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    # header row styling
    for col in range(len(col_labels)):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")
    # highlight best-slope rows in green
    for row_idx in highlight_rows:
        for col in range(len(col_labels)):
            tbl[(row_idx + 1, col)].set_facecolor("#d4edda")

    ax.set_title("Batch Analysis Summary  "
                 "(green = closest to theory slope −2.20)",
                 fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(comp_dir / "summary_table.png", dpi=200)
    plt.close(fig)

    print(f"  Comparison figures → {comp_dir}/")


# ── Save combined data ────────────────────────────────────────────────────────────

np.save(BATCH_DIR / "all_results.npy", results_list, allow_pickle=True)
print(f"\nAll results saved → {BATCH_DIR / 'all_results.npy'}")


# ── Final console output ──────────────────────────────────────────────────────────

n_ok = len(results_list)
n_total = len(VIDEOS)

print(f"\n{'='*60}")
print("=== Batch Analysis Complete ===")
print(f"Processed: {n_ok}/{n_total} videos successfully")
print(f"Results:   {BATCH_DIR}/")

if results_list:
    print()
    print("=== Summary ===")
    header = (f"{'Video':<12} {'ΔT':>6} {'Ra':>10} {'Re':>8} {'Nu':>8} "
              f"{'ω_rms':>10} {'slope':>8} {'Ptcl/f':>8}")
    print(header)
    print("-" * len(header))
    for r in sorted(results_list, key=lambda r: r["delta_T"], reverse=True):
        print(f"{Path(r['name']).stem:<12} "
              f"{r['delta_T']:>6} "
              f"{r['Ra']:>10.2e} "
              f"{r['Re']:>8.1f} "
              f"{r['Nu']:>8.1f} "
              f"{r['omega_rms']:>10.4f} "
              f"{r['E_k_slope']:>8.2f} "
              f"{r['n_particles_avg']:>8.1f}")

    print()
    print("=== Key Finding ===")
    best = min(results_list, key=lambda r: abs(r["E_k_slope"] - (-2.20)))
    print(f"Best inertial range match: {Path(best['name']).stem}  "
          f"(slope = {best['E_k_slope']:.2f},  "
          f"ΔT = {best['delta_T']}°C,  "
          f"Ra = {best['Ra']:.0f})")
