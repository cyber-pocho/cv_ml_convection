import cv2
import numpy as np
from src.config import RBConfig


def compute_piv(frame1: np.ndarray, frame2: np.ndarray, cfg: RBConfig) -> dict:
    """
    Classic PIV via phase correlation on a regular grid of interrogation windows.

    In LATERAL view: y=0 is the hot plate; positive v means upward flow (rising plume).
    In TOP view:     y is a horizontal axis; v has no wall-normal meaning.

    Returns dict with:
        u, v  — 2D arrays of shape (n_rows, n_cols), velocity in mm/s
        x, y  — 1D arrays of window-centre positions in mm
    """
    H, W = frame1.shape[:2]

    # Step size controls grid density; overlap < 1.0 means windows share pixels,
    # which increases spatial resolution at the cost of statistical independence
    step = int(cfg.piv_window_size * (1.0 - cfg.piv_overlap))
    win = cfg.piv_window_size

    # phaseCorrelate requires 32-bit float input
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)

    x_starts = list(range(0, W - win + 1, step))
    y_starts = list(range(0, H - win + 1, step))

    n_cols = len(x_starts)
    n_rows = len(y_starts)

    u_grid = np.zeros((n_rows, n_cols))  # horizontal velocity field
    v_grid = np.zeros((n_rows, n_cols))  # vertical   velocity field

    for row_idx, ys in enumerate(y_starts):
        for col_idx, xs in enumerate(x_starts):
            win1 = f1[ys : ys + win, xs : xs + win]
            win2 = f2[ys : ys + win, xs : xs + win]

            # phaseCorrelate returns (dx, dy) sub-pixel displacement and a peak response
            (dx, dy), _response = cv2.phaseCorrelate(win1, win2)

            # px displacement × (mm/px) × (frames/s) = mm/s
            u_grid[row_idx, col_idx] = dx * cfg.px_per_mm * cfg.fps
            v_grid[row_idx, col_idx] = dy * cfg.px_per_mm * cfg.fps

    # Window centres in mm — physical coordinates of each measurement point
    x_mm = np.array([xs + win // 2 for xs in x_starts], dtype=float) / cfg.px_per_mm
    y_mm = np.array([ys + win // 2 for ys in y_starts], dtype=float) / cfg.px_per_mm

    return {"u": u_grid, "v": v_grid, "x": x_mm, "y": y_mm}
