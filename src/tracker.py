import numpy as np
from scipy.optimize import linear_sum_assignment
from src.config import RBConfig, CameraView


def match_particles(particles_a, particles_b, cfg: RBConfig) -> list:
    """
    Hungarian matching between particles in two consecutive frames.
    Returns a list of (idx_a, idx_b) index pairs into the input lists.

    Typical caller pattern:
        active_ids = sorted(trajectories.keys())
        virtual_a  = [make_virtual_particle(trajectories[tid][-1]) for tid in active_ids]
        raw        = match_particles(virtual_a, curr_particles, cfg)
        matches    = [(active_ids[ia], ib) for ia, ib in raw]
        trajectories = update_trajectories(trajectories, matches, curr_particles, frame_idx)
    """
    n_a = len(particles_a)
    n_b = len(particles_b)

    if n_a == 0 or n_b == 0:
        return []

    # Build cost matrix: Euclidean distance between every (a, b) pair
    cost_matrix = np.zeros((n_a, n_b))
    for i, pa in enumerate(particles_a):
        for j, pb in enumerate(particles_b):
            dx = pa.x - pb.x
            dy = pa.y - pb.y
            cost_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reject physically implausible matches — particle cannot teleport further than the search radius
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= cfg.max_displacement_px:
            matches.append((r, c))

    return matches


def update_trajectories(trajectories: dict, matches: list, particles: list, frame_idx: int) -> dict:
    """
    Extend existing trajectories and birth new ones for unmatched particles.

    matches: (traj_id, idx_b) pairs — idx_b indexes into `particles` (the current frame).
             The caller is responsible for translating raw match indices into traj_ids
             (see match_particles docstring for the full calling pattern).
    trajectories: traj_id → [(x, y, frame_idx), ...]
    """
    matched_b_indices = set()

    for traj_id, idx_b in matches:
        p = particles[idx_b]
        if traj_id not in trajectories:
            trajectories[traj_id] = []
        trajectories[traj_id].append((p.x, p.y, frame_idx))
        matched_b_indices.add(idx_b)

    # Particles with no match birth new trajectories
    next_id = max(trajectories.keys(), default=-1) + 1
    for j, p in enumerate(particles):
        if j not in matched_b_indices:
            trajectories[next_id] = [(p.x, p.y, frame_idx)]
            next_id += 1

    return trajectories


def compute_velocities(trajectories: dict, cfg: RBConfig) -> list:
    """
    Instantaneous velocity from consecutive positions in each trajectory.

    In LATERAL view: vy is the wall-normal component — the key observable for
    heat transport (positive vy = rising hot plume, negative vy = sinking cold fluid).
    In TOP view: both vx and vy are horizontal; neither carries wall-normal information.
    """
    velocity_records = []

    for particle_id, positions in trajectories.items():
        for k in range(len(positions) - 1):
            x1, y1, f1 = positions[k]
            x2, y2, f2 = positions[k + 1]

            # Skip gaps in the trajectory — velocity is undefined across missing frames
            if f2 - f1 != 1:
                continue

            # px displacement × (mm/px) × (frames/s) = mm/s
            vx = (x2 - x1) * cfg.px_per_mm * cfg.fps
            vy = (y2 - y1) * cfg.px_per_mm * cfg.fps

            velocity_records.append({
                "x": x1,
                "y": y1,
                "vx": vx,
                "vy": vy,
                "particle_id": particle_id,
                "frame_idx": f1,
            })

    return velocity_records
