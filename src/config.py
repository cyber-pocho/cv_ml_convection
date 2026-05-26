from dataclasses import dataclass, field
from enum import Enum


class CameraView(Enum):
    # LATERAL: y is wall-normal; vy is the physically meaningful heat-transport component
    LATERAL = "lateral"
    # TOP: x and y are both horizontal; vertical (wall-normal) velocity is not observable
    TOP = "top"


@dataclass
class RBConfig:
    # --- Camera calibration
    px_per_mm: float = 10.0          # spatial scale: how many pixels span one millimetre
    fps: float = 30.0                # recording frame rate; drives velocity units (px/frame → mm/s)
    dt: float = field(init=False)    # inter-frame interval in seconds; derived from fps

    # --- Camera geometry
    # determines axis interpretation everywhere downstream — see CameraView docstring
    camera_view: CameraView = CameraView.LATERAL
    container_height_mm: float = 20.0  # H in RBC: hot-plate to cold-plate separation (wall-normal extent)
    container_width_mm: float  = 80.0  # horizontal extent of the visible flow field

    # --- Preprocessing
    gaussian_ksize: int = 5     # Gaussian blur kernel size; suppresses pixel noise before CLAHE
    bg_history: int = 50        # frames MOG2 averages over to build the background model
    clahe_clip: float = 2.0     # CLAHE contrast clip limit; prevents amplifying pure noise
    clahe_tile: int = 8         # CLAHE tile grid size (e.g. 8 → 8×8); balances local vs global contrast

    # --- Particle detection
    blob_min_area: float = 8.0          # minimum blob area in px²; rejects sub-pixel speckle
    blob_max_area: float = 300.0        # maximum blob area in px²; rejects particle aggregates
    blob_min_circularity: float = 0.2   # low threshold because mica particles are flat and irregular
    blob_min_inertia: float = 0.05      # inertia ratio: 0 = line, 1 = circle; rejects streaks
    blob_min_convexity: float = 0.4     # convexity ratio; rejects deeply concave noise blobs

    # --- Tracking
    max_displacement_px: float = 20.0   # Hungarian search radius; beyond this we assume no physical match
    min_particles_per_frame: int = 10   # if SimpleBlobDetector returns fewer, fall back to contour detection

    # --- PIV
    piv_window_size: int = 32   # interrogation window side length in pixels
    piv_overlap: float = 0.5    # fraction of window_size used as grid step (0.5 = 50% overlap)

    def __post_init__(self):
        self.dt = 1.0 / self.fps
