import cv2
import numpy as np
import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@dataclass
class RBConfig:
    """
    Physical and vision parameters for the RBC experiment.
    All values here are adjustable to match your real setup.
    """
    # --- Camera calibration
    px_per_mm: float = 10.0
    fps: float = 30.0
    dt: float = field(init=False)

    # --- Preprocessing
    gaussian_ksize: int = 5
    bg_history: int = 50
    clahe_clip: float = 2.0
    clahe_tile: int = 8          # tile grid size for CLAHE (e.g. 8 → 8×8 grid)

    # --- Particle detection
    blob_min_area: float = 8.0
    blob_max_area: float = 300.0
    blob_min_circularity: float = 0.2   # low → mica particles are flat/rectangular
    blob_min_inertia: float = 0.05
    blob_min_convexity: float = 0.4

    # --- Tracking
    max_displacement_px: float = 20.0
    min_particles_per_frame: int = 10

    # --- PIV
    piv_window_size: int = 32
    piv_overlap: float = 0.5

    def __post_init__(self):
        self.dt = 1.0 / self.fps


class FramePreprocessor:
    """
    Prepares a raw frame for particle detection.
    Returns an enhanced 8-bit image and a binary foreground mask.
    """

    def __init__(self, config: RBConfig):
        self.cfg = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.bg_history,
            varThreshold=16,
            detectShadows=False,
        )
        self.clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip,
            tileGridSize=(config.clahe_tile, config.clahe_tile),
        )
        self.bg_ready = False
        self._frame_count = 0

    def warm_up(self, frames: list) -> None:
        """
        Feed the first ~50 frames to the background estimator before processing.
        Call this with your opening frames before the main loop.
        """
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            self.bg_subtractor.apply(gray)
        self.bg_ready = True
        log.info(f"Background estimator warmed up with {len(frames)} frames.")

    def process(self, frame: np.ndarray) -> tuple:
        """
        Returns (enhanced, binary_mask).
          enhanced    — 8-bit image with improved contrast, ready for detection.
          binary_mask — white (255) where foreground particles are, black elsewhere.
        """
        # Convert to grayscale if color
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Reduce Gaussian noise
        blurred = cv2.GaussianBlur(
            gray,
            (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize),
            0,
        )

        # Adaptive contrast enhancement (helps with uneven illumination)
        enhanced = self.clahe.apply(blurred)

        # Foreground mask
        if self.bg_ready:
            fg_mask = self.bg_subtractor.apply(enhanced)
        else:
            # Fallback: Otsu threshold when background model isn't ready yet
            _, fg_mask = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # Clean up the mask: close small holes, remove speckle noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        self._frame_count += 1
        return enhanced, fg_mask
