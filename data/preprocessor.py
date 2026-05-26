import cv2
import numpy as np
import logging
from src.config import RBConfig

log = logging.getLogger(__name__)


class FramePreprocessor:

    def __init__(self, cfg: RBConfig):
        self.cfg = cfg
        # MOG2 models each pixel as a Gaussian mixture — robust to slow illumination drift
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.bg_history,
            varThreshold=16,
            detectShadows=False,
        )
        # CLAHE equalises locally so bright mica reflections don't crush surrounding contrast
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_tile, cfg.clahe_tile),
        )
        self.bg_ready = False
        self._frame_count = 0

    def warm_up(self, frames: list) -> None:
        # Prime the background model before real experiment frames start so the first
        # particle-bearing frame isn't mistaken for background
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            self.bg_subtractor.apply(gray)
        self.bg_ready = True
        log.info(f"Background estimator warmed up with {len(frames)} frames.")

    def process(self, frame: np.ndarray) -> tuple:
        # Returns (enhanced, fg_mask): enhanced feeds blob detection, mask gates the search region
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

        # Blur before CLAHE so noise isn't amplified into spurious particle candidates
        blurred = cv2.GaussianBlur(gray, (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize), 0)
        enhanced = self.clahe.apply(blurred)

        if self.bg_ready:
            fg_mask = self.bg_subtractor.apply(enhanced)
        else:
            # Otsu fallback until MOG2 has seen enough frames to model the background reliably
            _, fg_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close fills holes inside particle blobs; open removes isolated noise specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        self._frame_count += 1
        return enhanced, fg_mask
