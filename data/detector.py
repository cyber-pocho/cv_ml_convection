import cv2
import numpy as np
import logging
from dataclasses import dataclass
from src.config import RBConfig

log = logging.getLogger(__name__)


@dataclass
class Particle:
    x: float
    y: float
    area: float
    frame_idx: int


class MicaDetector:

    def __init__(self, cfg: RBConfig):
        self.cfg = cfg
        self.detector = self._build_detector()

    def _build_detector(self) -> cv2.SimpleBlobDetector:
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = self.cfg.blob_min_area
        params.maxArea = self.cfg.blob_max_area

        # Mica flakes are bright reflectors; blobColor=255 selects lit particles, not shadows
        params.filterByColor = True
        params.blobColor = 255

        # Shape thresholds are deliberately loose — mica is flat, irregular, and partially occluded
        params.filterByCircularity = True
        params.minCircularity = self.cfg.blob_min_circularity
        params.filterByInertia = True
        params.minInertiaRatio = self.cfg.blob_min_inertia
        params.filterByConvexity = True
        params.minConvexity = self.cfg.blob_min_convexity

        # Multi-scale voting: a blob must appear across ≥2 threshold levels to be accepted
        params.minThreshold = 10
        params.maxThreshold = 220
        params.thresholdStep = 10
        params.minRepeatability = 2
        params.minDistBetweenBlobs = 3.0

        return cv2.SimpleBlobDetector_create(params)

    def detect(self, enhanced: np.ndarray, mask: np.ndarray, frame_idx: int) -> list[Particle]:
        masked = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        keypoints = self.detector.detect(masked)

        particles = [
            Particle(x=kp.pt[0], y=kp.pt[1], area=np.pi * (kp.size / 2) ** 2, frame_idx=frame_idx)
            for kp in keypoints
        ]

        # SimpleBlobDetector misses low-contrast or oddly-shaped mica; contours are the safety net
        if len(particles) < self.cfg.min_particles_per_frame:
            particles = self._detect_by_contours(mask, frame_idx)

        log.debug(f"Frame {frame_idx}: {len(particles)} particles detected.")
        return particles

    def _detect_by_contours(self, mask: np.ndarray, frame_idx: int) -> list[Particle]:
        particles = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.cfg.blob_min_area <= area <= self.cfg.blob_max_area):
                continue

            # Image moments give sub-pixel centroid accuracy vs. bounding-box centre
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Convexity rejects torn or overlapping blobs that would produce bad centroid estimates
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and (area / hull_area) < self.cfg.blob_min_convexity:
                continue

            particles.append(Particle(x=cx, y=cy, area=area, frame_idx=frame_idx))

        return particles
