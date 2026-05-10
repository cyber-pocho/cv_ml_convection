import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import h5py
from pathlib import Path 
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log=logging.getlogger(__name__)

@dataclass
class RBConfig: 
    """
    Spanish:
    Parametros fisicos y de vision del experimento.
    Valores ajustables al experimento real. 
    """
    # Constants and preprocessing
    px_per_mm:float=10.0 # camera callibration
    fps: float=30.0 # Frames per second
    dt:float=field(init=False) # time intervales

    gaussian_ksize:int=5 # kernel para suavizado inicial  
    bg_history:int=50 # frames para estimar el fondo estatico
    clahe_clip: float=2.0 # contraste adaptivo

    # Particle detection
    blob_min_area:float=8.0 # minimum area in px^2 to estimate a particle
    blob_max_area=float=300.0 # maximum area
    blob_min_circularity: float=0.2 # circularity parameter, MICA particles are rectangles so we want low circularity
    blob_min_inertia:float=0.05 # we want there to be point particles not large, tall blobs.
    blob_min_convexity: float=0.4 # minimum convexity 
    
    # --- Tracking

    max_displacement_px:float=20.0
    min_particles_per_frame:int=10

    # ---- Classic PIV
    piv_window_size:int=32
    piv_overlap:float=0.05

    def __post_init__(self): 
        self.dt=1.0/self.fps


class FramePreprocessor: 
    """
    SPN: 
        PrepLara el frame raw para la deteccion de las particulas. 
    """
    def __init__(self, config):   
        self.cfg=config
        self.bg_subtractor=cv2.createBackgroundSubtractorMOG2(
            history=config.bg_history, 
            varThreshold=16, 
            detectShadows=False
            )
        self.clahe=cv2.createCLAHE(
            clipLimit=config.clahe_clip, 
            titleGridSize=config.clahe_title, 
            )
        self.bg_ready=False
        self._frame_count=0 
    def warm_up(self, frames:list[np.ndarray])->None:
        """
        SPN:
            Alimenta o llama los primeros ~50 frames antes de procesar
        """
        for f in frames: 
            gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) if f.ndim==3 else f self.bg_subtractor.apply(gray)
        self.bg_ready=True
        log.info(f"Background estimator warmed up with {len(frames)} frames")
    def process(self, frame:np.ndarray)->tuple[np.ndarray, np.ndarray]: 
        """
        Spn: 
            Devuelve (fram_enhanced, binary_mask) para deteccion. 
            frame_enhanced: imagen de 8-bits con mejor contraste. 
            binary_mask: mascara binaria donde los blobs brillantes son 255. 

        """
        # if 3, that means color and thus we'll have to convert to gray. 
        if frame.ndim==3:
            gray=cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
        else: 
            gray=frame.copy()
        # we now reduce Gaussian noise
        blurred=cv2.GaussianBlur(
                gray, 
                (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize), 
                0,
                )
        # CLAHE: For uneven ilumninatio (if controled setup, it can be a bit troublesome)
        enhanced=self.clahe.apply(blurred)

        # background substractor 
        if self._bg_ready: 
            fg_mask=self.bg_subtractor.apply(enhanced)
        else: 
            # without a clear Background
            _, fg_mask=cv2.threshold(
                    enhanced, 0, 255, cv2.THRESH_BINARY+cv2.
                    )

        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fg_mask=cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask=cv2.morphologyEx(fg_mask,cv2.MORPH_OPEN, kernel)
        self._fram_count+=1 
        return enhance, fg_mask
