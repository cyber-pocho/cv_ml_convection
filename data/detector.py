

@dataclass
class Particle:
    """
    SPN: 
        Representa una particula de MICA detectada en un frame. 
    
    """
    x:float
    y:float
    area:float
    frame_idx:int
class MicaDetector: 
    """
    SPN: 
        Detecta particulas de mica usando SimpleBlobDetector de OpenCV. 
        Tambien se puede usar un overkill con analisis de contornos con filtros de sombra. 
    """
    def __init__(self, config:RBConfig): 
        self.cfg=config
        self.detector=self._build_detector()
    def _build_detector(self)->cv2.SimpleBlobDetector:
        params=cv2.SimpleBlobDetector_Params()
        # Here we filter by area (physical size of the mica particle)
        params.filterByArea=True
        params.minArea=self.cfg.blob_min_area
        params.maxArea=self.cfg.blob_max_area
        # Once detected, we want to check for the bright and opaque mica particles
        params.filterByColor=True
        params.blobColor=255

        # Now, we want to check if the mica particle is not of a regular shape. 
        params.filterByCircularity
        params.minCircularity=self.cfg.blob_min_circularity
        params.filterByInertia=True
        params.minInertiaRatio=self.cfg.blob_min_inertia

        params.filterByConvexity=True
        params.minConvexity=self.cfg.blob_min_convexity

        # Multiscale detection
        params.minThreshold=10
        params.maxThreshold=220
        params.thresholdStep=10
        params.minRepeatability=2 
        params.minDistBetweenBlobs=3.0

        return cv2.SimpleBlobDetector_create(params)

