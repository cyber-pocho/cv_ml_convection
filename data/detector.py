

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
    
    def detect(self, enhanced:np.ndarray, mask:np.ndarray, frame_idx: int) -> list[Particle]: 
        """
        SPN: 
            Detecta particulas en el frame/cuadro procesado.
        """
        particles=[]
        masked=cv2.bitwise_and(enhanced, enhanced, mask=mask)
        keypoints=self.detector.detect(masked)

        for kp in keypoints:
            particles.append(Particle(
                x=kp.pt[0],
                y=kp.pt[1],
                area=np.pi*(kp.size/2)**2,
                frame_idx=frame_idx, 
                ))
        if len(particles) < self.cfg.min_particles_per_frame:
            particles=self._detect_by_contours(mask,frame_idx)
        log.debug(f"Frame {frame_idx}: {len(particles)} particles detected.")
        return particles
    
    def _detect_by_contours(self, mask:np.ndarray, frame_idx:int)->list[Particle]: 
        """
        SPN: 
            Deteccion por analisi de contornos con filtros de forma. 
        """
        particles=[]
        contours, _=cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
        for cnt in contours: 
            area=cv2.contourArea(cnt)
            if not (self.cfg.blob_min_area <=area <=self.cfg.blob_max_area): 
                continue
            # Centroids moment for subpixel
            M=cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx=M["m10"]/M["m00"]
            cy=M["m01"]/M["m00"]
            # convexity filter
            hull=cv2.convexHull(cnt)
            hull_area=cv2.contourArea(hull)
            if hull_area>0: 
                convexity = area/hull_area
                if convexity < self.cfg.blob_min_convexity: 
                    continue
            particles.append(Particle(x=cx, y=cy, area=area, frame_idx=frame_idx))
        return particles

