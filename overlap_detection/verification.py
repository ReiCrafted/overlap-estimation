import cv2
import numpy as np
from overlap_detection.types import Keypoint

_METHOD_MAP = {
    "PROSAC": cv2.USAC_PROSAC,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
}

def verify_affine(
    matches: np.ndarray,           # Mx3 from matching, sorted by distance
    keypoints_A: list[Keypoint],
    keypoints_B: list[Keypoint],
    estimator: str,                # "PROSAC" | "USAC_MAGSAC"
    ransac_threshold_px: float,
    ransac_max_iters: int,
    ransac_confidence: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Returns (affine_matrix, inlier_mask).
    affine_matrix is shape (2, 3) or None if estimation failed.
    inlier_mask is shape (M,) bool array indicating which matches
    were classified as inliers."""
    
    if matches is None or len(matches) < 3:
        return None, np.zeros((0,), dtype=bool)
        
    src_pts = np.array([[keypoints_A[int(m[0])].x, keypoints_A[int(m[0])].y] for m in matches], dtype=np.float32)
    dst_pts = np.array([[keypoints_B[int(m[1])].x, keypoints_B[int(m[1])].y] for m in matches], dtype=np.float32)
    
    method = _METHOD_MAP.get(estimator, cv2.USAC_MAGSAC)
    
    affine_matrix, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=method,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=ransac_max_iters,
        confidence=ransac_confidence
    )
    
    if affine_matrix is None:
        return None, np.zeros((len(matches),), dtype=bool)
        
    inlier_mask = inliers.flatten().astype(bool)
    if np.sum(inlier_mask) < 3:
        return None, inlier_mask
        
    return affine_matrix, inlier_mask
