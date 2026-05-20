import numpy as np
from shapely.geometry import Polygon

def compute_overlap_polygon(
    affine_matrix: np.ndarray,    # 2x3 mapping A -> B coordinates
    image_A_shape: tuple[int, int, int],
    image_B_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (overlap_in_A, overlap_in_B), each as Nx2 array of polygon
    vertices in clockwise order. The two polygons describe the same
    physical overlap region in each image's coordinate frame."""
    
    H_A, W_A = image_A_shape[:2]
    H_B, W_B = image_B_shape[:2]
    
    A_corners = np.array([
        [0, 0],
        [W_A, 0],
        [W_A, H_A],
        [0, H_A]
    ], dtype=np.float32)
    
    B_corners = np.array([
        [0, 0],
        [W_B, 0],
        [W_B, H_B],
        [0, H_B]
    ], dtype=np.float32)
    
    A_in_B = apply_affine(A_corners, affine_matrix)
    
    poly_A_in_B = Polygon(A_in_B)
    poly_B = Polygon(B_corners)
    
    intersection_B = poly_B.intersection(poly_A_in_B)
    
    if intersection_B.is_empty or intersection_B.geom_type not in ['Polygon', 'MultiPolygon']:
        return np.empty((0, 2)), np.empty((0, 2))
        
    if intersection_B.geom_type == 'MultiPolygon':
        # Should be rare, take the largest area
        intersection_B = max(intersection_B.geoms, key=lambda p: p.area)
        
    overlap_in_B = np.array(intersection_B.exterior.coords)[:-1] # Remove last repeated point
    
    # Ensure clockwise order. shapely exterior is counter-clockwise by default
    if Polygon(overlap_in_B).exterior.is_ccw:
        overlap_in_B = overlap_in_B[::-1]
        
    # Order top-left first (min sum of x and y)
    sums = overlap_in_B.sum(axis=1)
    start_idx = np.argmin(sums)
    overlap_in_B = np.roll(overlap_in_B, -start_idx, axis=0)
    
    # Project back to A
    inv_M = invert_affine(affine_matrix)
    overlap_in_A = apply_affine(overlap_in_B, inv_M)

    # An affine with negative determinant flips winding order on inversion.
    if Polygon(overlap_in_A).exterior.is_ccw:
        overlap_in_A = overlap_in_A[::-1]
    sums = overlap_in_A.sum(axis=1)
    overlap_in_A = np.roll(overlap_in_A, -np.argmin(sums), axis=0)

    return overlap_in_A.astype(np.float32), overlap_in_B.astype(np.float32)

def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine to Nx2 points, returns Nx2."""
    if len(points) == 0:
        return points
    points_hom = np.hstack([points, np.ones((len(points), 1))])
    return (M @ points_hom.T).T

def invert_affine(M: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine, returns 2x3."""
    R = M[:, :2]
    t = M[:, 2]
    
    inv_R = np.linalg.inv(R)
    inv_t = -inv_R @ t
    
    return np.hstack([inv_R, inv_t.reshape(2, 1)])
