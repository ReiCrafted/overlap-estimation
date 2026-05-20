import cv2
import numpy as np

def match(
    descriptors_A: np.ndarray,
    descriptors_B: np.ndarray,
    is_binary: bool,
    filter_mode: str,            # "mnn" | "mnn_nndr"
    nndr_threshold: float,
) -> np.ndarray:
    """Return Mx3 array of matches: columns are (idx_A, idx_B, distance).
    Matches are sorted by ascending distance — this ordering is consumed
    by PROSAC as the quality ranking."""
    
    if descriptors_A is None or descriptors_B is None or len(descriptors_A) == 0 or len(descriptors_B) == 0:
        return np.empty((0, 3), dtype=np.float32)
        
    norm_type = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
    
    if filter_mode == "mnn":
        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        cv_matches = matcher.match(descriptors_A, descriptors_B)
        
        matches = []
        for m in cv_matches:
            matches.append([m.queryIdx, m.trainIdx, m.distance])

        if not matches:
            return np.empty((0, 3), dtype=np.float32)
        matches_arr = np.array(matches, dtype=np.float32)
        matches_arr = matches_arr[matches_arr[:, 2].argsort()]
        return matches_arr
        
    elif filter_mode == "mnn_nndr":
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        
        # A to B
        matches_AB = matcher.knnMatch(descriptors_A, descriptors_B, k=2)
        good_AB = {}
        for m_list in matches_AB:
            if len(m_list) >= 2:
                m, n = m_list
                if m.distance < nndr_threshold * n.distance:
                    good_AB[m.queryIdx] = m

        # B to A
        matches_BA = matcher.knnMatch(descriptors_B, descriptors_A, k=2)
        good_BA = {}
        for m_list in matches_BA:
            if len(m_list) >= 2:
                m, n = m_list
                if m.distance < nndr_threshold * n.distance:
                    good_BA[m.queryIdx] = m
                
        # Intersect
        final_matches = []
        for idx_A, m in good_AB.items():
            idx_B = m.trainIdx
            if idx_B in good_BA and good_BA[idx_B].trainIdx == idx_A:
                final_matches.append([idx_A, idx_B, m.distance])
                
        if not final_matches:
            return np.empty((0, 3), dtype=np.float32)
        matches_arr = np.array(final_matches, dtype=np.float32)
        matches_arr = matches_arr[matches_arr[:, 2].argsort()]
        return matches_arr
    else:
        raise ValueError(f"Unknown filter mode: {filter_mode}")
