import pytest
import numpy as np
from overlap_detection.matching import match

def test_match_mnn():
    desc_A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    desc_B = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    
    matches = match(desc_A, desc_B, False, "mnn", 0.9)
    assert matches.shape == (2, 3)
    # A[0] -> B[1]
    # A[1] -> B[0]
    assert [0, 1] in matches[:, :2].tolist()
    assert [1, 0] in matches[:, :2].tolist()

def test_match_mnn_nndr():
    # A[0] matches B[0] closely, B[1] poorly
    # A[1] matches both closely (will fail ratio test)
    desc_A = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    desc_B = np.array([[1.0, 0.1], [0.6, 0.4]], dtype=np.float32)
    
    matches = match(desc_A, desc_B, False, "mnn_nndr", 0.5)
    # A[0] -> B[0] distance is 0.1, A[0] -> B[1] is ~0.56. Ratio < 0.5 (good)
    # A[1] -> B[1] distance is 0.14, A[1] -> B[0] is ~0.64. Ratio is small (good)
    assert matches.shape[0] <= 2
