import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import time

from overlap_detection.config import RunConfig
from overlap_detection.orchestrator import run_single_pair


def _align_worker(args: tuple[str, str, str]) -> tuple[str, np.ndarray | None]:
    """Worker function for multiprocessing pool.
    args: (pair_id, img_A_path, img_B_path)
    Returns: (pair_id, affine_matrix)
    """
    pair_id, path_a, path_b = args
    try:
        img_A = cv2.imread(path_a)
        if img_A is not None:
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.imread(path_b)
        if img_B is not None:
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        if img_A is None or img_B is None:
            return pair_id, None

        cfg_primary = RunConfig(
            detector="GFTT", descriptor="BRISK", estimator="PROSAC", mask_mode="fallback"
        )
        res_primary, _ = run_single_pair(img_A, img_B, cfg_primary)
        if res_primary.success and res_primary.affine_matrix is not None:
            return pair_id, res_primary.affine_matrix
            
        cfg_fallback = RunConfig(
            detector="FAST", descriptor="SIFT", estimator="PROSAC", mask_mode="fallback"
        )
        res_fallback, _ = run_single_pair(img_A, img_B, cfg_fallback)
        if res_fallback.success and res_fallback.affine_matrix is not None:
            return pair_id, res_fallback.affine_matrix
            
        return pair_id, None
    except Exception as e:
        print(f"Auto-align error on {pair_id}: {e}")
        return pair_id, None


class AutoAligner:
    """Manages background auto-alignment of image pairs using multiprocessing."""
    def __init__(self, workers: int = 6):
        self.pool = Pool(processes=workers)
        self.results = {}
        self._async_results = {}

    def queue_pairs(self, pairs: list[tuple[Path, Path]]):
        for img_a, img_b in pairs:
            pair_id = f"{img_a.stem}_{img_b.stem}"
            if pair_id not in self.results and pair_id not in self._async_results:
                args = (pair_id, str(img_a), str(img_b))
                # Fire and forget callback that stores the result
                res = self.pool.apply_async(_align_worker, (args,), callback=self._on_result)
                self._async_results[pair_id] = res

    def _on_result(self, result: tuple[str, np.ndarray | None]):
        pair_id, affine_matrix = result
        self.results[pair_id] = affine_matrix

    def get_alignment(self, pair_id: str) -> np.ndarray | None:
        """Returns the affine matrix if ready, otherwise None."""
        return self.results.get(pair_id, None)

    def is_processing(self, pair_id: str) -> bool:
        if pair_id in self.results:
            return False
        if pair_id in self._async_results:
            return not self._async_results[pair_id].ready()
        return False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()

    def shutdown(self):
        self.pool.terminate()
        self.pool.join()
