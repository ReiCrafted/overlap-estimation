import cv2
import os
import numpy as np
from pathlib import Path
from multiprocessing import get_context
import time

from overlap_detection.config import RunConfig
from overlap_detection.orchestrator import run_single_pair


_DEFAULT_AUTO_ALIGN_WORKER_CAP = 6


def default_auto_align_workers() -> int:
    """Pick a worker count for background auto-alignment: ``cpu_count - 1``
    capped at ``_DEFAULT_AUTO_ALIGN_WORKER_CAP`` and floored at 1."""
    return max(1, min((os.cpu_count() or 1) - 1, _DEFAULT_AUTO_ALIGN_WORKER_CAP))


def _try_with_fallback(img_A: np.ndarray, img_B: np.ndarray,
                       cfg_template: RunConfig) -> np.ndarray | None:
    """Auto-aligner's self-contained fallback: try ``no_mask`` first, then
    ``mask`` if the first attempt produced no transform.  Returns the first
    accepted affine matrix or ``None`` if both attempts failed.

    The main pipeline no longer has a ``"fallback"`` mask mode — the
    auto-aligner owns that behaviour locally because it's the only consumer
    that genuinely benefits from short-circuiting on first success (the
    experiment runner always wants both attempts for analysis).
    """
    from dataclasses import replace
    for mode in ("no_mask", "mask"):
        cfg = replace(cfg_template, mask_mode=mode)
        attempts = run_single_pair(img_A, img_B, cfg)
        # mask_mode is concrete here, so attempts has length 1
        result, _ = attempts[0]
        if result.affine_matrix is not None:
            return result.affine_matrix
    return None


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
            detector="GFTT", descriptor="BRISK", estimator="PROSAC",
        )
        affine = _try_with_fallback(img_A, img_B, cfg_primary)
        if affine is not None:
            return pair_id, affine

        cfg_secondary = RunConfig(
            detector="FAST", descriptor="SIFT", estimator="PROSAC",
        )
        affine = _try_with_fallback(img_A, img_B, cfg_secondary)
        if affine is not None:
            return pair_id, affine

        return pair_id, None
    except Exception as e:
        print(f"Auto-align error on {pair_id}: {e}")
        return pair_id, None


class AutoAligner:
    """Manages background auto-alignment of image pairs using multiprocessing.

    Uses the ``"spawn"`` start method explicitly so workers do not inherit
    the GUI process's cv2 / Tkinter state.  Matches the orchestrator's choice
    (see ``run_experiment_matrix``); fork-based pools have caused GUI deadlocks
    on Linux in similar setups.
    """
    def __init__(self, workers: int | None = None):
        if workers is None:
            workers = default_auto_align_workers()
        ctx = get_context("spawn")
        self.pool = ctx.Pool(processes=workers)
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
