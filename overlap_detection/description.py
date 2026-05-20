import cv2
import numpy as np
from overlap_detection.types import Keypoint
from overlap_detection.liop import liop_describe
from overlap_detection.mldb import mldb_describe

def is_binary_descriptor(descriptor_name: str) -> bool:
    """True for BRIEF, BRISK, SUFREAK, MLDB. False for SIFT, RootSIFT, USURF, DAISY, LIOP."""
    return descriptor_name in ["BRIEF", "BRISK", "SUFREAK", "MLDB"]

def describe(
    image: np.ndarray,
    keypoints: list[Keypoint],
    descriptor_name: str,
    descriptor_params: dict,
    default_sigma: float,
    detector_name: str = "",
) -> tuple[list[Keypoint], np.ndarray]:
    """Compute descriptors. Returns (filtered_keypoints, descriptor_matrix).
    Keypoints may be filtered if the descriptor rejects some (e.g., too close
    to the image edge). All descriptors are computed in upright mode — angle
    is fixed to 0.0 for every keypoint regardless of what the detector found.
    Convert internally from Keypoint dataclass to cv2.KeyPoint for OpenCV calls.

    detector_name is optional but should be passed when descriptor_name is
    "MLDB": it gates the native OpenCV MLDB path to AKAZE keypoints only,
    since KAZE keypoints also carry class_id but their level indices index
    into a linear diffusion pyramid incompatible with AKAZE's nonlinear one."""
    if not keypoints:
        return [], np.array([])

    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    cv_kps = []
    for kp in keypoints:
        sigma = kp.sigma if kp.sigma is not None else default_sigma
        octave = kp.octave if kp.octave is not None else 0
        cv_kp = cv2.KeyPoint(
            x=kp.x, y=kp.y, size=sigma * 2, angle=0.0,
            response=kp.response, octave=octave,
        )
        cv_kps.append(cv_kp)

    if descriptor_name == "SIFT":
        descriptor = cv2.SIFT_create(**descriptor_params)
    elif descriptor_name == "RootSIFT":
        descriptor = cv2.SIFT_create(**descriptor_params)
    elif descriptor_name == "USURF":
        params = descriptor_params.copy()
        params["upright"] = True
        descriptor = cv2.xfeatures2d.SURF_create(**params)
    elif descriptor_name == "DAISY":
        params = descriptor_params.copy()
        params["use_orientation"] = False   # upright — matches DAISY_create API
        descriptor = cv2.xfeatures2d.DAISY_create(**params)
    elif descriptor_name == "BRIEF":
        descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(**descriptor_params)
    elif descriptor_name == "BRISK":
        descriptor = cv2.BRISK_create(**descriptor_params)
    elif descriptor_name == "SUFREAK":
        params = descriptor_params.copy()
        params["orientationNormalized"] = False
        params["scaleNormalized"] = False
        descriptor = cv2.xfeatures2d.FREAK_create(**params)
    elif descriptor_name == "MLDB":
        # Use OpenCV's native MLDB only for AKAZE keypoints.  AKAZE sets
        # class_id to the nonlinear diffusion evolution layer; native MLDB
        # indexes its own pyramid with that value, so the match is exact.
        #
        # KAZE also sets class_id, but its values index KAZE's *linear*
        # diffusion pyramid.  Passing them to AKAZE's compute() would index
        # the wrong scale level in the wrong diffusion type — semantically
        # incorrect even though it wouldn't crash.  KAZE therefore falls
        # through to the custom NumPy MLDB path along with all other detectors.
        use_native = bool(
            detector_name == "AKAZE"
            and keypoints
            and keypoints[0].class_id is not None
            and keypoints[0].class_id >= 0
        )
        if use_native:
            # Rebuild cv2.KeyPoint objects with class_id preserved so that
            # AKAZE's Compute_Descriptors() can index into its pyramid.
            akaze_kps = [
                cv2.KeyPoint(
                    x=kp.x,
                    y=kp.y,
                    size=(kp.sigma * 2) if kp.sigma is not None else default_sigma * 2,
                    angle=0.0,
                    response=kp.response,
                    octave=kp.octave if kp.octave is not None else 0,
                    class_id=kp.class_id,
                )
                for kp in keypoints
            ]
            akaze_desc = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
                **descriptor_params,
            )
            out_kps, desc_mat = akaze_desc.compute(gray_img, akaze_kps)
            if desc_mat is None:
                return [], np.array([])
            filtered_kps = []
            for cv_kp in out_kps:
                x, y = cv_kp.pt
                filtered_kps.append(Keypoint(
                    x=float(x), y=float(y), response=float(cv_kp.response),
                    sigma=(cv_kp.size / 2) if cv_kp.size > 0 else None,
                    theta=np.radians(cv_kp.angle) if cv_kp.angle != -1 else None,
                    octave=cv_kp.octave,
                    class_id=cv_kp.class_id if cv_kp.class_id >= 0 else None,
                ))
            return filtered_kps, desc_mat
        else:
            return mldb_describe(gray_img, keypoints, default_sigma)
    elif descriptor_name == "LIOP":
        # LIOP is handled entirely in NumPy — bypass the OpenCV compute() path.
        return liop_describe(gray_img, keypoints, default_sigma)
    else:
        raise ValueError(f"Unknown descriptor: {descriptor_name}")

    out_kps, desc_mat = descriptor.compute(gray_img, cv_kps)

    if desc_mat is None:
        return [], np.array([])

    if descriptor_name == "RootSIFT":
        # L1 normalize row-wise
        eps = 1e-7
        l1_norm = np.sum(np.abs(desc_mat), axis=1, keepdims=True)
        desc_mat = desc_mat / (l1_norm + eps)
        # Element-wise sqrt
        desc_mat = np.sqrt(np.maximum(desc_mat, 0)).astype(np.float32)

    filtered_kps = []
    # OpenCV compute might filter keypoints. We need to map them back.
    for cv_kp in out_kps:
        x, y = cv_kp.pt
        sigma = (cv_kp.size / 2) if cv_kp.size > 0 else None
        theta = np.radians(cv_kp.angle) if cv_kp.angle != -1 else None
        filtered_kps.append(Keypoint(
            x=float(x), y=float(y), response=float(cv_kp.response),
            sigma=sigma, theta=theta, octave=cv_kp.octave
        ))

    return filtered_kps, desc_mat
