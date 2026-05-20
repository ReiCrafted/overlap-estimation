import cv2
import numpy as np
import json
import datetime
import time
from pathlib import Path
import math

from overlap_detection.auto_aligner import AutoAligner
from overlap_detection.orchestrator import list_image_pairs

def invert_affine(M: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    R = M[:, :2]
    t = M[:, 2]
    inv_R = np.linalg.inv(R)
    inv_t = -inv_R @ t
    return np.hstack([inv_R, inv_t.reshape(2, 1)])

def compose_affine(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine matrices M = M1 * M2"""
    M1_hom = np.vstack([M1, [0, 0, 1]])
    M2_hom = np.vstack([M2, [0, 0, 1]])
    M = M1_hom @ M2_hom
    return M[:2, :]

def decompose_affine(M: np.ndarray) -> tuple[float, float, float, float, float]:
    """Decompose 2x3 affine matrix into tx, ty, angle(deg), sx, sy.
    Assumes no shear for simplicity of the UI."""
    tx = M[0, 2]
    ty = M[1, 2]
    
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    
    sx = math.sqrt(a*a + c*c)
    sy = math.sqrt(b*b + d*d)
    
    angle = math.atan2(c, a)
    return tx, ty, math.degrees(angle), sx, sy

def build_affine(tx: float, ty: float, angle_deg: float, sx: float, sy: float) -> np.ndarray:
    """Build 2x3 affine matrix from components."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    M = np.array([
        [sx * cos_a, -sy * sin_a, tx],
        [sx * sin_a,  sy * cos_a, ty]
    ], dtype=np.float64)
    return M


class AnnotationGUI:
    def __init__(self, dataset_dir: Path, annotator_name: str):
        self.dataset_dir = dataset_dir
        self.annotations_dir = dataset_dir / "annotations"
        self.annotator_name = annotator_name
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        self.pairs = self._list_pairs()
        self.current_idx = 0
        
        self.aligner = AutoAligner(workers=6)
        self.aligner.queue_pairs(self.pairs)
        
        self.window_name = "Manual Alignment UI"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # State
        self.imgA_umat = None
        self.imgB_umat = None
        self.imgA_shape = (0, 0)
        self.imgB_shape = (0, 0)
        
        # Alignment state
        self.tx, self.ty, self.angle, self.sx, self.sy = 0.0, 0.0, 0.0, 1.0, 1.0
        
        # Viewport state
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        
        # Interaction state
        self.dragging = False
        self.drag_mode = None # "translate", "rotate", "pan", "scale_x", "scale_y"
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Render modes: 0=Normal, 1=Anaglyph
        self.render_mode = 0

        # Save notification: None | "saving" | "saved"
        self.save_state = None
        self.save_time = 0.0

        # True once auto-align result (or GT) has been applied for the current pair.
        # All user input is frozen until this is True.
        self._auto_applied = False

        self._load_pair()

    def _list_pairs(self) -> list[tuple[Path, Path]]:
        return list_image_pairs(self.dataset_dir)

    def _load_pair(self):
        if not self.pairs:
            print("No pairs found.")
            return
            
        img_a_path, img_b_path = self.pairs[self.current_idx]
        self.pair_id = f"{img_a_path.stem}_{img_b_path.stem}"
        
        # Read as UMat for OpenCL acceleration
        imgA = cv2.imread(str(img_a_path))
        imgB = cv2.imread(str(img_b_path))
        
        if imgA is None or imgB is None:
            print(f"Failed to load pair {self.pair_id}")
            return
            
        self.imgA_shape = imgA.shape
        self.imgB_shape = imgB.shape
        
        self.imgA_umat = cv2.UMat(imgA)
        self.imgB_umat = cv2.UMat(imgB)
        
        # Reset Viewport
        self.pan_x = 0
        self.pan_y = 0
        self.zoom = min(1280 / imgA.shape[1], 720 / imgA.shape[0]) * 0.9
        self.pan_x = (1280 - imgA.shape[1] * self.zoom) / 2
        self.pan_y = (720 - imgA.shape[0] * self.zoom) / 2
        
        # Try to load existing groundtruth
        self._auto_applied = False
        gt_file = self.annotations_dir / f"{self.pair_id}_groundtruth.json"
        if gt_file.exists():
            try:
                with open(gt_file, 'r') as f:
                    data = json.load(f)
                M_A_to_B = np.array(data["affine_matrix_A_to_B"], dtype=np.float64)
                M_B_to_A = invert_affine(M_A_to_B)
                self.tx, self.ty, self.angle, self.sx, self.sy = decompose_affine(M_B_to_A)
                self._auto_applied = True   # GT loaded — no freeze needed
            except Exception as e:
                print(f"Error loading {gt_file}: {e}")
                self._reset_alignment()
                self._auto_applied = True   # Don't freeze on a load error
        else:
            # Check if auto-aligner already has a result for this pair
            if self.pair_id in self.aligner.results:
                M_auto = self.aligner.results[self.pair_id]
                if M_auto is not None:
                    M_B_to_A = invert_affine(M_auto)
                    self.tx, self.ty, self.angle, self.sx, self.sy = decompose_affine(M_B_to_A)
                else:
                    self._reset_alignment()
                self._auto_applied = True
            else:
                # Still processing — freeze until the run loop picks up the result
                self._reset_alignment()
                self._auto_applied = False

        self._render()

    def _reset_alignment(self):
        self.tx = float(self.imgA_shape[1]) * 0.2  # Slight offset guess
        self.ty = 0.0
        self.angle = 0.0
        self.sx = 1.0
        self.sy = 1.0

    def _mouse_callback(self, event, x, y, flags, param):
        if not self._auto_applied:
            return   # frozen — ignore all mouse input until auto-align resolves

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y
            # Check if inside image B bounds for translate, or outside for pan
            self.drag_mode = "translate" if not (flags & cv2.EVENT_FLAG_CTRLKEY) else "scale_x"
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.drag_mode = "rotate"
            
        elif event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY):
            self.dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.drag_mode = "pan"

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                dx = x - self.last_mouse_x
                dy = y - self.last_mouse_y
                
                # Apply transformation based on mode
                if self.drag_mode == "translate":
                    # Screen space to world space
                    self.tx += dx / self.zoom
                    self.ty += dy / self.zoom
                elif self.drag_mode == "pan":
                    self.pan_x += dx
                    self.pan_y += dy
                elif self.drag_mode == "rotate":
                    # Rotation around center
                    self.angle += dx * 0.1
                elif self.drag_mode == "scale_x":
                    self.sx += dx * 0.001
                    self.sy += dy * 0.001
                    
                self.last_mouse_x = x
                self.last_mouse_y = y
                self._render(proxy=True)

        elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP]:
            if self.dragging:
                self.dragging = False
                self.drag_mode = None
                self._render(proxy=False)

        elif event == cv2.EVENT_MOUSEWHEEL:
            # In Python OpenCV, the wheel direction is indicated by the sign of flags
            delta = 1 if flags > 0 else -1
            zoom_factor = 1.1 if delta > 0 else 0.9
            
            # Zoom around cursor
            world_x = (x - self.pan_x) / self.zoom
            world_y = (y - self.pan_y) / self.zoom
            
            self.zoom *= zoom_factor
            
            self.pan_x = x - world_x * self.zoom
            self.pan_y = y - world_y * self.zoom
            self._render()

    def _render(self, proxy=False):
        # 1. Build M_B_to_A
        M_B_to_A = build_affine(self.tx, self.ty, self.angle, self.sx, self.sy)
        
        # 2. Viewport Matrix S
        S = np.array([
            [self.zoom, 0, self.pan_x],
            [0, self.zoom, self.pan_y]
        ], dtype=np.float64)
        
        # Output canvas size
        canvas_h, canvas_w = 720, 1280
        
        # Proxy scale optimization
        P_scale = 0.5 if proxy else 1.0
        P = np.array([
            [P_scale, 0, 0],
            [0, P_scale, 0]
        ], dtype=np.float64)
        
        if proxy:
            render_w = int(canvas_w * P_scale)
            render_h = int(canvas_h * P_scale)
            # Total matrix for A: P * S
            M_A_final = compose_affine(P, S)
            # Total matrix for B: P * S * M_B_to_A
            M_B_final = compose_affine(compose_affine(P, S), M_B_to_A)
        else:
            render_w = canvas_w
            render_h = canvas_h
            M_A_final = S
            M_B_final = compose_affine(S, M_B_to_A)
            
        # Warp using UMat
        warpA = cv2.warpAffine(self.imgA_umat, M_A_final, (render_w, render_h))
        warpB = cv2.warpAffine(self.imgB_umat, M_B_final, (render_w, render_h))

        # Brighten both images for visibility
        warpA = cv2.convertScaleAbs(warpA, alpha=1.5)
        warpB = cv2.convertScaleAbs(warpB, alpha=1.5)

        # Blending modes
        if self.render_mode == 0:
            # Normal: 50/50 alpha blend
            canvas = cv2.addWeighted(warpA, 0.5, warpB, 0.5, 0)
        else:
            # Anaglyph: natural colour outside overlap, red/cyan mismatch inside.
            warpA_np = warpA.get() if isinstance(warpA, cv2.UMat) else warpA
            warpB_np = warpB.get() if isinstance(warpB, cv2.UMat) else warpB

            has_A = np.any(warpA_np > 0, axis=2)
            has_B = np.any(warpB_np > 0, axis=2)
            overlap = has_A & has_B

            canvas = np.zeros_like(warpA_np)
            canvas[has_A & ~has_B] = warpA_np[has_A & ~has_B]
            canvas[has_B & ~has_A] = warpB_np[has_B & ~has_A]
            canvas[overlap, 0] = warpB_np[overlap, 0]  # blue  from B
            canvas[overlap, 1] = warpB_np[overlap, 1]  # green from B
            canvas[overlap, 2] = warpA_np[overlap, 2]  # red   from A

        # Status Text
        status_img = canvas.get() if isinstance(canvas, cv2.UMat) else canvas
        if proxy:
            status_img = cv2.resize(status_img, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
            
        modes = ["Normal", "Anaglyph"]
        info1 = f"Pair {self.current_idx+1}/{len(self.pairs)}: {self.pair_id} | Mode: {modes[self.render_mode]}"
        info2 = f"tx:{self.tx:.1f} ty:{self.ty:.1f} ang:{self.angle:.2f} sx:{self.sx:.2f} sy:{self.sy:.2f}"

        cv2.putText(status_img, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(status_img, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        hotkeys1 = "[L-Drag]: Translate  [R-Drag]: Rotate  [CTRL+L-Drag]: Scale  [Wheel]: Zoom/Pan"
        hotkeys2 = "[m]: Mode  [r]: Reset  [a/d]: Prev/Next  [s]: Save  [ESC]: Quit"
        cv2.putText(status_img, hotkeys1, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(status_img, hotkeys2, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Auto-align status line
        if self.aligner.is_processing(self.pair_id):
            aa_text, aa_color = "Auto-Align: Processing...", (0, 165, 255)
        elif self.pair_id in self.aligner.results:
            if self.aligner.results[self.pair_id] is not None:
                aa_text, aa_color = "Auto-Align: Succeeded", (0, 255, 0)
            else:
                aa_text, aa_color = "Auto-Align: Failed", (0, 0, 255)
        else:
            aa_text, aa_color = "Auto-Align: N/A", (128, 128, 128)
        cv2.putText(status_img, aa_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, aa_color, 2)

        # Bottom-left save notification with fade
        if self.save_state == "saving":
            cv2.putText(status_img, "Saving...", (10, canvas_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        elif self.save_state == "saved":
            elapsed = time.time() - self.save_time
            alpha = max(0.0, 1.0 - elapsed / 3.0)
            if alpha <= 0.0:
                self.save_state = None
            else:
                cv2.putText(status_img, "Saved", (10, canvas_h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, int(255 * alpha), 0), 2)
            
        cv2.imshow(self.window_name, status_img)

    def _save(self):
        M_B_to_A = build_affine(self.tx, self.ty, self.angle, self.sx, self.sy)
        M_A_to_B = invert_affine(M_B_to_A)

        img_a_path, img_b_path = self.pairs[self.current_idx]

        data = {
            "image_A_path": str(img_a_path),
            "image_B_path": str(img_b_path),
            "affine_matrix_A_to_B": M_A_to_B.tolist(),
            "image_a_shape": list(self.imgA_shape),
            "image_b_shape": list(self.imgB_shape),
            "annotator": self.annotator_name,
            "annotation_date": datetime.datetime.now().isoformat(),
        }

        gt_file = self.annotations_dir / f"{self.pair_id}_groundtruth.json"
        with open(gt_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {gt_file}")

    def run(self):
        while True:
            # Check if the user clicked the OS window close button (X)
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
                
            key = cv2.waitKey(100) & 0xFF

            # --- Frozen: waiting for auto-aligner to finish ---
            if not self._auto_applied:
                if key == 27:   # always allow quit
                    break
                # Check if the aligner has now finished
                if not self.aligner.is_processing(self.pair_id) and \
                        self.pair_id in self.aligner.results:
                    M_auto = self.aligner.results[self.pair_id]
                    if M_auto is not None:
                        M_B_to_A = invert_affine(M_auto)
                        self.tx, self.ty, self.angle, self.sx, self.sy = decompose_affine(M_B_to_A)
                    else:
                        self._reset_alignment()
                    self._auto_applied = True
                self._render()
                continue   # skip all other key handling while frozen

            # Periodically re-render to update auto-align status text
            if not self.dragging and key == 0xFF:
                self._render()

            if key == 27: # ESC
                break
            elif key == ord('s'):
                self.save_state = "saving"
                self._render()
                self._save()
                self.save_state = "saved"
                self.save_time = time.time()
                self._render()
            elif key == ord('a') or key == 81: # Left arrow
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self._load_pair()
            elif key == ord('d') or key == 83: # Right arrow
                if self.current_idx < len(self.pairs) - 1:
                    self.current_idx += 1
                    self._load_pair()
            elif key == ord('m'):
                self.render_mode = (self.render_mode + 1) % 2
                self._render()
            elif key == ord('r'):
                self._reset_alignment()
                self._render()
                
        self.aligner.shutdown()
        cv2.destroyAllWindows()
        # On Windows, OpenCV sometimes needs a few extra waitKey cycles to process the destroy event
        for _ in range(4):
            cv2.waitKey(1)
