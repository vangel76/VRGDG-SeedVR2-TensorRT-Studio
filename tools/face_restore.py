"""Tracked face restoration for decoded SeedVR2 batches (CodeFormer / GFPGAN via facexlib alignment).

Pipeline per frame: RetinaFace detection -> IoU track matching -> temporally smoothed 5-point
landmarks -> FFHQ-style 512 px alignment -> restorer -> detail-preserving blend back into the
frame under a parsed face mask. Track state persists across batches through a JSON file so the
21-frame batches of a render behave like one continuous clip.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FACE_MODELS = ("none", "codeformer", "gfpgan")
WEIGHTS = {
    "codeformer": ("codeformer.pth", "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"),
    "gfpgan": ("GFPGANv1.4.pth", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"),
    "detection": ("detection_Resnet50_Final.pth", "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"),
    "parsing": ("parsing_parsenet.pth", "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"),
}
FACE_SIZE = 512
# FFHQ 5-point template used by CodeFormer/GFPGAN for a 512 px crop (left eye, right eye, nose, mouth corners).
FACE_TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)
# Parsing classes kept in the paste-back mask (0 = background, 14 neck, 16 cloth, 17 hair, 18 hat are dropped).
MASK_CLASSES = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.float32)
DETECT_MAX_SIDE = 1600


# ----------------------------------------------------------------------------- tracking (pure)
def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


class FaceTracker:
    """Greedy IoU matching with a short landmark history per track for temporal smoothing."""

    def __init__(self, history: int = 5, max_missed: int = 2, iou_threshold: float = 0.3) -> None:
        self.history = history
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.tracks: dict[int, dict[str, Any]] = {}
        self.next_id = 1

    def to_state(self) -> dict[str, Any]:
        return {"next_id": self.next_id, "tracks": [{"id": tid, "bbox": t["bbox"].tolist(), "missed": t["missed"], "age": t["age"],
                                                     "landmarks": [lm.tolist() for lm in t["landmarks"]]} for tid, t in self.tracks.items()]}

    @classmethod
    def from_state(cls, state: dict[str, Any] | None, **kwargs) -> "FaceTracker":
        tracker = cls(**kwargs)
        if not state:
            return tracker
        tracker.next_id = int(state.get("next_id", 1))
        for track in state.get("tracks", []):
            tracker.tracks[int(track["id"])] = {
                "bbox": np.array(track["bbox"], dtype=np.float32), "missed": int(track.get("missed", 0)), "age": int(track.get("age", 0)),
                "landmarks": deque([np.array(lm, dtype=np.float32) for lm in track.get("landmarks", [])], maxlen=tracker.history),
            }
        return tracker

    def update(self, detections: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[int, np.ndarray, np.ndarray, int]]:
        """detections: [(bbox xyxy, landmarks 5x2)] -> [(track id, bbox, smoothed landmarks, age)]."""
        assigned: set[int] = set()
        results: list[tuple[int, np.ndarray, np.ndarray, int]] = []
        candidates = []
        for det_index, (bbox, _) in enumerate(detections):
            for tid, track in self.tracks.items():
                score = iou(bbox, track["bbox"])
                if score >= self.iou_threshold:
                    candidates.append((score, det_index, tid))
        candidates.sort(reverse=True)
        matched_dets: set[int] = set()
        for score, det_index, tid in candidates:
            if det_index in matched_dets or tid in assigned:
                continue
            matched_dets.add(det_index)
            assigned.add(tid)
            bbox, landmarks = detections[det_index]
            track = self.tracks[tid]
            track["bbox"] = bbox.astype(np.float32)
            track["missed"] = 0
            track["age"] += 1
            track["landmarks"].append(landmarks.astype(np.float32))
            results.append((tid, bbox, self._smoothed(track), track["age"]))
        for det_index, (bbox, landmarks) in enumerate(detections):
            if det_index in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": bbox.astype(np.float32), "missed": 0, "age": 1, "landmarks": deque([landmarks.astype(np.float32)], maxlen=self.history)}
            results.append((tid, bbox, landmarks.astype(np.float32), 1))
        for tid in list(self.tracks):
            if tid not in assigned and tid not in {r[0] for r in results}:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return results

    @staticmethod
    def _smoothed(track: dict[str, Any]) -> np.ndarray:
        """Weighted mean of recent landmarks, newest heaviest, so the crop glides instead of jittering."""
        history = list(track["landmarks"])
        weights = np.array([0.5 ** (len(history) - 1 - i) for i in range(len(history))], dtype=np.float32)
        weights /= weights.sum()
        return np.tensordot(weights, np.stack(history), axes=1).astype(np.float32)


# ----------------------------------------------------------------------------- models
def weights_path(name: str, weights_dir: Path) -> Path:
    return weights_dir / WEIGHTS[name][0]


def ensure_weights(names: list[str], weights_dir: Path) -> None:
    """Download any missing weight files (resumable, atomic rename)."""
    import urllib.request

    weights_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        filename, url = WEIGHTS[name]
        target = weights_dir / filename
        if target.exists() and target.stat().st_size > 1_000_000:
            continue
        print(f"Downloading {filename} ...", flush=True)
        part = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(target)


class FaceRestorer:
    def __init__(self, model: str, weights_dir: Path, device: torch.device, fidelity: float = 0.7) -> None:
        if model not in FACE_MODELS or model == "none":
            raise ValueError(f"Unknown face model: {model}")
        ensure_weights([model, "detection", "parsing"], weights_dir)
        from facexlib.detection import init_detection_model
        from facexlib.parsing import init_parsing_model

        self.model_name = model
        self.device = device
        self.fidelity = float(min(1.0, max(0.0, fidelity)))
        self.detector = init_detection_model("retinaface_resnet50", half=False, device=device, model_rootpath=str(weights_dir))
        self.parser = init_parsing_model(model_name="parsenet", device=device, model_rootpath=str(weights_dir))
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if model == "codeformer":
            from vendor.faces.codeformer.codeformer_arch import CodeFormer
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"])
            checkpoint = torch.load(weights_path(model, weights_dir), map_location="cpu", weights_only=False)
            net.load_state_dict(checkpoint["params_ema"])
        else:
            from vendor.faces.gfpgan.gfpganv1_clean_arch import GFPGANv1Clean
            net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None, fix_decoder=False,
                                num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)
            checkpoint = torch.load(weights_path(model, weights_dir), map_location="cpu", weights_only=False)
            net.load_state_dict(checkpoint.get("params_ema") or checkpoint["params"], strict=True)
        self.net = net.eval().to(device)

    # ----- detection
    @torch.no_grad()
    def detect(self, frame_rgb01: torch.Tensor, min_face: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """frame (3, H, W) in [0, 1] on device -> [(bbox xyxy, landmarks 5x2)] in frame pixels."""
        _, height, width = frame_rgb01.shape
        scale = min(1.0, DETECT_MAX_SIDE / max(height, width))
        image = frame_rgb01
        if scale < 1.0:
            image = F.interpolate(frame_rgb01.unsqueeze(0), scale_factor=scale, mode="area")[0]
        bgr = (image.flip(0).clamp(0, 1) * 255.0).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
        boxes = self.detector.detect_faces(bgr, 0.97)
        detections: list[tuple[np.ndarray, np.ndarray]] = []
        for box in boxes:
            box = np.asarray(box, dtype=np.float32)
            bbox = box[0:4] / scale
            landmarks = box[5:15].reshape(5, 2) / scale
            eye_distance = float(np.linalg.norm(landmarks[0] - landmarks[1]))
            face_side = float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            if face_side < min_face or eye_distance < min_face * 0.2:
                continue
            detections.append((bbox, landmarks))
        return detections

    # ----- restoration of one aligned crop
    @torch.no_grad()
    def restore_crop(self, crop_rgb01: np.ndarray) -> np.ndarray:
        """(512, 512, 3) float32 RGB in [0, 1] -> restored crop, same layout."""
        tensor = torch.from_numpy(crop_rgb01).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - 0.5) / 0.5
        try:
            if self.model_name == "codeformer":
                output = self.net(tensor, w=self.fidelity, adain=True)[0]
            else:
                output = self.net(tensor, return_rgb=False, weight=0.5)[0]
        except RuntimeError as exc:  # CUDA OOM etc.: fall back to the input crop
            print(f"Face restorer failed on one face ({exc}); keeping SeedVR2 pixels", flush=True)
            output = tensor
        output = ((output.clamp(-1, 1) + 1.0) * 0.5)[0]
        return output.permute(1, 2, 0).float().cpu().numpy()

    @torch.no_grad()
    def face_mask(self, restored_rgb01: np.ndarray) -> np.ndarray:
        """Soft paste-back mask (512, 512) from the parsing net on the restored crop, borders feathered."""
        tensor = torch.from_numpy(restored_rgb01).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - 0.5) / 0.5
        classes = self.parser(tensor)[0].argmax(dim=1)[0].cpu().numpy()
        mask = MASK_CLASSES[np.clip(classes, 0, len(MASK_CLASSES) - 1)]
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        border = 10
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        return mask.astype(np.float32)


# ----------------------------------------------------------------------------- blending
def detail_preserving_merge(restored: np.ndarray, original: np.ndarray, keep_detail: float, face_scale: float) -> np.ndarray:
    """Low band from the restorer (structure, skin), high band from SeedVR2's own pixels (texture).

    `face_scale` = crop pixels per frame pixel; the split radius grows with it so the band is
    defined in output-frame pixels, not crop pixels.
    """
    keep_detail = float(min(1.0, max(0.0, keep_detail)))
    if keep_detail <= 0:
        return restored
    sigma = max(0.8, 1.6 * face_scale)
    ksize = int(2 * math.ceil(3 * sigma) + 1)
    low_restored = cv2.GaussianBlur(restored, (ksize, ksize), sigma)
    low_original = cv2.GaussianBlur(original, (ksize, ksize), sigma)
    high_restored = restored - low_restored
    high_original = original - low_original
    return low_restored + keep_detail * high_original + (1.0 - keep_detail) * high_restored


def restore_frames(
    frames: torch.Tensor, restorer: FaceRestorer, tracker: FaceTracker, *, strength: float, keep_detail: float,
    min_face: int, true_height: int, true_width: int, min_track_age: int = 1,
) -> tuple[torch.Tensor, int]:
    """frames (T, 3, H, W) in [0, 1] on device, modified in place. Returns (frames, faces restored)."""
    strength = float(min(1.0, max(0.0, strength)))
    if strength <= 0:
        return frames, 0
    restored_faces = 0
    for index in range(frames.shape[0]):
        region = frames[index, :, :true_height, :true_width]
        detections = restorer.detect(region, min_face)
        frame_np: np.ndarray | None = None
        for track_id, bbox, landmarks, age in tracker.update(detections):
            if age < min_track_age:
                continue
            affine = cv2.estimateAffinePartial2D(landmarks, FACE_TEMPLATE, method=cv2.LMEDS)[0]
            if affine is None:
                continue
            if frame_np is None:
                frame_np = region.permute(1, 2, 0).contiguous().float().cpu().numpy()  # (h, w, 3) float32 RGB
            crop = cv2.warpAffine(frame_np, affine, (FACE_SIZE, FACE_SIZE), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
            restored = restorer.restore_crop(np.ascontiguousarray(crop, dtype=np.float32))
            face_scale = float(math.sqrt(abs(affine[0, 0] * affine[1, 1] - affine[0, 1] * affine[1, 0])))
            merged = detail_preserving_merge(restored, crop, keep_detail, face_scale)
            mask = restorer.face_mask(restored) * strength
            inverse = cv2.invertAffineTransform(affine)
            # Warp only into the bounding box of the crop's footprint to keep 4K frames cheap.
            corners = np.array([[0, 0, 1], [FACE_SIZE, 0, 1], [0, FACE_SIZE, 1], [FACE_SIZE, FACE_SIZE, 1]], dtype=np.float32) @ inverse.T
            x0, y0 = int(max(0, math.floor(corners[:, 0].min()))), int(max(0, math.floor(corners[:, 1].min())))
            x1, y1 = int(min(true_width, math.ceil(corners[:, 0].max()))), int(min(true_height, math.ceil(corners[:, 1].max())))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            shifted = inverse.copy()
            shifted[:, 2] -= (x0, y0)
            size = (x1 - x0, y1 - y0)
            warped = cv2.warpAffine(merged, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_mask = cv2.warpAffine(mask, shifted, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            target = frames[index, :, y0:y1, x0:x1]
            warped_t = torch.from_numpy(warped).permute(2, 0, 1).to(target.device, target.dtype)
            mask_t = torch.from_numpy(warped_mask).unsqueeze(0).to(target.device, target.dtype)
            frames[index, :, y0:y1, x0:x1] = (target * (1.0 - mask_t) + warped_t * mask_t).clamp(0, 1)
            restored_faces += 1
    return frames, restored_faces


def load_state(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_state(path: Path | None, tracker: FaceTracker) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracker.to_state()), encoding="utf-8")
