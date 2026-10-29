from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio


HOUSTON_CLASS_NAMES = [
    "grass healthy",
    "grass stressed",
    "trees",
    "water",
    "residential buildings",
    "non-residential buildings",
    "road",
]


@dataclass(frozen=True)
class SceneData:
    image: np.ndarray
    ground_truth: np.ndarray | None
    class_names: list[str]
    # Per-pixel log spectral magnitude, standardized per scene (label-free).
    amplitude: np.ndarray | None = None


def _load_mat(path: Path) -> dict:
    """Read a .mat written by MATLAB v7.2 (scipy) or v7.3/HDF5 (h5py)."""
    try:
        return {k: v for k, v in sio.loadmat(str(path)).items() if not k.startswith("__")}
    except (NotImplementedError, ValueError):
        pass
    with h5py.File(path, "r") as handle:
        return {k: np.asarray(handle[k][:]) for k in handle.keys()}


def _pick(mat: dict, names: tuple[str, ...], ndim: int):
    for key in names:
        value = mat.get(key)
        if isinstance(value, np.ndarray) and value.ndim == ndim:
            return value
    for value in mat.values():
        if isinstance(value, np.ndarray) and value.ndim == ndim:
            return value
    return None


def _read_mat_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing hyperspectral image: {path}")
    cube = _pick(_load_mat(path), ("ori_data", "data", "image", "HSI"), 3)
    if cube is None:
        raise KeyError(f"{path} contains no 3-D hyperspectral cube.")
    # The band axis is the shortest one.  Houston arrives as (48, 954, 210)
    # through h5py; Pavia arrives as (610, 340, 102) through scipy.  Both are
    # normalized to (height, width, bands) here.
    # 判断 layout：
    # - 若 shape[0] 同时小于 shape[1] 和 shape[2] → 认为是 [C, H, W]，转成 [H, W, C]
    # - 否则认为是 [H, W, C]，保持不变
    # 这样能同时兼容：Houston(48,954,210) / Pavia(610,340,102) / IndianPines(145,145,200) / Salinas(512,217,200)
    if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
        cube = cube.transpose(1, 2, 0)
    return np.asarray(cube, dtype=np.float32)


def _read_mat_ground_truth(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing ground-truth map: {path}")
    label_map = _pick(_load_mat(path), ("map", "gt", "ground_truth", "label"), 2)
    if label_map is None:
        raise KeyError(f"{path} contains no 2-D ground-truth map.")
    return np.asarray(label_map, dtype=np.int64)


def _valid_pixel_mask(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image).all(axis=-1)
    energy = np.linalg.norm(np.nan_to_num(image, copy=False), axis=-1)
    return finite & (energy > 0)


def robust_scene_normalize(image: np.ndarray, mode: str = "l2") -> np.ndarray:
    """Normalize each spectrum without scene-wise band standardization.

    Cross-scene Houston transfer is sensitive to the relative shape of the
    reflectance curve.  Per-band median/IQR normalization performed separately
    in the two scenes can rotate that shape and was observed to amplify the
    grass/tree and building/road confusion.  The self-contained recovery path
    therefore uses only finite-value cleanup, one global positive scale and
    per-pixel L2 normalization.  No labels are used.
    """
    image = np.asarray(image, dtype=np.float32).copy()
    image[~np.isfinite(image)] = 0.0
    valid = _valid_pixel_mask(image)
    if not valid.any():
        raise ValueError("The scene contains no finite non-zero spectra.")

    if mode == "median":
        # Divide by one scene-level scalar instead of the per-pixel norm, so
        # relative albedo survives.  Pavia's Bitumen / Bricks / BareSoil are
        # near-parallel in shape (cosine 0.98-1.00) and separable only by
        # magnitude; the median makes that magnitude comparable across scenes.
        norms = np.linalg.norm(image, axis=-1, keepdims=True)
        scale = float(np.median(norms[valid]))
        image = image / max(scale, 1e-6)
    elif mode == "band":
        # Literature-style normalization (MSDA / TAPDA / SCLUDA): z-score each
        # band over the whole cube.  Unlike the per-pixel L2 norm this keeps the
        # relative brightness of every pixel, which is the only cue separating
        # near-parallel spectra.
        flat = image.reshape(-1, image.shape[-1])
        mask = valid.reshape(-1)
        mean = flat[mask].mean(axis=0, keepdims=True)
        std = flat[mask].std(axis=0, keepdims=True)
        flat = (flat - mean) / np.maximum(std, 1e-6)
        image = flat.reshape(image.shape).astype(np.float32)
    elif mode == "scale":
        # Literature protocol (sklearn.preprocessing.scale over ALL pixels,
        # background included).  Background is deliberately NOT zeroed so that
        # patch content matches the official MSDA/TAPDA pipeline.
        flat = image.reshape(-1, image.shape[-1])
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)
        image = ((flat - mean) / np.maximum(std, 1e-6)).reshape(image.shape)
        return np.asarray(image, dtype=np.float32)
    else:
        global_scale = float(np.max(np.abs(image[valid])))
        if global_scale > 0.0:
            image /= global_scale
        norms = np.linalg.norm(image, axis=-1, keepdims=True)
        image = image / np.maximum(norms, 1e-6)
    image[~valid] = 0.0
    return np.asarray(image, dtype=np.float32)


def compute_log_amplitude(image: np.ndarray) -> np.ndarray:
    """Per-pixel log spectral magnitude, robustly standardized per scene.

    Per-pixel L2 normalization removes magnitude, which is the only cue that
    separates near-parallel spectra (Pavia Bitumen / Bricks / BareSoil).
    The map is computed from the RAW spectrum, then centered by the scene
    median and scaled by the MAD so source and target become comparable.
    No labels are used.
    """
    image = np.asarray(image, dtype=np.float32)
    valid = _valid_pixel_mask(image)
    if not valid.any():
        raise ValueError("The scene contains no finite non-zero spectra.")
    norms = np.linalg.norm(np.nan_to_num(image, copy=False), axis=-1)
    amplitude = np.log(np.maximum(norms, 1e-6))
    values = amplitude[valid]
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, 1e-3)
    amplitude = (amplitude - center) / scale
    amplitude[~valid] = 0.0
    return np.clip(amplitude, -5.0, 5.0).astype(np.float32)


def align_target_bands(
    source_image: np.ndarray,
    target_image: np.ndarray,
    strength: float = 0.35,
    scale_clip: float = 0.20,
    shift_clip: float = 0.08,
    renormalize: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mild label-free robust band alignment.

    The target cube is nudged toward source robust band statistics.  The bounded
    interpolation avoids erasing target-domain structure and is independent of
    target labels.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return target_image.copy(), {
            "mean_abs_scale_delta": 0.0,
            "mean_abs_shift_delta": 0.0,
        }

    src_valid = _valid_pixel_mask(source_image)
    tgt_valid = _valid_pixel_mask(target_image)
    src = source_image[src_valid]
    tgt = target_image[tgt_valid]
    if src.size == 0 or tgt.size == 0:
        return target_image.copy(), {
            "mean_abs_scale_delta": 0.0,
            "mean_abs_shift_delta": 0.0,
        }

    src_median = np.median(src, axis=0)
    tgt_median = np.median(tgt, axis=0)
    src_iqr = np.percentile(src, 75.0, axis=0) - np.percentile(src, 25.0, axis=0)
    tgt_iqr = np.percentile(tgt, 75.0, axis=0) - np.percentile(tgt, 25.0, axis=0)
    src_iqr = np.maximum(src_iqr, 1e-4)
    tgt_iqr = np.maximum(tgt_iqr, 1e-4)

    raw_scale = src_iqr / tgt_iqr
    scale_delta = np.clip(raw_scale - 1.0, -scale_clip, scale_clip)
    shift_delta = np.clip(src_median - tgt_median, -shift_clip, shift_clip)
    scale = 1.0 + strength * scale_delta
    shift = strength * shift_delta

    aligned = target_image * scale.reshape(1, 1, -1) + shift.reshape(1, 1, -1)
    aligned[~tgt_valid] = 0.0
    if renormalize:
        norms = np.linalg.norm(aligned, axis=-1, keepdims=True)
        aligned = aligned / np.maximum(norms, 1e-6)
    aligned[~tgt_valid] = 0.0
    diagnostics = {
        "mean_abs_scale_delta": float(np.mean(np.abs(strength * scale_delta))),
        "mean_abs_shift_delta": float(np.mean(np.abs(strength * shift_delta))),
    }
    return np.asarray(aligned, dtype=np.float32), diagnostics


def load_source_scene(data_path: str | Path, name: str = "Houston13", num_classes: int = 7, normalize_mode: str = "l2") -> SceneData:
    root = Path(data_path)
    # WHU-Hi dataset support
    if name.lower() in ['longkou', 'honghu']:
        image = load_whu_image(str(root), name)
        ground_truth = load_whu_gt(str(root), name, num_classes)
        image = robust_scene_normalize(image, mode=normalize_mode)
        return SceneData(image=image, ground_truth=ground_truth, class_names=[f"Class{i}" for i in range(1, num_classes+1)])
    image = _read_mat_image(root / f"{name}.mat")
    ground_truth = _read_mat_ground_truth(root / f"{name}_{num_classes}gt.mat")
    if image.shape[:2] != ground_truth.shape:
        raise ValueError(
            f"Source image and label shapes differ: {image.shape[:2]} vs {ground_truth.shape}."
        )
    amplitude = compute_log_amplitude(image)
    image = robust_scene_normalize(image, mode=normalize_mode)
    return SceneData(image=image, ground_truth=ground_truth,
                     class_names=HOUSTON_CLASS_NAMES.copy(), amplitude=amplitude)


def load_target_image_only(data_path: str | Path, name: str = "Houston18", normalize_mode: str = "l2") -> SceneData:
    root = Path(data_path)
    # WHU-Hi dataset support
    if name.lower() in ['longkou', 'honghu']:
        image = load_whu_image(str(root), name)
        image = robust_scene_normalize(image, mode=normalize_mode)
        return SceneData(image=image, ground_truth=None, class_names=[])
    raw_image = _read_mat_image(root / f"{name}.mat")
    amplitude = compute_log_amplitude(raw_image)
    image = robust_scene_normalize(raw_image, mode=normalize_mode)
    return SceneData(image=image, ground_truth=None,
                     class_names=HOUSTON_CLASS_NAMES.copy(), amplitude=amplitude)


def load_target_ground_truth_only(data_path: str | Path, name: str = "Houston18", num_classes: int = 7) -> np.ndarray:
    # WHU-Hi dataset support
    if name.lower() in ['longkou', 'honghu']:
        return load_whu_gt(str(data_path), name, num_classes)
    return _read_mat_ground_truth(Path(data_path) / f"{name}_{num_classes}gt.mat")


def stratified_source_split(
    ground_truth: np.ndarray,
    train_ratio: float,
    seed: int,
    samples_per_class: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = np.zeros_like(ground_truth, dtype=np.int64)
    validation = np.zeros_like(ground_truth, dtype=np.int64)
    classes = [int(c) for c in np.unique(ground_truth) if int(c) > 0]
    for class_id in classes:
        coordinates = np.argwhere(ground_truth == class_id)
        rng.shuffle(coordinates)
        if samples_per_class > 0:
            split = min(int(samples_per_class), max(len(coordinates) - 1, 0))
            split = max(split, 1) if len(coordinates) > 1 else len(coordinates)
        elif len(coordinates) <= 1:
            split = len(coordinates)
        else:
            split = int(round(len(coordinates) * train_ratio))
            split = min(max(split, 1), len(coordinates) - 1)
        train_coords = coordinates[:split]
        val_coords = coordinates[split:]
        if len(train_coords):
            train[train_coords[:, 0], train_coords[:, 1]] = class_id
        if len(val_coords):
            validation[val_coords[:, 0], val_coords[:, 1]] = class_id
    return train, validation


def budgeted_source_split(ground_truth: np.ndarray, total_budget_ratio: float,
                          seed: int, samples_per_class: int = 0,
                          train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    """Only budgeted SOURCE labels can be used for fitting AND model selection.

    A ratio of 0.05 means at most 5% of source-labeled pixels per class in
    train+validation combined (rounding at most one pixel per class). Training
    and validation use disjoint SOURCE coordinates. All excluded pixels become
    unlabeled zeros and never reach either source loader.
    """
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must lie in (0,1]")
    if samples_per_class <= 0 and not 0 < total_budget_ratio <= 1:
        raise ValueError("total_budget_ratio must be in (0,1]")
    rng = np.random.default_rng(seed)
    train = np.zeros_like(ground_truth, dtype=np.int64)
    validation = np.zeros_like(ground_truth, dtype=np.int64)
    for class_id in [int(i) for i in np.unique(ground_truth) if i > 0]:
        coordinates = np.argwhere(ground_truth == class_id)
        rng.shuffle(coordinates)
        budget = min(len(coordinates), (samples_per_class if samples_per_class > 0
                         else max(1, int(round(len(coordinates)*total_budget_ratio)))))
        # Two distinct samples are needed for a source validation split.
        if budget < 2:
            raise ValueError(f"class {class_id}: source label budget {budget} is too small for "
                             "disjoint train+validation; increase budget")
        if train_fraction >= 1.0:
            # Literature-aligned protocol: the whole budget is used for training
            # and the source validation split is empty.
            ntrain = budget
        else:
            ntrain = min(budget - 1, max(1, round(budget * train_fraction)))
        a, b = coordinates[:ntrain], coordinates[ntrain:budget]
        train[a[:,0],a[:,1]] = class_id
        validation[b[:,0],b[:,1]] = class_id
    return train, validation


def spatially_stratified_subset(
    coordinates: np.ndarray,
    maximum: int,
    seed: int,
    grid_size: int = 30,
    stratified_fraction: float = 0.80,
) -> np.ndarray:
    if maximum <= 0 or len(coordinates) <= maximum:
        return np.arange(len(coordinates), dtype=np.int64)
    rng = np.random.default_rng(seed)
    coordinates = np.asarray(coordinates, dtype=np.int64)
    rows, cols = coordinates[:, 0], coordinates[:, 1]
    cell_rows = rows // max(1, grid_size)
    cell_cols = cols // max(1, grid_size)
    keys = cell_rows.astype(np.int64) * (cell_cols.max(initial=0) + 1) + cell_cols
    groups: dict[int, list[int]] = {}
    for index, key in enumerate(keys.tolist()):
        groups.setdefault(int(key), []).append(index)

    stratified_count = int(round(maximum * float(np.clip(stratified_fraction, 0.0, 1.0))))
    selected: list[int] = []
    non_empty = list(groups.values())
    rng.shuffle(non_empty)
    while len(selected) < stratified_count:
        progressed = False
        for group in non_empty:
            if not group:
                continue
            pick_pos = int(rng.integers(0, len(group)))
            selected.append(group.pop(pick_pos))
            progressed = True
            if len(selected) >= stratified_count:
                break
        if not progressed:
            break

    selected_set = set(selected)
    remaining = np.asarray([i for i in range(len(coordinates)) if i not in selected_set], dtype=np.int64)
    need = maximum - len(selected)
    if need > 0:
        extra = rng.choice(remaining, size=need, replace=False)
        selected.extend(int(i) for i in extra.tolist())
    selected = np.asarray(selected[:maximum], dtype=np.int64)
    selected.sort()
    return selected


def build_neighbor_ids(
    coordinates: np.ndarray,
    radius: int = 3,
    max_neighbors: int = 12,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.int64)
    n = len(coordinates)
    neighbors = np.full((n, max_neighbors), -1, dtype=np.int64)
    if n == 0 or max_neighbors <= 0:
        return neighbors

    cell_size = max(1, radius)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (row, col) in enumerate(coordinates.tolist()):
        buckets.setdefault((row // cell_size, col // cell_size), []).append(index)

    radius_sq = radius * radius
    for index, (row, col) in enumerate(coordinates.tolist()):
        cell = (row // cell_size, col // cell_size)
        candidates: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for other in buckets.get((cell[0] + dr, cell[1] + dc), []):
                    if other == index:
                        continue
                    rr, cc = coordinates[other]
                    distance_sq = int((rr - row) ** 2 + (cc - col) ** 2)
                    if distance_sq <= radius_sq:
                        candidates.append((distance_sq, other))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for slot, (_, other) in enumerate(candidates[:max_neighbors]):
            neighbors[index, slot] = other
    return neighbors


class PatchDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        coordinates: np.ndarray,
        patch_size: int,
        labels: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd.")
        self.patch_size = int(patch_size)
        self.radius = self.patch_size // 2
        self.image = np.pad(
            np.asarray(image, dtype=np.float32),
            ((self.radius, self.radius), (self.radius, self.radius), (0, 0)),
            mode="symmetric",
        )
        self.coordinates = np.asarray(coordinates, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.augment = bool(augment)
        if self.labels is not None and len(self.labels) != len(self.coordinates):
            raise ValueError("labels and coordinates must have the same length.")

    def __len__(self) -> int:
        return len(self.coordinates)

    @staticmethod
    def _augment_patch(patch: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            patch = np.flip(patch, axis=0)
        if np.random.random() < 0.5:
            patch = np.flip(patch, axis=1)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.92, 1.08)
            noise = np.random.normal(0.0, 0.015, size=patch.shape).astype(np.float32)
            patch = scale * patch + noise
        # Frequency-domain augmentation: simulate sensor/illumination variance
        if np.random.random() < 0.4:
            patch = PatchDataset._frequency_augment(patch)
        return patch
    
    @staticmethod
    def _frequency_augment(patch: np.ndarray) -> np.ndarray:
        """Fourier-domain perturbation for cross-scene robustness.
        
        Perturbs phase (sensor noise) and amplitude (radiometric response) in
        spectral dimension to synthesize domain-shifted samples without labels.
        """
        # patch shape: (H, W, C) where C is spectral bands
        H, W, C = patch.shape
        # Work on flattened spatial (H*W, C) for per-pixel spectrum FFT
        flat = patch.reshape(-1, C).astype(np.float32)
        
        # Real FFT along spectral axis (C dimension)
        spectrum_fft = np.fft.rfft(flat, axis=1)  # (H*W, C//2+1) complex
        
        # Phase perturbation: simulate sensor registration/calibration noise
        phase_noise_std = 0.08
        phase_noise = np.random.randn(*spectrum_fft.shape).astype(np.float32) * phase_noise_std
        magnitude = np.abs(spectrum_fft)
        phase = np.angle(spectrum_fft) + phase_noise
        spectrum_fft = magnitude * np.exp(1j * phase)
        
        # Amplitude perturbation: simulate radiometric response variance
        # Stronger at high freq (sensor-specific), weaker at low freq (physical)
        freq_count = spectrum_fft.shape[1]
        freq_weights = np.linspace(0.02, 0.10, freq_count).astype(np.float32)
        amp_noise = 1.0 + np.random.randn(*spectrum_fft.shape).astype(np.float32) * freq_weights[None, :]
        spectrum_fft = spectrum_fft * amp_noise
        
        # Inverse FFT back to spatial-spectral
        augmented_flat = np.fft.irfft(spectrum_fft, n=C, axis=1).real.astype(np.float32)
        augmented = augmented_flat.reshape(H, W, C)
        
        return augmented

    def __getitem__(self, index: int):
        row, col = self.coordinates[index]
        row += self.radius
        col += self.radius
        r0, r1 = row - self.radius, row + self.radius + 1
        c0, c1 = col - self.radius, col + self.radius + 1
        patch = np.asarray(self.image[r0:r1, c0:c1], dtype=np.float32)
        if self.augment:
            patch = self._augment_patch(patch)
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
        tensor = torch.from_numpy(patch)
        if self.labels is None:
            return tensor, torch.tensor(index, dtype=torch.long)
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


def coordinates_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(np.asarray(mask, dtype=bool)).astype(np.int64)


def labels_at_coordinates(ground_truth: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.int64)
    return ground_truth[coordinates[:, 0], coordinates[:, 1]].astype(np.int64)


# ========== WHU-Hi Dataset Support ==========

def load_whu_image(data_path: str, scene_name: str) -> np.ndarray:
    """Load WHU-Hi image (LongKou or HongHu).
    
    Returns: (H, W, C) float32 array, unnormalized reflectance.
    """
    scene_map = {
        'longkou': 'LongKou/longkou_image.npy',
        'honghu': 'HongHu/honghu_image.npy',
    }
    if scene_name.lower() not in scene_map:
        raise ValueError(f"Unknown WHU-Hi scene: {scene_name}. Choose 'longkou' or 'honghu'.")
    
    path = Path(data_path) / scene_map[scene_name.lower()]
    if not path.exists():
        raise FileNotFoundError(f"WHU-Hi image not found: {path}")
    
    image = np.load(path)
    return image.astype(np.float32)


def load_whu_gt(data_path: str, scene_name: str, num_classes: int) -> np.ndarray:
    """Load WHU-Hi ground truth.
    
    Args:
        num_classes: 9 (shared task) or 10 (LongKou full) or 23 (HongHu full)
    
    Returns: (H, W) int array, 0=background, 1-N=classes.
    """
    scene_map_9cls = {
        'longkou': 'LongKou/longkou_gt_9cls.npy',
        'honghu': 'HongHu/honghu_gt_9cls.npy',
    }
    scene_map_full = {
        'longkou': 'LongKou/longkou_gt.npy',
        'honghu': 'HongHu/honghu_gt.npy',
    }
    
    scene_name_lower = scene_name.lower()
    if num_classes == 9:
        gt_file = scene_map_9cls.get(scene_name_lower)
    else:
        gt_file = scene_map_full.get(scene_name_lower)
    
    if gt_file is None:
        raise ValueError(f"Unknown WHU-Hi scene: {scene_name}")
    
    path = Path(data_path) / gt_file
    if not path.exists():
        raise FileNotFoundError(f"WHU-Hi GT not found: {path}")
    
    gt = np.load(path)
    return gt.astype(np.int64)
