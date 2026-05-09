#
# This file is adapted from the data augmentation logic originally implemented in
# OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow), refactored into a standalone
# module with a unified augmentation interface, and added BeamDropout to simulate
# LiDAR beam-level sparsity.
#

from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, List, Optional, Union
import numpy as np
import torch

try:
    from omegaconf import OmegaConf, DictConfig, ListConfig
except Exception:
    OmegaConf, DictConfig, ListConfig = None, dict, list


FLOW_ALIGNED_KEYS = [
    "flow", "flow_is_valid", "flow_category_indices",
    "flow_instance_id", "dufo", "eval_mask"
]

def _to_py(obj):
    if OmegaConf is None:
        return obj
    try:
        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return obj

def _as_pair(x, cast=float):
    x = _to_py(x)
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        assert len(x) == 2, f"Expected pair, got {x}"
        return (cast(x[0]), cast(x[1]))
    return (cast(x), cast(x))

def _section(cfg, name: str):
    if cfg is None:
        return {}
    node = cfg.get(name, {}) if hasattr(cfg, "get") else {}
    out = _to_py(node)
    return out or {}

def _gm_key(pc_key: str) -> str:
    return f"gm{pc_key[2:]}"

def _dyn_key(pc_key: str) -> str:
    return f"{pc_key}_dynamic"

def _beam_key(pc_key: str) -> Optional[str]:
    if pc_key == "pc0": return "beam0"
    if pc_key == "pc1": return "beam1"
    if pc_key.startswith("pch"):
        return f"beamh{pc_key[3:]}"
    return None

def _point_frame_keys(d: Dict[str, Any]) -> List[str]:
    return [k for k in d.keys() if k.startswith("pc") and not k.endswith("dynamic")]

def _apply_mask_single_frame(d: Dict[str, Any], pc_key: str, mask: np.ndarray) -> None:
    d[pc_key] = d[pc_key][mask]

    gk = _gm_key(pc_key)
    if gk in d: d[gk] = d[gk][mask]

    dk = _dyn_key(pc_key)
    if dk in d: d[dk] = d[dk][mask]

    bk = _beam_key(pc_key)
    if bk in d: d[bk] = d[bk][mask]

    if pc_key == "pc0":
        for k in FLOW_ALIGNED_KEYS:
            if k in d:
                d[k] = d[k][mask]


class Compose:
    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_transforms(aug_cfg: Union[Dict[str, Any], DictConfig]) -> Compose:
    steps = []
    used = []

    bd = _section(aug_cfg, "beam_dropout")
    if bd.get("enable", False):
        steps.append(
            BeamDropout(
                p=float(bd.get("p", 0.8)),
                strategy=str(bd.get("strategy", "half")),
                drop_ratio_range=_as_pair(bd.get("drop_ratio_range", (0.05, 0.40)), float),
                min_points=int(bd.get("min_points", 512)),
                verbose=bool(bd.get("verbose", False)),
            )
        )
        used.append("BeamDropout")

    if bool(aug_cfg.get("basic_augment", False)):
        steps.append(RandomHeight(p=0.8))
        steps.append(RandomFlip(p=0.2))
        steps.append(RandomJitter())
        used.append("RandomHeight")
        used.append("RandomFlip")
        used.append("RandomJitter")

    steps.append(ToTensor())
    used.append("ToTensor")

    print(f"[Augmentations] Enabled: {', '.join(used)}")

    return Compose(steps)

class RandomJitter:
    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        assert clip > 0
        self.sigma = float(sigma)
        self.clip = float(clip)

    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in d.items():
            if k.startswith("pc") and not k.endswith("dynamic"):
                if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= 3:
                    jitter = np.clip(self.sigma * np.random.randn(v.shape[0], 3), -self.clip, self.clip)
                    v[:, :3] += jitter
        return d

class RandomFlip(object):
    def __init__(self, p=0.5, verbose=False):
        """p: probability of flipping"""
        self.p = p
        self.verbose = verbose

    def __call__(self, data_dict):
        flip_x = np.random.rand() < self.p
        flip_y = np.random.rand() < self.p

        # If no flip, return directly
        if not (flip_x or flip_y):
            return data_dict
        
        for key in data_dict.keys():
            if (key.startswith("pc") or (key.startswith("flow") and data_dict[key].dtype == np.float32)) and not key.endswith("dynamic"):
                if flip_x:
                    data_dict[key][:, 0] = -data_dict[key][:, 0]
                if flip_y:
                    data_dict[key][:, 1] = -data_dict[key][:, 1]
            if key.startswith("pose"):
                if flip_x:
                    pose = data_dict[key].copy()
                    pose[:, 0] *= -1
                    data_dict[key] = pose
                if flip_y:
                    pose = data_dict[key].copy()
                    pose[:, 1] *= -1
                    data_dict[key] = pose

        if "ego_motion" in data_dict:
            # need recalculate the ego_motion
            data_dict["ego_motion"] = np.linalg.inv(data_dict['pose1']) @ data_dict['pose0']
        if self.verbose:
            print(f"RandomFlip: flip_x={flip_x}, flip_y={flip_y}")
        return data_dict

class RandomHeight(object):
    def __init__(self, p=0.5, verbose=False):
        """p: probability of changing height"""
        self.p = p
        self.verbose = verbose

    def __call__(self, data_dict):
        # NOTE(Qingwen): The reason set -0.5 to 2.0 is because some dataset axis origin is around the ground level. (vehicle base etc.)
        random_height = np.random.uniform(-0.5, 2.0)
        if np.random.rand() < self.p:
            for key in data_dict.keys():
                if key.startswith("pc") and not key.endswith("dynamic"):
                    data_dict[key][:, 2] += random_height
            if self.verbose:
                print(f"RandomHeight: {random_height}")
        return data_dict

class BeamDropout:
    """
    Drop out entire LiDAR beams (by beam_id) to simulate realistic sparsity.

    Strategies:
      - "half"   (default): randomly drop either all even or all odd beams.
      - "random": sample a set of beams to drop (using drop_ratio_range).

    Args:
        p (float): probability to apply.
        strategy (str): "half" (default) or "random".
        drop_ratio_range (tuple): only used if strategy="random".
        min_points (int): per-frame minimum points after dropout.
        verbose (bool): optional logging.
    """
    def __init__(
        self,
        p: float = 0.5,
        strategy: str = "half",
        drop_ratio_range: Tuple[float, float] = (0.25, 0.50),
        min_points: int = 512,
        verbose: bool = False,
    ):
        self.p = float(p)
        self.strategy = str(strategy).lower()
        self.drop_ratio_range = tuple(drop_ratio_range)
        self.min_points = int(min_points)
        self.verbose = bool(verbose)
        self._warned_missing_beamid = False
        assert self.strategy in ("half", "random"), f"Unknown strategy: {strategy}"
        assert 0.0 <= float(drop_ratio_range[0]) <= float(drop_ratio_range[1]) <= 1.0, f"drop_ratio_range must be within [0,1], got {drop_ratio_range}"

    def _choose_drop_ids(self, beams_union: np.ndarray) -> np.ndarray:
        if beams_union.size == 0:
            return np.array([], dtype=np.uint8)

        beams = np.array(sorted(beams_union.tolist()), dtype=np.uint8)

        if self.strategy == "half":
            parity = np.random.randint(0, 2)  # 0 = drop even, 1 = drop odd
            drop_ids = beams[beams % 2 == parity].astype(np.uint8, copy=False)
            if self.verbose:
                which = "even" if parity == 0 else "odd"
                print(f"[BeamDropout] half: drop {which} → {len(drop_ids)}/{len(beams)} beams")
            return drop_ids

        # strategy == "random"
        lo, hi = self.drop_ratio_range
        r = float(np.random.uniform(lo, hi))
        k = max(1, int(round(len(beams) * r)))
        k = max(0, min(k, len(beams)))
        if k == 0:
            return np.array([], dtype=np.uint8)
        drop_ids = np.random.choice(beams, size=k, replace=False).astype(np.uint8)
        if self.verbose:
            print(f"[BeamDropout] random: drop_ratio≈{r:.3f} → drop {k}/{len(beams)} beams")
        return drop_ids

    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() > self.p:
            return d

        pc_keys = _point_frame_keys(d)
        if not pc_keys:
            return d

        # Collect all beam ids that appear in this sample (across frames)
        beams_union: set = set()
        for pk in pc_keys:
            bk = _beam_key(pk)
            if bk in d and isinstance(d[bk], np.ndarray):
                beams_union.update(np.unique(d[bk]).tolist())
        if len(beams_union) == 0:
            if not self._warned_missing_beamid:
                print(
                    "[WARNING] beam_id not found in current dataset. "
                    "BeamDropout will be skipped for this dataset."
                )
                self._warned_missing_beamid = True
            return d

        drop_ids = self._choose_drop_ids(np.array(list(beams_union), dtype=np.uint8))
        if drop_ids.size == 0:
            return d

        for pk in pc_keys:
            bk = _beam_key(pk)
            if bk not in d or not isinstance(d[bk], np.ndarray):
                continue

            bid = d[bk]
            n = bid.shape[0]
            if n <= 1:
                continue

            keep = ~np.isin(bid, drop_ids)

            # Enforce a minimum number of points kept
            min_keep = min(self.min_points, n)
            if int(keep.sum()) < min_keep:
                deficit = min_keep - int(keep.sum())
                dropped_idx = np.where(~keep)[0]
                if dropped_idx.size > 0:
                    take_back = np.random.choice(
                        dropped_idx, size=min(deficit, dropped_idx.size), replace=False
                    )
                    keep[take_back] = True

            _apply_mask_single_frame(d, pk, keep)

        return d
    
class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in list(d.items()):
            if k in ["scene_id", "timestamp", "eval_flag"]:
                continue
            elif isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                if k.startswith("pose") or k == "ego_motion":
                    t = t.float()
                d[k] = t
            else:
                print(f"Warning: {k} is not a numpy array. Type: {type(d[k])}")
        return d
