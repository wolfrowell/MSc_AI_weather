"""Fine-tuning by freezing entire GraphCast modules.

Trainable / frozen split is at the top-level Haiku module boundary
(grid2mesh_gnn, mesh_gnn, mesh2grid_gnn). No new parameters are added.

Typical configurations
----------------------
Encoder only  : trainable=["grid2mesh_gnn"], frozen=["mesh_gnn","mesh2grid_gnn"]
                freeze_encoder=False, freeze_processor=False
                (grads must flow through processor+decoder → higher memory)

Decoder only  : trainable=["mesh2grid_gnn"], frozen=["grid2mesh_gnn","mesh_gnn"]
                freeze_encoder=True, freeze_processor=True
                (stop_gradient after encoder+processor → lower memory)
"""

from typing import Any

import jax
import numpy as np
import xarray
from graphcast import autoregressive, casting, graphcast, normalization

from .base import FineTuningState, FineTuningStrategy


_ALL_MODULES = ["grid2mesh_gnn", "mesh_gnn", "mesh2grid_gnn"]

_PRESETS = {
    "encoder": {
        "trainable_modules": ["grid2mesh_gnn"],
        "freeze_encoder": False,
        "freeze_processor": False,
    },
    "processor": {
        "trainable_modules": ["mesh_gnn"],
        "freeze_encoder": True,
        "freeze_processor": False,
    },
    "decoder": {
        "trainable_modules": ["mesh2grid_gnn"],
        "freeze_encoder": True,
        "freeze_processor": True,
    },
}


class FrozenModuleStrategy(FineTuningStrategy):
    """Freeze entire GraphCast modules; train the rest with stop_gradient shortcuts.

    Parameters
    ----------
    trainable_modules:
        List of top-level Haiku module names to keep trainable.
        Everything else is frozen.
    freeze_encoder / freeze_processor:
        Insert jax.lax.stop_gradient after those module outputs during training.
        Set to True only when the corresponding module is NOT in trainable_modules,
        to avoid cutting the gradient path to the modules you want to train.
    preset:
        Shortcut — one of "encoder", "processor", "decoder". Overrides the
        trainable_modules / freeze_encoder / freeze_processor arguments.
    """

    def __init__(
        self,
        trainable_modules: list[str] = None,
        freeze_encoder: bool = True,
        freeze_processor: bool = True,
        preset: str = None,
    ):
        if preset is not None:
            if preset not in _PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Choose from {list(_PRESETS)}")
            cfg = _PRESETS[preset]
            trainable_modules = cfg["trainable_modules"]
            freeze_encoder = cfg["freeze_encoder"]
            freeze_processor = cfg["freeze_processor"]

        if trainable_modules is None:
            trainable_modules = ["mesh2grid_gnn"]

        unknown = set(trainable_modules) - set(_ALL_MODULES)
        if unknown:
            raise ValueError(f"Unknown module names: {unknown}. Valid: {_ALL_MODULES}")

        self.trainable_modules = trainable_modules
        self.frozen_modules = [m for m in _ALL_MODULES if m not in trainable_modules]
        self.freeze_encoder = freeze_encoder
        self.freeze_processor = freeze_processor

    # ── FineTuningStrategy interface ──────────────────────────────────────────

    def prepare(self, params, state, model_config, task_config) -> FineTuningState:
        trainable, frozen = self._partition(params)
        return FineTuningState(
            trainable_params=trainable,
            frozen_params=frozen,
            model_state=state,
        )

    def build_predictor(self, model_config, task_config, norm_kwargs, for_training=True):
        fe = self.freeze_encoder if for_training else False
        fp = self.freeze_processor if for_training else False
        predictor = graphcast.GraphCast(
            model_config, task_config,
            freeze_encoder=fe,
            freeze_processor=fp,
        )
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **norm_kwargs)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    def merge_for_inference(self, ft_state: FineTuningState) -> dict:
        return _merge(ft_state.trainable_params, ft_state.frozen_params)

    def print_summary(self, ft_state: FineTuningState) -> None:
        stats = self._count_params(ft_state)
        n_t, n_f, n_tot = stats["trainable"], stats["frozen"], stats["total"]
        print("=" * 60)
        print(f"Strategy : FrozenModuleStrategy")
        print(f"Trainable: {', '.join(self.trainable_modules)}")
        print(f"Frozen   : {', '.join(self.frozen_modules)}")
        print("=" * 60)
        print(f"Trainable parameters : {n_t:,}  ({100*n_t/n_tot:.1f}%)")
        print(f"Frozen parameters    : {n_f:,}  ({100*n_f/n_tot:.1f}%)")
        print(f"Total parameters     : {n_tot:,}")
        print("=" * 60)
        print(f"stop_gradient after encoder  : {self.freeze_encoder}")
        print(f"stop_gradient after processor: {self.freeze_processor}")
        print("=" * 60)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _partition(self, params: dict) -> tuple[dict, dict]:
        trainable, frozen = {}, {}
        for path, sub in params.items():
            top = path.split("/")[0]
            if top in self.trainable_modules:
                trainable[path] = sub
            else:
                frozen[path] = sub
        return trainable, frozen

    @staticmethod
    def _count_params(ft_state: FineTuningState) -> dict:
        def _n(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return sum(int(np.prod(l.shape)) for l in leaves if hasattr(l, "shape"))
        n_t = _n(ft_state.trainable_params)
        n_f = _n(ft_state.frozen_params)
        return {"trainable": n_t, "frozen": n_f, "total": n_t + n_f}


def _merge(trainable: dict, frozen: dict) -> dict:
    merged = {}
    for key in set(frozen) | set(trainable):
        if key in frozen and key in trainable:
            merged[key] = {**frozen[key], **trainable[key]}
        elif key in frozen:
            merged[key] = frozen[key]
        else:
            merged[key] = trainable[key]
    return merged
