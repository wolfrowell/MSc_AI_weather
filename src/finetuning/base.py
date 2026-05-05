"""Abstract base for GraphCast fine-tuning strategies.

To add a new method (LoRA, adapters, prefix tuning, …):
  1. Create a new file in src/finetuning/ (e.g. lora.py)
  2. Subclass FineTuningStrategy and implement the four abstract methods
  3. Pass an instance to GraphCastModel

The strategy is responsible for:
  - Knowing which params are trainable vs frozen
  - (Optionally) initializing extra parameters (e.g. LoRA matrices, adapter weights)
  - Building the Haiku predictor that embodies the method
  - Merging everything back into a single param tree for inference/eval
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import xarray
from graphcast import graphcast


@dataclass
class FineTuningState:
    """Carries all parameter groups needed by a strategy.

    trainable_params : updated by the optimizer every step
    frozen_params    : never touched after initialization
    extra_params     : strategy-specific additions (LoRA matrices, adapter weights, …)
                       empty dict for strategies that don't add new parameters
    model_state      : haiku state (batch norm running stats, etc.)
    """
    trainable_params: dict
    frozen_params: dict
    model_state: dict
    extra_params: dict = field(default_factory=dict)


class FineTuningStrategy(ABC):
    """Interface that every fine-tuning strategy must implement."""

    @abstractmethod
    def prepare(
        self,
        params: dict,
        state: dict,
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig,
    ) -> FineTuningState:
        """Partition (and optionally augment) params into a FineTuningState.

        Called once before training starts.
        """

    @abstractmethod
    def build_predictor(
        self,
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig,
        norm_kwargs: dict,
        for_training: bool,
    ) -> Any:
        """Return the Haiku predictor to use inside hk.transform_with_state.

        for_training=True  → may insert stop_gradient, adapters, LoRA branches, …
        for_training=False → vanilla forward pass for inference/eval
        """

    @abstractmethod
    def merge_for_inference(self, ft_state: FineTuningState) -> dict:
        """Produce a single merged param dict suitable for run_forward / eval."""

    @abstractmethod
    def print_summary(self, ft_state: FineTuningState) -> None:
        """Print a human-readable summary of what is trainable vs frozen."""
