"""Training loop for GraphCast — strategy-agnostic."""

import gc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray

from .finetuning.base import FineTuningState
from .model import GraphCastModel


@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 1e-3
    log_every: int = 1
    seed: int = 0


class Trainer:
    """Trains whatever the strategy marks as trainable, leaving everything else frozen."""

    def __init__(self, model: GraphCastModel, config: TrainingConfig = None):
        self.model = model
        self.config = config or TrainingConfig()

    def fit(
        self,
        ft_state: FineTuningState,
        train_inputs: xarray.Dataset,
        train_targets: xarray.Dataset,
        train_forcings: xarray.Dataset,
    ) -> tuple[FineTuningState, list[float]]:
        """Runs the training loop.

        Returns:
            (updated_ft_state, loss_history)
        """
        optimizer = optax.adam(self.config.learning_rate)
        opt_state = optimizer.init(ft_state.trainable_params)
        current = ft_state
        losses: list[float] = []

        for epoch in range(self.config.num_epochs):
            loss, diagnostics, current, grads = self.model.compute_grads(
                current, train_inputs, train_targets, train_forcings
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            new_trainable = optax.apply_updates(current.trainable_params, updates)
            current = FineTuningState(
                trainable_params=new_trainable,
                frozen_params=current.frozen_params,
                model_state=current.model_state,
                extra_params=current.extra_params,
            )

            losses.append(float(loss))
            if (epoch + 1) % self.config.log_every == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: Loss = {loss:.6f}")

        print(f"\nTraining complete. Final loss: {losses[-1]:.6f}")
        return current, losses

    def print_gradient_stats(
        self,
        ft_state: FineTuningState,
        train_inputs: xarray.Dataset,
        train_targets: xarray.Dataset,
        train_forcings: xarray.Dataset,
    ) -> None:
        loss, _, _, grads = self.model.compute_grads(
            ft_state, train_inputs, train_targets, train_forcings
        )
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        norms = [float(jnp.abs(g).mean()) for g in flat_grads if hasattr(g, "shape")]
        print("=" * 60)
        print("GRADIENT STATISTICS")
        print("=" * 60)
        print(f"Loss        : {loss:.6f}")
        print(f"Mean |grad| : {np.mean(norms):.6f}" if norms else "No gradients found")
        print(f"Grad tensors: {len(flat_grads)}")
        print("=" * 60)

    @staticmethod
    def free_memory(*var_names: str, globs: dict = None) -> None:
        if globs is None:
            import __main__
            globs = vars(__main__)
        for name in var_names:
            globs.pop(name, None)
        gc.collect()
        jax.clear_caches()
        print("GPU memory freed.")
