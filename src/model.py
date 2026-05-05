"""Builds jitted GraphCast functions: forward pass, loss, and gradient computation.

GraphCastModel is strategy-agnostic: it delegates all fine-tuning decisions
(which params are trainable, how the predictor is built) to a FineTuningStrategy.
"""

import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray

from graphcast import rollout, xarray_jax, xarray_tree

from .finetuning.base import FineTuningState, FineTuningStrategy


class GraphCastModel:
    """Wraps GraphCast with a pluggable fine-tuning strategy.

    The strategy controls:
      - Which parameters are trainable / frozen
      - Whether stop_gradient is inserted after encoder / processor
      - How extra parameters (LoRA, adapters, …) are initialized and merged
    """

    def __init__(
        self,
        model_config,
        task_config,
        diffs_stddev_by_level: xarray.Dataset,
        mean_by_level: xarray.Dataset,
        stddev_by_level: xarray.Dataset,
        strategy: FineTuningStrategy,
    ):
        self.model_config = model_config
        self.task_config = task_config
        self._norm_kwargs = dict(
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        self.strategy = strategy

        self._run_forward = self._build_run_forward()
        self._loss_fn = self._build_loss_fn()
        self._grads_fn = self._build_grads_fn()

        self._run_forward_jitted = jax.jit(self._with_configs(self._run_forward.init))
        self._grads_fn_jitted = jax.jit(self._with_configs(self._grads_fn))

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self, params: dict, state: dict) -> FineTuningState:
        """Delegates param partitioning (and any extra param init) to the strategy."""
        return self.strategy.prepare(params, state, self.model_config, self.task_config)

    def predict(
        self,
        ft_state: FineTuningState,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: xarray.Dataset,
    ) -> xarray.Dataset:
        """Autoregressive rollout using the merged (inference) params."""
        merged = self.strategy.merge_for_inference(ft_state)

        def _forward(rng, inputs, targets_template, forcings):
            preds, _ = self._run_forward.apply(
                merged, ft_state.model_state, rng,
                self.model_config, self.task_config,
                inputs, targets_template, forcings,
            )
            return preds

        return rollout.chunked_prediction(
            _forward,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets * np.nan,
            forcings=forcings,
        )

    def compute_grads(
        self,
        ft_state: FineTuningState,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: xarray.Dataset,
    ) -> tuple[float, dict, FineTuningState, dict]:
        """Returns (loss, diagnostics, updated_ft_state, grads).

        Gradients are computed only w.r.t. ft_state.trainable_params.
        """
        loss, diagnostics, next_state, grads = self._grads_fn_jitted(
            ft_state.trainable_params,
            ft_state.frozen_params,
            ft_state.model_state,
            inputs=inputs,
            targets=targets,
            forcings=forcings,
        )
        updated_ft_state = FineTuningState(
            trainable_params=ft_state.trainable_params,
            frozen_params=ft_state.frozen_params,
            model_state=next_state,
            extra_params=ft_state.extra_params,
        )
        return loss, diagnostics, updated_ft_state, grads

    # ── Internal builders ─────────────────────────────────────────────────────

    def _build_run_forward(self):
        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings):
            predictor = self.strategy.build_predictor(
                model_config, task_config, self._norm_kwargs, for_training=False
            )
            return predictor(inputs, targets_template=targets_template, forcings=forcings)
        return run_forward

    def _build_loss_fn(self):
        @hk.transform_with_state
        def loss_fn(model_config, task_config, inputs, targets, forcings):
            predictor = self.strategy.build_predictor(
                model_config, task_config, self._norm_kwargs, for_training=True
            )
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
            return xarray_tree.map_structure(
                lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
                (loss, diagnostics),
            )
        return loss_fn

    def _build_grads_fn(self):
        loss_fn = self._loss_fn
        strategy = self.strategy

        def grads_fn(trainable_params, frozen_params, model_state,
                     model_config, task_config, inputs, targets, forcings):
            def _aux(tp):
                merged = strategy.merge_for_inference(
                    FineTuningState(tp, frozen_params, model_state)
                )
                (loss, diagnostics), next_state = loss_fn.apply(
                    merged, model_state, jax.random.PRNGKey(0),
                    model_config, task_config,
                    inputs, targets, forcings,
                )
                return loss, (diagnostics, next_state)

            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
                _aux, has_aux=True
            )(trainable_params)
            return loss, diagnostics, next_state, grads

        return grads_fn

    def _with_configs(self, fn):
        return functools.partial(fn, model_config=self.model_config, task_config=self.task_config)
