"""Baseline predictors for comparison against fine-tuned GraphCast.

Baselines
---------
PersistenceBaseline : x̂(t + τ) = x(t)  — last available input, for all lead times.
PretrainedBaseline  : pre-trained GraphCast with no fine-tuning.
"""

import numpy as np
import xarray

from .model import GraphCastModel
from .finetuning.base import FineTuningState


class PersistenceBaseline:
    """Predicts the last input state for every lead time.

    This is the simplest possible baseline: tomorrow = today.
    Competitive at short lead times, degrades quickly beyond ~24h.
    """

    def predict(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
    ) -> xarray.Dataset:
        """Tile the last input time step to match every target lead time."""
        last_input = inputs.isel(time=-1)
        n_steps = targets_template.sizes["time"]

        slices = []
        for t in range(n_steps):
            step = last_input.expand_dims(time=1)
            step = step.assign_coords(
                time=[targets_template.coords["time"].values[t]]
            )
            slices.append(step)

        return xarray.concat(slices, dim="time")


class PretrainedBaseline:
    """Runs the pre-trained GraphCast model with no fine-tuning.

    This is the most important baseline: if fine-tuning does not beat this,
    the method is not working.
    """

    def __init__(self, model: GraphCastModel, params: dict, state: dict):
        self.model = model
        self._ft_state = model.prepare(params, state)

    def predict(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: xarray.Dataset,
    ) -> xarray.Dataset:
        return self.model.predict(self._ft_state, inputs, targets, forcings)
