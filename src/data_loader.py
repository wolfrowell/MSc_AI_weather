"""Loads GraphCast model checkpoints, weather datasets, and normalization stats from GCS."""

import dataclasses
from typing import Optional

import xarray
from google.cloud import storage
from graphcast import checkpoint, data_utils, graphcast


_GCS_BUCKET = "dm_graphcast"
_DIR_PREFIX = "graphcast/"


def _parse_file_parts(file_name: str) -> dict:
    return dict(part.split("-", 1) for part in file_name.split("_"))


class GCSDataLoader:
    """Downloads GraphCast assets from the public GCS bucket."""

    def __init__(self, bucket_name: str = _GCS_BUCKET, dir_prefix: str = _DIR_PREFIX):
        self._client = storage.Client.create_anonymous_client()
        self._bucket = self._client.get_bucket(bucket_name)
        self._prefix = dir_prefix

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def load_checkpoint(self, params_file: str) -> graphcast.CheckPoint:
        blob_path = f"{self._prefix}params/{params_file}"
        with self._bucket.blob(blob_path).open("rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        print("Model description:\n", ckpt.description)
        print("Model license:\n", ckpt.license)
        return ckpt

    # ── Dataset ───────────────────────────────────────────────────────────────

    def list_datasets(
        self,
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig,
    ) -> list[str]:
        all_blobs = self._bucket.list_blobs(prefix=self._prefix + "dataset/")
        names = [
            blob.name.removeprefix(self._prefix + "dataset/")
            for blob in all_blobs
        ]
        return [n for n in names if n and self._data_valid(n, model_config, task_config)]

    def load_dataset(self, file_name: str) -> xarray.Dataset:
        blob_path = f"{self._prefix}dataset/{file_name}"
        with self._bucket.blob(blob_path).open("rb") as f:
            ds = xarray.load_dataset(f).compute()
        assert ds.dims["time"] >= 3, "Dataset needs at least 3 time steps (2 input + 1 target)."
        parts = _parse_file_parts(file_name.removesuffix(".nc"))
        print(", ".join(f"{k}: {v}" for k, v in parts.items()))
        return ds

    # ── Normalization stats ───────────────────────────────────────────────────

    def load_normalization_stats(self) -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
        """Returns (diffs_stddev_by_level, mean_by_level, stddev_by_level)."""
        files = {
            "diffs_stddev_by_level": "stats/diffs_stddev_by_level.nc",
            "mean_by_level": "stats/mean_by_level.nc",
            "stddev_by_level": "stats/stddev_by_level.nc",
        }
        loaded = {}
        for key, path in files.items():
            with self._bucket.blob(self._prefix + path).open("rb") as f:
                loaded[key] = xarray.load_dataset(f).compute()
        return loaded["diffs_stddev_by_level"], loaded["mean_by_level"], loaded["stddev_by_level"]

    # ── Data splitting ────────────────────────────────────────────────────────

    @staticmethod
    def split_inputs_targets(
        example_batch: xarray.Dataset,
        task_config: graphcast.TaskConfig,
        train_steps: int,
        eval_steps: Optional[int] = None,
    ) -> tuple:
        """Extract (train_inputs, train_targets, train_forcings, eval_inputs, eval_targets, eval_forcings)."""
        if eval_steps is None:
            eval_steps = example_batch.sizes["time"] - 2

        train = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{train_steps * 6}h"),
            **dataclasses.asdict(task_config),
        )
        eval_ = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{eval_steps * 6}h"),
            **dataclasses.asdict(task_config),
        )
        return (*train, *eval_)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _data_valid(
        file_name: str,
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig,
    ) -> bool:
        parts = _parse_file_parts(file_name.removesuffix(".nc"))
        has_precip = "total_precipitation_6hr" in task_config.input_variables
        source_ok = (
            parts["source"] in ("era5", "fake") if has_precip
            else parts["source"] in ("hres", "fake")
        )
        return (
            model_config.resolution in (0, float(parts["res"]))
            and len(task_config.pressure_levels) == int(parts["levels"])
            and source_ok
        )
