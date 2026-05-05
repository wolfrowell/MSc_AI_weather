"""Entry point: fine-tunes GraphCast with a pluggable strategy.

Swap the strategy object to change the fine-tuning method — everything else stays the same.

Current strategy: FrozenModuleStrategy(preset="encoder")
  Trainable : grid2mesh_gnn  (encoder)
  Frozen    : mesh_gnn, mesh2grid_gnn

Memory note: training the encoder requires storing processor + decoder activations
during backprop. Reduce TRAIN_STEPS if OOM.
"""

import os
import warnings

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
warnings.filterwarnings("ignore")

from src.patch import GraphCastPatcher
from src.data_loader import GCSDataLoader
from src.finetuning import FrozenModuleStrategy
from src.model import GraphCastModel
from src.trainer import Trainer, TrainingConfig


PARAMS_FILE = (
    "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 "
    "- pressure levels 13 - mesh 2to6 - precipitation output only.npz"
)
TRAIN_STEPS = 1
EVAL_STEPS = None


def main():
    # 1. Patch graphcast.py
    GraphCastPatcher().apply()

    # 2. Load checkpoint
    loader = GCSDataLoader()
    ckpt = loader.load_checkpoint(PARAMS_FILE)
    params, state = ckpt.params, {}
    model_config, task_config = ckpt.model_config, ckpt.task_config

    # 3. Load dataset
    datasets = loader.list_datasets(model_config, task_config)
    dataset_file = datasets[0]
    print(f"Using dataset: {dataset_file}")
    example_batch = loader.load_dataset(dataset_file)

    # 4. Split train / eval
    (
        train_inputs, train_targets, train_forcings,
        eval_inputs, eval_targets, eval_forcings,
    ) = GCSDataLoader.split_inputs_targets(
        example_batch, task_config,
        train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS,
    )

    # 5. Normalization stats
    diffs_stddev, mean_by_level, stddev_by_level = loader.load_normalization_stats()

    # ── Strategy selection ────────────────────────────────────────────────────
    # Swap this line to change the fine-tuning method.
    # Examples:
    #   FrozenModuleStrategy(preset="decoder")   → train only decoder (low memory)
    #   FrozenModuleStrategy(preset="encoder")   → train only encoder
    #   FrozenModuleStrategy(preset="processor") → train only processor
    #   FrozenModuleStrategy(trainable_modules=["grid2mesh_gnn", "mesh2grid_gnn"])
    strategy = FrozenModuleStrategy(preset="encoder")

    # 6. Build model
    model = GraphCastModel(
        model_config=model_config,
        task_config=task_config,
        diffs_stddev_by_level=diffs_stddev,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
        strategy=strategy,
    )

    # 7. Prepare fine-tuning state (partition params)
    ft_state = model.prepare(params, state)
    strategy.print_summary(ft_state)

    # 8. Eval before training
    print("\nRunning eval rollout before training...")
    predictions = model.predict(ft_state, eval_inputs, eval_targets, eval_forcings)
    print("Predictions shape:", dict(predictions.dims))

    # 9. Free GPU memory before backprop
    del predictions
    Trainer.free_memory(globs=globals())

    # 10. Train
    config = TrainingConfig(num_epochs=100, learning_rate=1e-3, log_every=10)
    trainer = Trainer(model, config)

    trainer.print_gradient_stats(ft_state, train_inputs, train_targets, train_forcings)

    ft_state, losses = trainer.fit(
        ft_state, train_inputs, train_targets, train_forcings
    )

    print(f"\nFinal loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
