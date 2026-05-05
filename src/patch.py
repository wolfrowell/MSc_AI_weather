"""Patches the pip-installed graphcast.py to support freeze_encoder / freeze_processor."""

import importlib
import importlib.util
import inspect
from pathlib import Path

_GRAPHCAST_PATH = Path(importlib.util.find_spec("graphcast.graphcast").origin)

_PATCHES = [
    {
        "name": "Patch 0: add 'import jax'",
        "check_absent": "import jax\n",
        "anchor": "import jax.numpy as jnp",
        "old": "import jax.numpy as jnp",
        "new": "import jax\nimport jax.numpy as jnp",
    },
    {
        "name": "Patch 1: freeze flags in __init__ signature",
        "old": "  def __init__(self, model_config: ModelConfig, task_config: TaskConfig):",
        "new": (
            "  def __init__(self, model_config: ModelConfig, task_config: TaskConfig,\n"
            "               freeze_encoder: bool = False, freeze_processor: bool = False):"
        ),
    },
    {
        "name": "Patch 2: store freeze flags in __init__ body",
        "old": (
            '    """Initializes the predictor."""\n'
            "    self._spatial_features_kwargs"
        ),
        "new": (
            '    """Initializes the predictor."""\n'
            "    self._freeze_encoder = freeze_encoder\n"
            "    self._freeze_processor = freeze_processor\n"
            "    self._spatial_features_kwargs"
        ),
    },
    {
        "name": "Patch 3: stop_gradient after encoder output",
        "old": (
            "     ) = self._run_grid2mesh_gnn(grid_node_features)\n\n"
            "    # Run message passing in the multimesh."
        ),
        "new": (
            "     ) = self._run_grid2mesh_gnn(grid_node_features)\n\n"
            "    # Stop gradient through encoder outputs to save HBM memory during backprop.\n"
            "    # This prevents JAX from storing encoder activations for gradient computation.\n"
            "    if self._freeze_encoder:\n"
            "      latent_mesh_nodes = jax.lax.stop_gradient(latent_mesh_nodes)\n"
            "      latent_grid_nodes = jax.lax.stop_gradient(latent_grid_nodes)\n\n"
            "    # Run message passing in the multimesh."
        ),
        # Upgrade path for a prior run that applied the block without comments.
        "old_no_comment": (
            "     ) = self._run_grid2mesh_gnn(grid_node_features)\n\n"
            "    if self._freeze_encoder:\n"
            "      latent_mesh_nodes = jax.lax.stop_gradient(latent_mesh_nodes)\n"
            "      latent_grid_nodes = jax.lax.stop_gradient(latent_grid_nodes)\n\n"
            "    # Run message passing in the multimesh."
        ),
        "skip_marker": "# Stop gradient through encoder outputs",
    },
    {
        "name": "Patch 4: stop_gradient after processor output",
        "old": (
            "    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)\n\n"
            "    # Transfer data frome the mesh to the grid."
        ),
        "new": (
            "    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)\n\n"
            "    # Stop gradient through processor outputs to save HBM memory during backprop.\n"
            "    # This prevents JAX from storing processor activations (16 steps) for gradients.\n"
            "    if self._freeze_processor:\n"
            "      updated_latent_mesh_nodes = jax.lax.stop_gradient(updated_latent_mesh_nodes)\n\n"
            "    # Transfer data frome the mesh to the grid."
        ),
        "old_no_comment": (
            "    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)\n\n"
            "    if self._freeze_processor:\n"
            "      updated_latent_mesh_nodes = jax.lax.stop_gradient(updated_latent_mesh_nodes)\n\n"
            "    # Transfer data frome the mesh to the grid."
        ),
        "skip_marker": "# Stop gradient through processor outputs",
    },
]


class GraphCastPatcher:
    """Applies in-place string patches to the pip-installed graphcast.py."""

    def __init__(self, path: Path = _GRAPHCAST_PATH):
        self.path = path

    def apply(self) -> None:
        src = self.path.read_text(encoding="utf-8")
        changed = False

        for patch in _PATCHES:
            src, patched = self._apply_one(src, patch)
            if patched:
                changed = True

        if changed:
            self.path.write_text(src, encoding="utf-8")
            print(f"Written: {self.path}  ({self.path.stat().st_size:,} bytes)")
        else:
            print("All patches already present — file unchanged.")

        self._verify()

    def _apply_one(self, src: str, patch: dict) -> tuple[str, bool]:
        name = patch["name"]
        skip_marker = patch.get("skip_marker")
        old = patch["old"]
        new = patch["new"]
        old_no_comment = patch.get("old_no_comment")

        if skip_marker and skip_marker in src:
            print(f"{name}: skipped (already present)")
            return src, False

        if new in src:
            print(f"{name}: skipped (already present)")
            return src, False

        if old in src:
            src = src.replace(old, new, 1)
            print(f"{name}: applied")
            return src, True

        if old_no_comment and old_no_comment in src:
            src = src.replace(old_no_comment, new, 1)
            print(f"{name}: upgraded (added comments)")
            return src, True

        raise AssertionError(
            f"{name} FAILED — anchor not found.\n"
            f"The pip-installed graphcast version may have changed. Check:\n  {self.path}"
        )

    def _verify(self) -> None:
        from graphcast import graphcast as _gc_module
        importlib.reload(_gc_module)
        sig = inspect.signature(_gc_module.GraphCast.__init__)
        assert "freeze_encoder" in sig.parameters, "freeze_encoder missing after patch"
        assert "freeze_processor" in sig.parameters, "freeze_processor missing after patch"
        print(f"\ngraphcast.py active. GraphCast.__init__{sig}")
