"""Assembler step for the California Housing Lightning workflow.

Packages the trained Lightning model (an intra-pipeline ``ModelVariable``)
into a registry-ready ``AssembledModel`` via ``torch_assembler``, replacing
``push_step``'s previous ad-hoc ``ModelArtifact`` wrapping now that an OSS
assembler task exists. When ``pipeline.py``'s ``native_transform`` stage
produced a fitted transform module, it is fused into the packaged artifact
too (``torch_assembler``'s ``native_transform_model`` parameter), so a real
serving deployment can run the packaged artifact directly on raw inputs
without re-applying the transform itself.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import fsspec
import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.schema.assembler import (
    TabularAssemblerConfig,
    TorchAssemblerConfig,
)
from michelangelo.workflow.tasks.tabular_assembler.torch.assembler import (
    torch_assembler,
)
from michelangelo.workflow.variables.types import AssembledModel, ModelArtifact

from michelangelo_examples.california_housing.pipelines.pytorch_train._backend import (
    resolve_storage_backend,
)

if TYPE_CHECKING:
    from michelangelo.lib.artifact_manager.storage_backend import StorageBackend
    from michelangelo.workflow.variables import ModelVariable

log = logging.getLogger(__name__)

__all__ = ["assembler"]


def _to_model_artifact(
    model_variable: ModelVariable,
    storage_backend: StorageBackend,
    upload_prefix: str,
) -> ModelArtifact:
    """Re-upload a ``ModelVariable``'s state-dict through ``storage_backend``.

    ``ModelVariable``s produced by intra-pipeline tasks are persisted under
    ``UF_STORAGE_URL`` -- not yet uploaded through a ``StorageBackend`` URI
    that ``torch_assembler`` can pull from directly. Downloads the state-dict
    file via the same fsspec mechanism ``ModelVariable`` itself uses, then
    re-uploads it through the given storage backend so ``torch_assembler``'s
    ``storage_backend.download()`` call resolves a URI that backend instance
    actually wrote (a ``LocalStorageBackend`` only accepts URIs it produced
    itself).
    """
    local_dir = tempfile.mkdtemp(prefix=f"{upload_prefix}_")
    local_model_path = os.path.join(local_dir, "model.pt")
    fs, remote_model_path = fsspec.core.url_to_fs(model_variable.path)
    fs.get(remote_model_path, local_model_path)

    model_uri = storage_backend.upload(local_model_path, f"{upload_prefix}/model.pt")
    return ModelArtifact(path=model_uri, metadata=model_variable.metadata)


@uniflow.task(
    config=RayTask(head_cpu=1, head_memory="2Gi", worker_instances=0),
)
def assembler(
    model_variable: ModelVariable,
    native_transform_model: ModelVariable | None = None,
) -> AssembledModel:
    """Package the trained Lightning model into deployable and raw Triton packages.

    Args:
        model_variable: Result of the ``train`` task -- a ``ModelVariable``
            wrapping the trained Lightning model, with ``schema``/
            ``sample_data``/``hyperparameters`` set on its metadata.
        native_transform_model: ``native_transform``'s fitted transform
            module (``NativeTransformResult.model``), or ``None`` when the
            transform spec produced no layers. When set, ``torch_assembler``
            fuses it with ``model_variable`` into a single servable artifact
            with a combined schema, so the packaged artifact can run
            directly on raw (pre-transform) inputs.

    Returns:
        ``AssembledModel`` with the deployable and raw packaged artifacts.
    """
    storage_backend, _ = resolve_storage_backend("california_lightning_assemble_")

    raw_model = _to_model_artifact(model_variable, storage_backend, "raw_model")
    native_tx_artifact = (
        _to_model_artifact(
            native_transform_model, storage_backend, "native_transform_model"
        )
        if native_transform_model is not None
        else None
    )

    return torch_assembler(
        # include_import_prefixes scopes the packager's static import walk to
        # this project's own package -- matching internal Michelangelo's
        # convention of always scoping to ["uber"]. Leaving this unset lets
        # the walk wander into every reachable third-party/stdlib module
        # (torch, numpy.f2py, CPython's own test package, ...), several of
        # which have CLI-style import-time side effects (printing help text,
        # calling sys.exit(), raising from torch's ConfigModule) that crash
        # the walk.
        TabularAssemblerConfig(
            torch=TorchAssemblerConfig(
                include_import_prefixes=["michelangelo_examples"]
            )
        ),
        raw_model,
        native_transform_model=native_tx_artifact,
        storage_backend=storage_backend,
    )
