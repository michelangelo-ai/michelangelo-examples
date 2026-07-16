"""Assembler step for the California Housing Lightning workflow.

Packages the trained model produced by ``train`` into a registry-ready
``AssembledModel`` via michelangelo's ``tabular_assembler`` task. This is a
thin adapter: ``train_tabular()`` already populates ``ModelVariable.metadata``
with everything ``tabular_assembler`` needs (training framework, model class,
hyperparameters, schema, sample data), so this step only has to resolve a
storage backend, unpickle the trainer's serialised schema/sample-data
payloads into the live fields ``tabular_assembler`` actually reads, and
convert the intra-pipeline ``ModelVariable`` into the ``ModelArtifact`` shape
``tabular_assembler`` expects.

NOT MERGEABLE YET: ``tabular_assembler`` doesn't exist in any released
michelangelo version -- it's from michelangelo-ai/michelangelo#1430, still
open/unmerged. This module only imports cleanly against that PR's branch
(see ma-oss-api-features spec 011 for how it was validated). Blocked on
#1430 merging and shipping in a real release before this can land on main.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import michelangelo.uniflow.core as uniflow
from michelangelo_examples.california_housing.pipelines.pytorch_train._backend import (
    resolve_storage_backend,
)
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.schema.assembler import TabularAssemblerConfig
from michelangelo.workflow.tasks.tabular_assembler.task import tabular_assembler
from michelangelo.workflow.variables.types import AssembledModel, ModelArtifact

if TYPE_CHECKING:
    from michelangelo.workflow.variables import ModelVariable

__all__ = ["assembler"]


@uniflow.task(
    config=RayTask(head_cpu=1, head_memory="2Gi", worker_instances=0),
)
def assembler(model_variable: ModelVariable) -> AssembledModel:
    """Package the trained model into a registry-ready ``AssembledModel``.

    Single-process download/package/upload step — no distributed workers
    needed, unlike ``train``'s multi-worker Ray Train job. ``worker_instances``
    must be set explicitly: leaving it unset (``None``) does not mean
    "head-only" -- the underlying Ray task template falls back to
    ``RAY_DEFAULT_WORKER_INSTANCES`` (1) whenever it isn't set, which leaves
    this step stuck waiting on a worker pod it never needs.

    Args:
        model_variable: Result of the ``train`` task -- a ``ModelVariable``
            wrapping the trained Lightning model, persisted under
            ``UF_STORAGE_URL`` with metadata already populated by
            ``train_tabular()`` (training framework, model class,
            hyperparameters, and the serialised ``_schema``/``_sample_data``
            payloads).

    Returns:
        An ``AssembledModel`` with the deployable and raw Triton packages,
        ready to hand to ``push``.
    """
    storage_backend, _ = resolve_storage_backend("california_lightning_assemble_")

    metadata = model_variable.metadata
    # train_tabular() only populates the serialised _schema/_sample_data
    # payloads (the registry-bound form); tabular_assembler reads the live
    # schema/sample_data fields instead (see ModelMetadata's docstring for
    # why these are kept distinct). Unpickle here so the assembler sees them.
    if metadata.schema is None and metadata._schema is not None:
        metadata.schema = pickle.loads(metadata._schema.getvalue())
    if metadata.sample_data is None and metadata._sample_data is not None:
        metadata.sample_data = pickle.loads(metadata._sample_data.getvalue())

    raw_model = ModelArtifact(path=model_variable.path, metadata=metadata)

    return tabular_assembler(
        TabularAssemblerConfig(),
        raw_model,
        storage_backend=storage_backend,
    )

