"""Unit tests for the pytorch_train ``assembler`` task's native-transform fallback.

``torch_assembler`` and the storage helpers are replaced with fakes, so no Ray,
object storage or real model is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from michelangelo_examples.california_housing.pipelines.pytorch_train import (
    assembler as assembler_module,
)

# The ``@uniflow.task`` decorator wraps the function; call the plain one.
_assemble = getattr(
    assembler_module.assembler, "__wrapped__", assembler_module.assembler
)


@pytest.fixture
def calls(monkeypatch):
    """Patch out storage and record every ``torch_assembler`` call."""
    recorded: list[object] = []
    monkeypatch.setattr(
        assembler_module,
        "resolve_storage_backend",
        lambda prefix: (SimpleNamespace(), None),
    )
    monkeypatch.setattr(
        assembler_module,
        "_to_model_artifact",
        lambda variable, backend, prefix: f"artifact:{prefix}",
    )
    return recorded


def _patch_torch_assembler(monkeypatch, recorded, fail_on_native_transform):
    def fake(config, raw_model, native_transform_model=None, storage_backend=None):
        recorded.append(native_transform_model)
        if native_transform_model is not None and fail_on_native_transform:
            raise NotImplementedError("native-transform package unavailable")
        return "assembled"

    monkeypatch.setattr(assembler_module, "torch_assembler", fake)


def test_fuses_native_transform_when_supported(monkeypatch, calls):
    _patch_torch_assembler(monkeypatch, calls, fail_on_native_transform=False)

    assert _assemble(object(), object()) == "assembled"
    assert calls == ["artifact:native_transform_model"]


def test_falls_back_without_native_transform_when_unsupported(monkeypatch, calls):
    _patch_torch_assembler(monkeypatch, calls, fail_on_native_transform=True)

    assert _assemble(object(), object()) == "assembled"
    assert calls == ["artifact:native_transform_model", None]


def test_does_not_mask_errors_when_no_native_transform(monkeypatch, calls):
    def always_fails(*args, **kwargs):
        raise NotImplementedError("unrelated")

    monkeypatch.setattr(assembler_module, "torch_assembler", always_fails)

    with pytest.raises(NotImplementedError):
        _assemble(object(), None)
