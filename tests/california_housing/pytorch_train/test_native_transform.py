"""Unit tests for ``native_transform``'s transform-spec DAG.

Builds and executes ``_TRANSFORM_SPEC`` through
``michelangelo.lib.native_transform.torch.transform_spec.TransformSpec``
directly -- the same DAG engine ``tabular_native_transform()`` uses
internally -- without going through Ray. This exercises the real
topological-sort/materialize/execute path that determines whether the
transform chain is wired correctly, independent of the Ray dataset
plumbing that ``tabular_native_transform()`` also handles.
"""

from __future__ import annotations

import torch
from michelangelo.lib.native_transform.torch.transform_spec import TransformSpec

from michelangelo_examples.california_housing.pipelines.pytorch_train.native_transform import (
    _TRANSFORM_SPEC,
    FEATURE_COLUMNS,
)


def _run_transform_spec(spec: TransformSpec, data: dict[str, torch.Tensor]) -> dict:
    data = dict(data)
    for level in range(spec.get_max_transform_level() + 1):
        for layer in spec.to_transform_layers(level):
            out = layer(data)
            data.update(out)
    return data


def test_transform_spec_has_two_levels():
    spec = TransformSpec(raw_transform_specs=_TRANSFORM_SPEC)

    assert spec.get_max_transform_level() == 1


def test_transform_spec_level_1_depends_on_level_0_output():
    spec = TransformSpec(raw_transform_specs=_TRANSFORM_SPEC)

    level_0_outputs = set(spec.get_transform_output_cols(0))
    level_1_inputs = set(spec.get_transform_input_cols(1))

    assert level_1_inputs <= level_0_outputs


def test_transform_spec_produces_all_feature_columns():
    spec = TransformSpec(raw_transform_specs=_TRANSFORM_SPEC)
    data = {
        "AveRooms": torch.tensor([5.0, 6.0, 7.0, 4.0]),
        "derived_house_age": torch.tensor([11.0, 6.0, 26.0, 16.0]),
        "derived_population": torch.tensor([500.0, 300.0, 1200.0, 800.0]),
    }

    result = _run_transform_spec(spec, data)

    for column in FEATURE_COLUMNS:
        assert column in result


def test_scale_layer_applies_factor():
    spec = TransformSpec(raw_transform_specs=_TRANSFORM_SPEC)
    data = {
        "AveRooms": torch.tensor([5.0, 10.0]),
        "derived_house_age": torch.tensor([1.0, 1.0]),
        "derived_population": torch.tensor([0.0, 0.0]),
    }

    result = _run_transform_spec(spec, data)

    torch.testing.assert_close(result["AveRooms_scaled"], torch.tensor([0.5, 1.0]))


def test_clip_layer_bounds_values():
    spec = TransformSpec(raw_transform_specs=_TRANSFORM_SPEC)
    data = {
        "AveRooms": torch.tensor([1.0, 1.0]),
        "derived_house_age": torch.tensor([1.0, 1.0]),
        "derived_population": torch.tensor([-100.0, 10_000.0]),
    }

    result = _run_transform_spec(spec, data)

    torch.testing.assert_close(
        result["derived_population_clipped"], torch.tensor([0.0, 5000.0])
    )
