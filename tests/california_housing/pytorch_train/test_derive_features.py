"""Unit tests for ``derive_features``'s derived-column logic.

Exercises ``_add_derived_columns`` directly against a plain pandas
``DataFrame`` -- no Ray/uniflow task machinery involved, since the derivation
itself is pure pandas arithmetic.
"""

from __future__ import annotations

import pandas as pd

from michelangelo_examples.california_housing.pipelines.pytorch_train.derive_features import (
    _add_derived_columns,
)


def test_add_derived_columns_basic_values():
    batch = pd.DataFrame({"HouseAge": [10.0, 25.0], "Population": [500.0, 1200.0]})

    out = _add_derived_columns(batch)

    assert list(out["derived_house_age"]) == [11.0, 26.0]
    assert list(out["derived_population"]) == [500.0, 1200.0]


def test_add_derived_columns_guards_against_nulls():
    batch = pd.DataFrame({"HouseAge": [10.0, None], "Population": [500.0, None]})

    out = _add_derived_columns(batch)

    assert list(out["derived_house_age"]) == [11.0, 1.0]
    assert list(out["derived_population"]) == [500.0, 0.0]


def test_add_derived_house_age_is_never_zero_or_negative():
    # LogTransform (native_transform.py) requires derived_house_age > 0;
    # the +1 offset must hold even for a HouseAge of 0.
    batch = pd.DataFrame({"HouseAge": [0.0], "Population": [0.0]})

    out = _add_derived_columns(batch)

    assert (out["derived_house_age"] > 0).all()
