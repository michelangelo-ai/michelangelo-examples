"""Unit tests for california_housing's ``resolve_storage_backend()`` helpers.

``pytorch_train/_backend.py`` and ``xgb_train/_backend.py`` share the same
logic, so every test runs against both. ``FsspecStorageBackend`` is replaced
with a recorder, so no object store or network is needed.
"""

from __future__ import annotations

import pytest
from michelangelo.lib import artifact_manager
from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend

from michelangelo_examples.california_housing.pipelines.pytorch_train import (
    _backend as pytorch_backend,
)
from michelangelo_examples.california_housing.pipelines.xgb_train import (
    _backend as xgb_backend,
)

_ENV_VARS = (
    "AWS_ENDPOINT_URL",
    "AWS_S3_BUCKET",
    "MA_FILE_SYSTEM",
    "UF_STORAGE_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


@pytest.fixture(params=[pytorch_backend, xgb_backend], ids=["pytorch", "xgb"])
def backend_module(request, monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return request.param


@pytest.fixture
def recorded(monkeypatch):
    """Replace ``FsspecStorageBackend`` and record its constructor arguments."""
    calls: list[tuple[tuple, dict]] = []

    class _FakeFsspecStorageBackend:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self._base_uri = args[0]

        def get_storage_location(self):
            return self._base_uri

    monkeypatch.setattr(
        artifact_manager, "FsspecStorageBackend", _FakeFsspecStorageBackend
    )
    return calls


def test_local_when_endpoint_unset(backend_module, recorded):
    backend, is_remote = backend_module.resolve_storage_backend("test_backend_")

    assert isinstance(backend, LocalStorageBackend)
    assert is_remote is False
    assert recorded == []


def test_remote_builds_fsspec_backend(backend_module, recorded, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://objectstore:9091")
    monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    backend, is_remote = backend_module.resolve_storage_backend("test_backend_")

    assert is_remote is True
    assert backend.get_storage_location() == "s3://my-bucket"
    [(args, kwargs)] = recorded
    base_uri, storage_options = args
    assert base_uri == "s3://my-bucket"
    assert kwargs == {"create_bucket_if_missing": True}
    assert storage_options["client_kwargs"]["endpoint_url"] == (
        "http://objectstore:9091"
    )
    assert storage_options["key"] == "key"
    assert storage_options["secret"] == "secret"


def test_https_endpoint_and_bucket_from_ma_file_system(
    backend_module, recorded, monkeypatch
):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("MA_FILE_SYSTEM", "s3://from-ma-fs/some/prefix")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    backend_module.resolve_storage_backend("test_backend_")

    [(args, _)] = recorded
    base_uri, storage_options = args
    assert base_uri == "s3://from-ma-fs"
    assert storage_options["client_kwargs"]["endpoint_url"] == (
        "https://s3.example.com"
    )


def test_endpoint_without_scheme_is_rejected(backend_module, recorded, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "objectstore:9091")

    with pytest.raises(ValueError, match="missing a scheme"):
        backend_module.resolve_storage_backend("test_backend_")
    assert recorded == []
