"""Pusher step for the BERT CoLA fine-tuning workflow.

Registers the assembled model (produced by ``assembler``) in a model
registry via ``ModelPusherPlugin``. Constructs its own storage backend and
registry client, independently of ``assembler.py`` -- uniflow tasks can't
share live objects across the workflow boundary.
"""

from __future__ import annotations

import logging
import os
import time
from urllib.parse import urlparse

import grpc
import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask
from michelangelo.workflow.schema.pusher import (
    ModelPluginConfig,
    PusherConfig,
    PusherPluginConfig,
)
from michelangelo.workflow.tasks.pusher import push
from michelangelo.workflow.tasks.pusher.exceptions import PusherPluginError

# Kept top-level (not TYPE_CHECKING) despite only being used in annotations --
# uniflow's @uniflow.task needs these resolvable as real objects at the
# workflow boundary.
from michelangelo.workflow.variables.types import (
    AssembledModel,
    PusherResult,
)

log = logging.getLogger(__name__)

__all__ = ["push_step"]


@uniflow.task(
    config=RayTask(head_cpu=1, head_memory="1Gi", worker_instances=0),
)
def push_step(assembled: AssembledModel) -> list[PusherResult]:
    """Push the assembled BERT CoLA model to storage and the model registry.

    Args:
        assembled: Result of ``assembler`` -- deployable and raw Triton
            packages for the fine-tuned classifier.

    Returns:
        List of ``PusherResult``, one per artifact pushed (just ``model`` here).
    """
    s3_endpoint = os.environ.get("AWS_ENDPOINT_URL", "")
    if s3_endpoint:
        parsed = urlparse(s3_endpoint)
        endpoint = parsed.netloc
        if not endpoint:
            raise ValueError(
                f"AWS_ENDPOINT_URL={s3_endpoint!r} is missing a scheme. "
                "Use a full URL like http://minio:9091"
            )
        bucket = (
            os.environ.get("AWS_S3_BUCKET")
            or os.environ.get("MA_FILE_SYSTEM", "s3://default")
            .removeprefix("s3://")
            .split("/")[0]
        )
        from michelangelo.lib.artifact_manager.minio_backend import MinioStorageBackend

        storage_backend = MinioStorageBackend(
            endpoint=endpoint,
            bucket=bucket,
            access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            secure=parsed.scheme == "https",
            create_bucket_if_missing=True,
        )
        log.info(
            "push_step: using MinioStorageBackend (remote) -> %s",
            storage_backend.get_storage_location(),
        )
    else:
        import tempfile

        from michelangelo.lib.artifact_manager.storage_backend import (
            LocalStorageBackend,
        )

        local_dir = tempfile.mkdtemp(prefix="bert_cola_push_")
        storage_backend = LocalStorageBackend(local_dir)
        log.info("push_step: using LocalStorageBackend (local/CI) -> %s", local_dir)

    registry_endpoint = os.environ.get("REGISTRY_ENDPOINT")
    if registry_endpoint:
        from michelangelo.api.v2 import APIClient
        from michelangelo.lib.model_manager.registry.api_client import (
            APIRegistryClient,
        )

        _insecure = os.environ.get("REGISTRY_INSECURE", "true").lower() != "false"
        _credentials = None if _insecure else grpc.ssl_channel_credentials()
        # Keepalive pings guard against the k8s overlay network silently
        # dropping an idle connection between channel creation and the first
        # RPC (observed as UNAVAILABLE/"tcp handshaker shutdown" on the k3d
        # sandbox) -- without them the channel's initial TCP connection can
        # go stale during the raw/deployable artifact upload that precedes
        # registration.
        _channel_options = [
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]
        _channel = (
            grpc.insecure_channel(registry_endpoint, options=_channel_options)
            if _insecure
            else grpc.secure_channel(
                registry_endpoint, _credentials, options=_channel_options
            )
        )
        _api_client = APIClient(caller="bert-cola-push-step", channel=_channel)
        registry_client = APIRegistryClient(
            svc=_api_client.ModelService,
            namespace=os.environ.get(
                "REGISTRY_NAMESPACE", os.environ.get("MA_NAMESPACE", "default")
            ),
        )
        log.info("push_step: using APIRegistryClient at %s", registry_endpoint)
    else:
        from michelangelo.lib.model_manager.registry.client import (
            InMemoryRegistryClient,
        )

        registry_client = InMemoryRegistryClient()
        log.warning(
            "REGISTRY_ENDPOINT not set -- using InMemoryRegistryClient. "
            "Model registration will not be persisted."
        )

    config = PusherConfig(
        items=[
            PusherPluginConfig(
                name="model",
                model_plugin=ModelPluginConfig(
                    description="BERT fine-tuned for CoLA linguistic acceptability",
                    labels={"framework": "transformers"},
                    tar_deployable_package=True,
                ),
            ),
        ]
    )

    # Retry on transient gRPC UNAVAILABLE errors talking to the registry --
    # observed intermittently on the k3d sandbox even with channel
    # keepalive enabled (see comment above).
    _max_attempts = 3
    for attempt in range(1, _max_attempts + 1):
        try:
            results = push(
                config=config,
                artifacts={"model": assembled},
                storage_backend=storage_backend,
                registry_client=registry_client,
            )
            break
        except PusherPluginError as exc:
            cause = exc.__cause__
            is_transient = (
                isinstance(cause, grpc.RpcError)
                and cause.code() == grpc.StatusCode.UNAVAILABLE
            )
            if not is_transient or attempt == _max_attempts:
                raise
            log.warning(
                "push_step: transient registry UNAVAILABLE on attempt %d/%d, retrying: %s",
                attempt,
                _max_attempts,
                cause,
            )
            time.sleep(2**attempt)

    for r in results:
        log.info(
            "push %s (%s): success=%s value=%s error=%s",
            r.name,
            r.plugin,
            r.success,
            r.value,
            r.error,
        )

    return results
