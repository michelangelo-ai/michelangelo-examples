"""Unit tests for the Qwen dual-encoder training helpers.

No Ray cluster or pretrained weights are needed: the encoder is replaced by a
tiny stub, and the checkpoint round-trip uses Ray's public ``Checkpoint`` API.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from ray.train import Checkpoint
from torch import nn

from michelangelo_examples.amazon_books_qwen.pipelines.train.train import (
    CHECKPOINT_FILENAME,
    QwenDualEncoder,
    load_checkpoint_state,
)


class _StubTokenizer:
    def __call__(self, texts, **_kwargs):
        n = len(texts)
        return {
            "input_ids": torch.ones(n, 3, dtype=torch.long),
            "attention_mask": torch.ones(n, 3, dtype=torch.long),
        }


class _StubEncoder(nn.Module):
    """Fails like a CUDA model fed CPU tensors if inputs are on another device."""

    def __init__(self, device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, device=device))
        self.expected = torch.device(device)

    def forward(self, input_ids, attention_mask):
        for tensor in (input_ids, attention_mask):
            if tensor.device != self.expected:
                raise RuntimeError(
                    f"Expected all tensors on {self.expected}, got {tensor.device}"
                )
        return SimpleNamespace(last_hidden_state=torch.zeros(len(input_ids), 3, 4))


def _make_encoder(device: str) -> QwenDualEncoder:
    model = QwenDualEncoder.__new__(QwenDualEncoder)
    nn.Module.__init__(model)
    model.tokenizer = _StubTokenizer()
    model.max_query_length = model.max_doc_length = 8
    model.query_encoder = _StubEncoder(device)
    model.doc_encoder = _StubEncoder(device)
    model.query_projection = nn.Linear(4, 2)
    model.doc_projection = nn.Linear(4, 2)
    return model


def test_encode_moves_tokenizer_inputs_to_encoder_device():
    # The meta device stands in for CUDA: a CPU tensor reaching the encoder
    # would trip the stub's device check, just as it does on a GPU.
    model = _make_encoder("meta")

    query_embeddings = model.encode_queries(["q1", "q2"])
    doc_embeddings = model.encode_documents(["d1", "d2"])

    assert query_embeddings.shape == (2, 2)
    assert doc_embeddings.shape == (2, 2)


def test_checkpoint_directory_round_trip(tmp_path):
    model = nn.Linear(4, 2)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "train_loss": 0.5,
        "val_loss": 0.25,
    }
    torch.save(state, tmp_path / CHECKPOINT_FILENAME)

    checkpoint = Checkpoint.from_directory(str(tmp_path))
    with checkpoint.as_directory() as checkpoint_dir:
        loaded = load_checkpoint_state(checkpoint_dir)

    assert loaded["epoch"] == 1
    assert loaded["val_loss"] == 0.25
    for key, value in model.state_dict().items():
        assert torch.equal(loaded["model_state_dict"][key], value)
