"""Unit tests for the llm_prediction backends' predictors and entry-point guards.

No model weights, GPU, or Ray cluster are needed: the HF text-generation
pipeline and vLLM's ``LLM`` are replaced by stubs. The vLLM tests install a
fake ``vllm`` module, so they also run where the ``llm-prediction-vllm``
extra can't be installed (anything other than Linux AMD64).
"""

from __future__ import annotations

import importlib
import runpy
import sys
from collections import namedtuple
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

_VersionInfo = namedtuple(
    "_VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"]
)

_PIPELINES = "michelangelo_examples.llm_prediction.pipelines"


def test_hf_predictor_returns_prompt_and_generated_text(monkeypatch):
    pytest.importorskip("transformers")
    predict_mod = importlib.import_module(f"{_PIPELINES}.hf_predict.predict")
    calls = []

    def fake_pipeline(task, model, device_map):
        assert (task, model, device_map) == ("text-generation", "tiny", "auto")

        def generate(prompt, **kwargs):
            calls.append(kwargs)
            return [{"generated_text": prompt + " -> out"}]

        return generate

    monkeypatch.setattr(predict_mod, "pipeline", fake_pipeline)
    predictor = predict_mod.HFPredictor(
        model_name="tiny", temperature=0.5, top_p=0.9, max_tokens=16
    )

    result = predictor({"text": np.array(["a", "b"])})

    assert result == {"prompt": ["a", "b"], "generated_text": ["a -> out", "b -> out"]}
    assert calls[0]["max_new_tokens"] == 16
    assert calls[0]["temperature"] == 0.5
    assert calls[0]["top_p"] == 0.9


@pytest.fixture
def fake_vllm(monkeypatch):
    llm_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            llm_kwargs.update(kwargs)

        def generate(self, prompts, sampling_params):
            return [
                SimpleNamespace(
                    prompt=p,
                    outputs=[SimpleNamespace(text="x"), SimpleNamespace(text="y")],
                )
                for p in prompts
            ]

    module = ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.delitem(
        sys.modules, f"{_PIPELINES}.vllm_predict.predict", raising=False
    )
    return llm_kwargs


def test_vllm_predictor_joins_outputs_per_prompt(fake_vllm):
    predict_mod = importlib.import_module(f"{_PIPELINES}.vllm_predict.predict")
    predictor = predict_mod.VLLMPredictor(
        model_name="tiny",
        tensor_parallel_size=2,
        temperature=0.5,
        top_p=0.9,
        max_tokens=16,
    )

    result = predictor({"text": np.array(["a", "b"])})

    assert fake_vllm == {"model": "tiny", "tensor_parallel_size": 2}
    assert predictor.sampling_params == {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 16,
    }
    assert result == {"prompt": ["a", "b"], "generated_text": ["x y", "x y"]}


@pytest.mark.parametrize(
    ("backend", "version", "message"),
    [
        ("hf_predict", (3, 10), "requires Python 3.11+"),
        ("vllm_predict", (3, 10), "requires Python 3.11 or 3.12"),
        ("vllm_predict", (3, 13), "requires Python 3.11 or 3.12"),
    ],
)
def test_entry_point_rejects_unsupported_python(monkeypatch, backend, version, message):
    monkeypatch.setattr(sys, "version_info", _VersionInfo(*version, 0, "final", 0))

    with pytest.raises(RuntimeError, match=message):
        runpy.run_module(f"{_PIPELINES}.{backend}", run_name="__main__")
