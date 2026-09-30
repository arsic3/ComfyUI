import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.modules.diffusionmodules.model as vae_model  # noqa: E402

try:
    torch.ops.torchvision.nms
except AttributeError:
    _torchvision_library = torch.library.Library("torchvision", "DEF")
    _torchvision_library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    _torchvision_library.define("qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")

import comfy.sd as comfy_sd  # noqa: E402


def test_vae_attention_backend_selection(monkeypatch):
    expected = {
        "split": vae_model.normal_attention,
        "default-sdpa": vae_model.pytorch_attention_default,
        "flash": vae_model.pytorch_attention_flash,
        "efficient": vae_model.pytorch_attention_efficient,
        "math": vae_model.pytorch_attention_math,
    }
    for backend, function in expected.items():
        monkeypatch.setattr(vae_model.model_management, "vae_attention_backend", lambda backend=backend: backend)
        assert vae_model.vae_attention() is function


def test_default_vae_attention_selection_is_unchanged(monkeypatch):
    monkeypatch.setattr(vae_model.model_management, "vae_attention_backend", lambda: None)
    monkeypatch.setattr(vae_model.model_management, "xformers_enabled_vae", lambda: False)
    monkeypatch.setattr(vae_model.model_management, "pytorch_attention_enabled_vae", lambda: False)
    assert vae_model.vae_attention() is vae_model.normal_attention


@pytest.mark.parametrize(
    ("function", "backend"),
    ((vae_model.pytorch_attention_flash, torch.nn.attention.SDPBackend.FLASH_ATTENTION),
     (vae_model.pytorch_attention_efficient, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
     (vae_model.pytorch_attention_math, torch.nn.attention.SDPBackend.MATH)),
)
def test_forced_sdpa_uses_single_backend_without_fallback(monkeypatch, function, backend):
    q = torch.randn(1, 4, 2, 2)
    context = patch("torch.nn.attention.sdpa_kernel")
    sdpa = patch("torch.nn.functional.scaled_dot_product_attention", side_effect=RuntimeError("unsupported"))
    split = patch.object(vae_model, "slice_attention")
    monkeypatch.setattr(vae_model.model_management, "vae_attention_diagnostics", lambda: False)

    with context as kernel, sdpa, split as split_attention, pytest.raises(RuntimeError, match="unsupported"):
        function(q, q, q)

    kernel.assert_called_once_with(backend)
    split_attention.assert_not_called()


def test_attention_diagnostics_are_json_and_report_reshaped_tensors(monkeypatch):
    q = torch.randn(1, 4, 2, 2)
    messages = []
    monkeypatch.setattr(vae_model.model_management, "vae_attention_diagnostics", lambda: True)
    monkeypatch.setattr(vae_model, "detail", lambda message, value: messages.append((message, value)))

    output = vae_model.pytorch_attention(q, q, q)

    assert output.shape == q.shape
    assert len(messages) == 1
    assert messages[0][0] == "VAE_BENCH %s"
    event = json.loads(messages[0][1])
    assert event["backend"] == "automatic-sdpa"
    assert event["requested_mode"] is None
    assert event["q"] == {"shape": [1, 1, 4, 4], "dtype": "torch.float32", "strides": [16, 16, 4, 1]}
    assert isinstance(event["flash_eligible"], bool)
    assert isinstance(event["efficient_eligible"], bool)
    assert event["output"] == {"shape": [1, 4, 2, 2], "dtype": "torch.float32", "numel": 16}
    assert event["elapsed_ms"] >= 0


def test_explicit_default_sdpa_has_no_split_fallback(monkeypatch):
    q = torch.randn(1, 4, 2, 2)
    monkeypatch.setattr(vae_model.model_management, "vae_attention_diagnostics", lambda: False)

    with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=RuntimeError("no viable backend")), patch.object(vae_model, "slice_attention") as split_attention, pytest.raises(RuntimeError, match="no viable backend"):
        vae_model.pytorch_attention_default(q, q, q)

    split_attention.assert_not_called()


def test_no_viable_backend_is_unsupported():
    assert vae_model._vae_exception_category(RuntimeError("No viable backend for scaled_dot_product_attention")) == "unsupported"


def test_attention_diagnostics_do_not_reset_allocator_peak(monkeypatch):
    monkeypatch.setattr(vae_model.torch.cuda, "synchronize", lambda device: None)
    reset = patch.object(vae_model.torch.cuda, "reset_peak_memory_stats")

    with reset as reset_peak:
        vae_model._vae_attention_start(torch.device("cuda"))

    reset_peak.assert_not_called()


def _vae_for_fallback_test(operation):
    model = SimpleNamespace(comfy_has_chunked_io=False)
    setattr(model, operation, Mock(side_effect=torch.cuda.OutOfMemoryError("oom")))
    vae = SimpleNamespace(
        throw_exception_if_invalid=Mock(), latent_dim=2, extra_1d_channel=None, not_video=True,
        device=torch.device("cpu"), output_device=torch.device("cpu"), vae_dtype=torch.float32,
        disable_offload=False, patcher=SimpleNamespace(get_free_memory=lambda device: 2**40),
        first_stage_model=model, process_input=lambda value: value, process_output=lambda value: value,
        vae_output_dtype=lambda: torch.float32, format_encoded=None, handles_tiling=False,
        memory_used_decode=lambda shape, dtype: 1, memory_used_encode=lambda shape, dtype: 1,
        vae_encode_crop_pixels=lambda value: value,
        decode_tiled_=Mock(return_value=torch.zeros(1, 3, 2, 2)),
        encode_tiled_=Mock(return_value=torch.zeros(1, 4, 1, 1)),
    )
    return vae


@pytest.mark.parametrize("operation", ("decode", "encode"))
def test_explicit_modes_disable_automatic_tiled_fallback(monkeypatch, operation):
    vae = _vae_for_fallback_test(operation)
    monkeypatch.setattr(comfy_sd.model_management, "vae_attention_backend", lambda: "math")
    monkeypatch.setattr(comfy_sd.model_management, "cuda_device_context", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(comfy_sd.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy_sd.comfy.model_prefetch, "pause_malloc_graph", contextlib.nullcontext)
    function = getattr(comfy_sd.VAE, operation).__wrapped__
    value = torch.zeros(1, 4, 2, 2) if operation == "decode" else torch.zeros(1, 2, 2, 3)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="oom"):
        function(vae, value)

    vae.decode_tiled_.assert_not_called()
    vae.encode_tiled_.assert_not_called()


@pytest.mark.parametrize("operation", ("decode", "encode"))
def test_omitted_backend_preserves_automatic_tiled_fallback(monkeypatch, operation):
    vae = _vae_for_fallback_test(operation)
    monkeypatch.setattr(comfy_sd.model_management, "vae_attention_backend", lambda: None)
    monkeypatch.setattr(comfy_sd.model_management, "cuda_device_context", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(comfy_sd.model_management, "load_models_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy_sd.model_management, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(comfy_sd.comfy.model_prefetch, "pause_malloc_graph", contextlib.nullcontext)
    function = getattr(comfy_sd.VAE, operation).__wrapped__
    value = torch.zeros(1, 4, 2, 2) if operation == "decode" else torch.zeros(1, 2, 2, 3)

    output = function(vae, value)

    getattr(vae, operation + "_tiled_").assert_called_once()
    assert torch.is_tensor(output)


def test_operation_diagnostics_reraises_deferred_synchronize_failure(monkeypatch):
    messages = []
    synchronize = Mock(side_effect=[None, RuntimeError("deferred failure")])
    monkeypatch.setattr(comfy_sd.model_management, "vae_attention_diagnostics", lambda: True)
    monkeypatch.setattr(comfy_sd.model_management, "vae_attention_backend", lambda: "math")
    monkeypatch.setattr(comfy_sd.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(comfy_sd.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(comfy_sd.torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(comfy_sd, "detail", lambda message, value: messages.append(json.loads(value)))
    wrapped = comfy_sd._vae_operation_diagnostics("decode")(lambda self: torch.ones(1))

    with pytest.raises(RuntimeError, match="deferred failure"):
        wrapped(SimpleNamespace(device=torch.device("cuda")))

    assert messages[0]["deferred_exception"]["message"] == "deferred failure"


def test_operation_diagnostics_preserves_primary_exception(monkeypatch):
    synchronize = Mock(side_effect=[None, RuntimeError("deferred failure")])
    monkeypatch.setattr(comfy_sd.model_management, "vae_attention_diagnostics", lambda: True)
    monkeypatch.setattr(comfy_sd.model_management, "vae_attention_backend", lambda: "math")
    monkeypatch.setattr(comfy_sd.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(comfy_sd.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(comfy_sd.torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(comfy_sd, "detail", lambda message, value: None)
    wrapped = comfy_sd._vae_operation_diagnostics("decode")(Mock(side_effect=ValueError("primary failure")))

    with pytest.raises(ValueError, match="primary failure"):
        wrapped(SimpleNamespace(device=torch.device("cuda")))


def test_operation_metrics_are_captured_before_output_validation():
    source = (Path(__file__).parents[2] / "comfy" / "sd.py").read_text()
    elapsed = source.index('event["elapsed_ms"] = (time.perf_counter() - started) * 1000')
    validation = source.index("validation_started = time.perf_counter()", elapsed)
    peak = source.rindex('event["peak_allocated"]', 0, elapsed)
    reductions = source.index("output_min, output_max = torch.aminmax(output)", validation)

    assert peak < elapsed < validation < reductions
    assert "torch.count_nonzero(output)" in source[reductions:]
    assert "torch.isfinite(output)" not in source[validation:]
    assert "output[finite]" not in source[validation:]
    assert 'event["validation_ms"]' in source[reductions:]


def test_forced_backend_exception_has_normalized_category(monkeypatch):
    q = torch.randn(1, 4, 2, 2)
    messages = []
    monkeypatch.setattr(vae_model.model_management, "vae_attention_backend", lambda: "flash")
    monkeypatch.setattr(vae_model.model_management, "vae_attention_diagnostics", lambda: True)
    monkeypatch.setattr(vae_model, "detail", lambda message, value: messages.append(json.loads(value)))

    with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=RuntimeError("unsupported kernel")), pytest.raises(RuntimeError):
        vae_model.pytorch_attention_flash(q, q, q)

    assert messages[0]["requested_mode"] == "flash"
    assert messages[0]["exception"]["category"] == "unsupported"
    assert "allocator_peak_observed" in messages[0]


@pytest.mark.parametrize("backend", ("split", "default-sdpa", "math"))
def test_multi_head_vae_attention_matches_sdpa(monkeypatch, backend):
    monkeypatch.setattr(vae_model.model_management, "vae_attention_diagnostics", lambda: False)
    monkeypatch.setattr(vae_model.model_management, "get_free_memory", lambda device: 2**40)
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    v = torch.randn(2, 3, 5, 4)

    output = vae_model.vae_attention_bhsd(q, k, v, backend)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    assert output.shape == q.shape
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)


def test_minimax_explicit_backend_routes_through_vae_attention(monkeypatch):
    import comfy.ldm.minimax.vae as minimax_vae

    monkeypatch.setattr(minimax_vae.comfy.model_management, "vae_attention_backend", lambda: "split")
    routed = patch.object(minimax_vae, "vae_attention_bhsd", side_effect=lambda q, k, v, backend: v)
    optimized = patch.object(minimax_vae, "optimized_attention")
    attention = minimax_vae.Attention(heads=2, dim_head=4)

    with routed as routed_attention, optimized as optimized_attention, torch.no_grad():
        attention(torch.randn(1, 3, 8), None, pre_norm=attention.norm_q, residual=None, residual_scale=None)

    routed_attention.assert_called_once()
    assert routed_attention.call_args.args[3] == "split"
    assert routed_attention.call_args.args[0].shape == (1, 2, 3, 4)
    optimized_attention.assert_not_called()


def test_minimax_omitted_backend_preserves_optimized_route(monkeypatch):
    import comfy.ldm.minimax.vae as minimax_vae

    monkeypatch.setattr(minimax_vae.comfy.model_management, "vae_attention_backend", lambda: None)
    monkeypatch.setattr(minimax_vae, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False)
    routed = patch.object(minimax_vae, "vae_attention_bhsd")
    optimized = patch.object(minimax_vae, "optimized_attention", side_effect=lambda q, k, v, heads, skip_reshape: v.transpose(1, 2).reshape(1, 3, 8))
    attention = minimax_vae.Attention(heads=2, dim_head=4)

    with routed as routed_attention, optimized as optimized_attention, torch.no_grad():
        attention(torch.randn(1, 3, 8), None, pre_norm=attention.norm_q, residual=None, residual_scale=None)

    routed_attention.assert_not_called()
    optimized_attention.assert_called_once()


def test_minimax_explicit_backend_supersedes_int8_route(monkeypatch):
    import comfy.ldm.minimax.vae as minimax_vae

    attention = minimax_vae.Attention(heads=2, dim_head=4)
    quantized_type = type("TestQuantizedTensor", (), {})
    monkeypatch.setattr(minimax_vae.comfy.model_management, "vae_attention_backend", lambda: "default-sdpa")
    monkeypatch.setattr(minimax_vae, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(minimax_vae.comfy.ops, "QuantizedTensor", quantized_type)
    attention.to_qkv._parameters.pop("weight")
    attention.to_qkv.weight = quantized_type()
    monkeypatch.setattr(minimax_vae.comfy.ops, "linear_input_act", Mock(side_effect=[torch.randn(1, 3, 24), torch.randn(1, 3, 8)]))
    int8_attention = Mock(side_effect=AssertionError("int8 route used"))
    monkeypatch.setattr(minimax_vae.comfy.quant_ops, "ck", Mock(int8_attention=int8_attention))

    with patch.object(minimax_vae, "vae_attention_bhsd", side_effect=lambda q, k, v, backend: v) as routed_attention, torch.no_grad():
        attention(torch.randn(1, 3, 8), None, pre_norm=attention.norm_q, residual=None, residual_scale=None)

    routed_attention.assert_called_once()
    int8_attention.assert_not_called()
