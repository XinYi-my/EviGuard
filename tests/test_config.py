from src.utils.helpers import apply_runtime_path_overrides, redact_config

def test_explicit_config_model_survives_without_environment(monkeypatch):
    monkeypatch.delenv("LLAVA_MODEL_PATH", raising=False)
    cfg=apply_runtime_path_overrides({"model":{"arch":"llava","name":"/models/my-llava"}})
    assert cfg["model"]["name"] == "/models/my-llava"

def test_secrets_not_persisted():
    clean=redact_config({"evaluation":{"api_key":"test-sensitive-value","api_url":"https://example.invalid"},"token":"secret","rsf":{"bottleneck_dim":512}})
    assert "test-sensitive-value" not in str(clean)
    assert clean["rsf"]["bottleneck_dim"]==512


def test_quantization_preserves_trainable_connector():
    from src.utils.helpers import get_quantization_kwargs
    cfg={"model":{"arch":"llava"},"runtime":{"quantization":{"load_in_4bit":True}}}
    assert "multi_modal_projector" in get_quantization_kwargs(cfg)["llm_int8_skip_modules"]
