"""Complete, versioned EviGuard state; adapter-only folders are not sufficient.

The immutable backbone still comes from resolved_config.model.name. All trained
backbone tensors (including LoRA), connector buffers, RSF, and subspace state are
saved explicitly. A manifest verifies every component before restoration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """A complete checkpoint is missing, corrupt, or incompatible."""


def _cpu_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, default=str) + "\n", encoding="utf-8")


def _subspace_state(subspace):
    if subspace is None:
        return None
    state = {}
    for name in ("basis", "centroid", "eigenvalues", "explained_var", "rank_k",
                 "min_fit_samples", "is_fitted"):
        value = getattr(subspace, name)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value.copy())
        elif torch.is_tensor(value):
            value = value.detach().cpu().clone()
        state[name] = value
    return state


def save_checkpoint(eg, output_dir, *, subspace=None, resolved_config=None,
                    manifest=None, training_state=None):
    """Save inference-complete state, with optional trainer/RNG/optimizer state.

    Returns the written manifest for the supplied EviGuardModel.
    """
    from peft import PeftModel

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if isinstance(eg.model, PeftModel):
        eg.model.save_pretrained(path / "adapter", safe_serialization=True)
        adapter_kind = "peft"
    else:
        adapter_kind = "trainable_state"
    trained = {name: tensor for name, tensor in eg.model.named_parameters() if tensor.requires_grad}
    torch.save(_cpu_state(trained), path / "trainable.pt")
    torch.save(_cpu_state(eg.connector.state_dict()), path / "connector.pt")
    torch.save(_cpu_state(eg.rsf.state_dict()), path / "rsf.pt")
    torch.save(_subspace_state(subspace), path / "subspace.pt")
    from src.utils.helpers import redact_config
    _json_dump(path / "resolved_config.json", redact_config(resolved_config if resolved_config is not None else eg.cfg))
    filenames = ["trainable.pt", "connector.pt", "rsf.pt", "subspace.pt", "resolved_config.json"]
    if adapter_kind == "peft":
        filenames.extend(str(p.relative_to(path)).replace("\\", "/")
                         for p in (path / "adapter").rglob("*") if p.is_file())
    if training_state is not None:
        torch.save(training_state, path / "training_state.pt")
        filenames.append("training_state.pt")
    payload = {**(manifest or {}), "format": "eviguard-complete", "format_version": FORMAT_VERSION,
               "arch": eg.arch, "adapter_kind": adapter_kind,
               "trainable_names": sorted(trained),
               "files": {name: _digest(path / name) for name in sorted(filenames)},
               "has_training_state": training_state is not None}
    _json_dump(path / "manifest.json", payload)
    return payload


def read_manifest(checkpoint_dir):
    """Verify required components before model loading or any state mutation."""
    path = Path(checkpoint_dir)
    try:
        payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CheckpointError(f"Complete EviGuard checkpoint manifest missing/invalid: {path}") from error
    if payload.get("format") != "eviguard-complete" or payload.get("format_version") != FORMAT_VERSION:
        raise CheckpointError("Unsupported or incomplete EviGuard checkpoint format")
    files = payload.get("files", {})
    required = {"trainable.pt", "connector.pt", "rsf.pt", "subspace.pt", "resolved_config.json"}
    if payload.get("adapter_kind") == "peft":
        required |= {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"}
    if payload.get("has_training_state"):
        required.add("training_state.pt")
    if not required.issubset(files):
        raise CheckpointError(f"Manifest omits required components: {sorted(required - set(files))}")
    for name, expected in files.items():
        component = (path / name).resolve()
        if not component.is_relative_to(path.resolve()):
            raise CheckpointError(f"Invalid checkpoint component path: {name}")
        if not component.is_file():
            raise CheckpointError(f"Checkpoint component missing: {name}")
        if _digest(component) != expected:
            raise CheckpointError(f"Checkpoint digest mismatch: {name}")
    return payload


def _tensor_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"Invalid tensor checkpoint component: {Path(path).name}") from error


def _validate_state(actual, expected, name):
    if set(actual) != set(expected):
        raise CheckpointError(f"{name} tensor names differ from the configured model")
    for key, value in actual.items():
        if not torch.is_tensor(value) or value.shape != expected[key].shape:
            raise CheckpointError(f"{name} shape mismatch: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise CheckpointError(f"{name} contains non-finite values: {key}")


def restore_checkpoint(eg, checkpoint_dir, *, subspace=None, load_training_state=True):
    """Restore into the same configured backbone/PEFT architecture, strictly.

    Returns config, manifest, subspace, and optional training_state. The caller
    restores optimizer/scheduler/scaler/RNG from training_state when resuming.
    """
    path = Path(checkpoint_dir)
    manifest = read_manifest(path)
    if manifest["arch"] != eg.arch:
        raise CheckpointError("Checkpoint architecture does not match the model")
    config = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
    if config.get("model", {}).get("name") != eg.cfg.get("model", {}).get("name"):
        raise CheckpointError("Checkpoint backbone identity does not match the model config")
    trained = _tensor_load(path / "trainable.pt")
    connector = _tensor_load(path / "connector.pt")
    rsf = _tensor_load(path / "rsf.pt")
    expected = {n: p for n, p in eg.model.named_parameters() if p.requires_grad}
    _validate_state(trained, expected, "trainable")
    _validate_state(connector, eg.connector.state_dict(), "connector")
    _validate_state(rsf, eg.rsf.state_dict(), "rsf")
    subspace_state = _tensor_load(path / "subspace.pt")
    if subspace_state is not None:
        from src.modules.clsu import RiskSubspace
        if subspace is None:
            subspace = RiskSubspace(subspace_state["rank_k"], device="cpu",
                                    min_fit_samples=subspace_state["min_fit_samples"])
        basis, centroid = subspace_state["basis"], subspace_state["centroid"]
        if subspace_state["is_fitted"]:
            if (basis is None or centroid is None or basis.ndim != 2 or centroid.shape != (basis.shape[0],)
                    or not torch.isfinite(basis).all() or not torch.isfinite(centroid).all()
                    or basis.shape[0] != eg.rsf.embed_dim):
                raise CheckpointError("Invalid fitted subspace dimensions or values")
        if subspace_state["is_fitted"]:
            RiskSubspace._validate_geometry(basis, centroid, subspace_state["rank_k"])
        for name, value in subspace_state.items():
            if name == "eigenvalues" and torch.is_tensor(value):
                value = value.numpy()
            elif name in ("basis", "centroid") and value is not None:
                value = value.to(subspace.device)
            setattr(subspace, name, value)
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(trained[name].to(parameter))
    eg.connector.load_state_dict(connector, strict=True)
    eg.rsf.load_state_dict(rsf, strict=True)
    # This fingerprint identifies all weights, resolved configuration and protocol
    # metadata, and can safely distinguish generation response caches.
    eg.checkpoint_identity = _digest(path / "manifest.json")
    eg.risk_subspace = subspace
    training_state = None
    if load_training_state and manifest.get("has_training_state"):
        # Optimizer and Python/NumPy RNG payloads need the full local torch format.
        training_state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    return {"manifest": manifest, "config": config, "subspace": subspace, "training_state": training_state}


def load_inference_checkpoint(checkpoint_dir, *, model_loader=None, backbone_path=None):
    """Return a restored EviGuardModel with generate and generate_response.

    Optional model_loader(cfg) returns (backbone, processor, tokenizer, None),
    allowing offline/local model factories and CPU regression tests.
    """
    from src.models.eviguard_model import EviGuardModel, load_model_and_processor

    path = Path(checkpoint_dir)
    read_manifest(path)
    cfg = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
    from copy import deepcopy
    loading_cfg = deepcopy(cfg)
    if backbone_path is not None:
        loading_cfg["model"]["name"] = str(backbone_path)
    loaded = (model_loader(loading_cfg) if model_loader is not None
              else load_model_and_processor(loading_cfg, load_reference=False))
    model, processor, tokenizer, _ = loaded
    eg = EviGuardModel(model, None, processor, tokenizer, cfg)
    restore_checkpoint(eg, path, load_training_state=False)
    eg.eval()
    return eg
