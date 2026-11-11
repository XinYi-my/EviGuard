from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("eviguard.llava_utils")


def resolve_llava_submodules(model):
    core = getattr(model, "model", model)
    vision_tower = getattr(core, "vision_tower", None)
    if vision_tower is None:
        vision_tower = getattr(model, "vision_tower", None)
    projector = getattr(core, "multi_modal_projector", None)
    if projector is None:
        projector = getattr(model, "multi_modal_projector", None)
    return core, vision_tower, projector


def language_lora_targets(model, requested):
    """Use complete language-module paths, excluding every vision module."""
    targets=[name for name,module in model.named_modules()
             if "language_model" in name.split(".") and name.split(".")[-1] in requested
             and isinstance(module,nn.Linear)]
    if not targets:
        raise ValueError("No language-model LoRA targets match this backbone")
    return targets


def configure_llava_processor(processor, config):
    """Match modern Transformers visual-token expansion to the saved backbone."""
    processor.patch_size = config.vision_config.patch_size
    processor.vision_feature_select_strategy = config.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1 if "clip" in str(getattr(config.vision_config,"model_type","clip")) else 0


def load_llava_backbone(mcfg, lcfg, bnb, ref_dev, cast_trainable_params_fp32, *, load_reference=True):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import LoraConfig, TaskType, get_peft_model

    processor = AutoProcessor.from_pretrained(mcfg["name"], use_fast=False)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = LlavaForConditionalGeneration.from_pretrained(
        mcfg["name"],
        quantization_config=bnb,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
    )
    if getattr(model.config, "loss_type", None) is None:
        model.config.loss_type = "ForCausalLM"

    configure_llava_processor(processor, model.config)
    _, vision_tower, projector = resolve_llava_submodules(model)
    if mcfg.get("freeze_vision", True):
        if vision_tower is None:
            raise AttributeError("Cannot find vision_tower on loaded LLaVA model.")
        for p in vision_tower.parameters():
            p.requires_grad = False
    if projector is None:
        raise AttributeError("Cannot find multi_modal_projector on loaded LLaVA model.")

    lora = LoraConfig(
        r=lcfg["rank"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=language_lora_targets(model,lcfg["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    for n, p in model.named_parameters():
        if "multi_modal_projector" in n and p.dtype in (torch.float16, torch.float32, torch.float64):
            p.requires_grad = True
    cast_trainable_params_fp32(model)

    if not load_reference:
        return model, processor, tok, None

    ref = LlavaForConditionalGeneration.from_pretrained(
        mcfg["name"],
        quantization_config=bnb,
        torch_dtype=torch.float16,
        device_map={"": ref_dev},
    )
    if getattr(ref.config, "loss_type", None) is None:
        ref.config.loss_type = "ForCausalLM"
    _, ref_vision_tower, ref_projector = resolve_llava_submodules(ref)
    if mcfg.get("freeze_vision", True) and ref_vision_tower is None:
        raise AttributeError("Cannot find vision_tower on reference LLaVA model.")
    if ref_projector is None:
        raise AttributeError("Cannot find multi_modal_projector on reference LLaVA model.")

    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return model, processor, tok, ref


def find_llava_connector_module(model) -> Optional[nn.Module]:
    for path in (
        "base_model.model.multi_modal_projector",
        "model.multi_modal_projector",
        "multi_modal_projector",
    ):
        mod = model
        try:
            for p in path.split("."):
                mod = getattr(mod, p)
            return mod
        except AttributeError:
            pass
    for n, m in model.named_modules():
        if "multi_modal_projector" in n or n.endswith("projector"):
            return m
    return None


def get_llava_embedding_layer(model) -> Optional[nn.Embedding]:
    candidate_paths = (
        "base_model.model.language_model.model.embed_tokens",
        "model.language_model.model.embed_tokens",
        "language_model.model.embed_tokens",
        "base_model.model.model.embed_tokens",
        "model.model.embed_tokens",
        "model.embed_tokens",
        "embed_tokens",
    )
    for path in candidate_paths:
        mod = model
        try:
            for p in path.split("."):
                mod = getattr(mod, p)
            if isinstance(mod, nn.Embedding):
                return mod
        except AttributeError:
            pass

    for obj in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if obj is None:
            continue
        getter = getattr(obj, "get_input_embeddings", None)
        if callable(getter):
            try:
                emb = getter()
            except Exception:
                emb = None
            if isinstance(emb, nn.Embedding):
                return emb
    return None


def llava_forward(model, **kwargs):
    return model(**kwargs)
