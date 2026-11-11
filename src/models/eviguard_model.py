import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import logging
import warnings
from typing import Dict
from contextlib import contextmanager
import torch, torch.nn as nn
from transformers import BitsAndBytesConfig

warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*doesn't directly inherit from `GenerationMixin`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter in the current model.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*None of the inputs have requires_grad=True.*",
    category=UserWarning,
)
from peft import LoraConfig, get_peft_model, TaskType
from src.modules.rsf import (RSFModule, EvidenceEncoder,
                              ConnectorHookWithGrad, ConnectorInterventionHook)
from src.models.llava_utils import (
    load_llava_backbone,
    find_llava_connector_module,
    get_llava_embedding_layer,
    llava_forward,
)
from src.models.internvl_utils import (
    find_internvl_connector_module,
    get_internvl_embedding_layer,
)
from src.utils.helpers import get_common_value, get_quantization_kwargs, load_common_config

logger = logging.getLogger("eviguard.model")
_COMMON_CFG = load_common_config()


def _suppress_noisy_library_logs():
    noisy_prefixes = [
        "transformers_modules",
        "accelerate.utils.modeling",
        "timm",
    ]
    for name in list(logging.root.manager.loggerDict.keys()):
        if any(name.startswith(prefix) for prefix in noisy_prefixes):
            logging.getLogger(name).setLevel(logging.ERROR)


@contextmanager
def _maybe_cuda_device(device: str | None):
    if not torch.cuda.is_available() or not device or not str(device).startswith("cuda"):
        yield
        return
    prev_idx = torch.cuda.current_device()
    new_idx = int(str(device).split(":")[-1])
    torch.cuda.set_device(new_idx)
    try:
        yield
    finally:
        torch.cuda.set_device(prev_idx)


def _load_4bit_without_device_map(model_cls, model_name: str, bnb, target_device: str, *, trust_remote_code=False):
    common = dict(
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    if trust_remote_code:
        common["trust_remote_code"] = True

    load_attempts = [
        dict(common, low_cpu_mem_usage=True, device_map={"": target_device}),
        dict(common, low_cpu_mem_usage=True, device_map="auto"),
        dict(common, low_cpu_mem_usage=False),
    ]

    last_err = None
    with _maybe_cuda_device(target_device):
        for i, kwargs in enumerate(load_attempts, start=1):
            try:
                logger.info(
                    "4-bit load attempt %d for %s on %s (device_map=%s, low_cpu_mem_usage=%s)",
                    i, model_name, target_device, kwargs.get("device_map", None), kwargs.get("low_cpu_mem_usage"),
                )
                return model_cls.from_pretrained(model_name, **kwargs)
            except (NotImplementedError, ValueError) as e:
                msg = str(e)
                last_err = e
                if ("meta tensor" in msg or "low_cpu_mem_usage" in msg or "device_map" in msg):
                    logger.warning("4-bit load attempt %d failed, retrying with safer loader path: %s", i, e)
                    continue
                raise
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Unable to load 4-bit model: {model_name}")



def _maybe_enable_gradient_checkpointing(model):
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
    except Exception as e:
        logger.warning("Enable gradient checkpointing failed: %s", e)
    for obj in [model, getattr(model, "model", None), getattr(model, "base_model", None), getattr(model, "language_model", None)]:
        if obj is None:
            continue
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False


def _cast_trainable_params_fp32(model):
    casted = 0
    names = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.float()
            casted += p.numel()
            if len(names) < 12:
                names.append(n)
    logger.info("Cast trainable params to fp32: %d tensors%s",
                len(names), f" e.g. {names}" if names else "")
    return casted




def _load_llava(mcfg, lcfg, bnb, ref_dev, *, load_reference=True):
    return load_llava_backbone(
        mcfg,
        lcfg,
        bnb,
        ref_dev,
        cast_trainable_params_fp32=_cast_trainable_params_fp32,
        load_reference=load_reference,
    )


def _load_internvl(mcfg, lcfg, bnb, ref_dev, *, load_reference=True):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        mcfg["name"], trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    primary_device = get_common_value("runtime.device_policy.internvl_primary_device", "cuda:0", _COMMON_CFG)
    logger.info("InternVL: load 4-bit without device_map on %s (avoid meta init)", primary_device)
    model = _load_4bit_without_device_map(
        AutoModel, mcfg["name"], bnb, primary_device, trust_remote_code=True)
    if mcfg.get("freeze_vision", True):
        vision_mod = getattr(model, "vision_model", None) or getattr(model, "visual", None)
        if vision_mod is not None:
            for p in vision_mod.parameters():
                p.requires_grad = False
            logger.info("InternVL: vision_model frozen")

    from src.models.internvl_utils import build_internvl_lora_config, ensure_internvl_connector_output_dtype
    max_patches = int(mcfg.get("max_dynamic_patches", get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG)))
    setattr(model, "eviguard_max_dynamic_patches", max_patches)
    ensure_internvl_connector_output_dtype(model, target_dtype=torch.float16)

    lora = build_internvl_lora_config(model, lcfg, bias="none")
    model = get_peft_model(model, lora)
    _maybe_enable_gradient_checkpointing(model)
    model.print_trainable_parameters()

    mlp1_unfrozen = 0
    for n, p in model.named_parameters():
        if "mlp1" in n and p.dtype in (torch.float16, torch.float32, torch.float64):
            p.requires_grad = True
            mlp1_unfrozen += 1
    logger.info("InternVL: mlp1 connector unfrozen %d params", mlp1_unfrozen)
    _cast_trainable_params_fp32(model)

    ref = None
    if load_reference:
        logger.info("InternVL: load reference 4-bit without device_map on %s", ref_dev)
        ref = _load_4bit_without_device_map(
            AutoModel, mcfg["name"], bnb, ref_dev, trust_remote_code=True)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        ensure_internvl_connector_output_dtype(ref, target_dtype=torch.float16)
        setattr(ref, "eviguard_max_dynamic_patches", max_patches)

    from src.models.internvl_utils import _resolve_img_context_token_id, _set_internvl_img_context_token_id_by_id
    img_context_token_id = _resolve_img_context_token_id(tok)
    _set_internvl_img_context_token_id_by_id(model, img_context_token_id)
    if ref is not None:
        _set_internvl_img_context_token_id_by_id(ref, img_context_token_id)

    processor = tok
    return model, processor, tok, ref


def load_model_and_processor(cfg, *, load_reference=True):
    _suppress_noisy_library_logs()
    mcfg, lcfg = cfg["model"], cfg["lora"]
    arch = mcfg.get("arch", "llava")
    logger.info("加载模型: arch=%s  path=%s", arch, mcfg["name"])
    quantization = get_quantization_kwargs(cfg)
    bnb = BitsAndBytesConfig(**quantization) if quantization["load_in_4bit"] else None
    ref_dev = get_common_value("runtime.device_policy.ref_device_multi_gpu", "cuda:1", _COMMON_CFG) if torch.cuda.device_count() > 1 else get_common_value("runtime.device_policy.ref_device_single_gpu", "cuda:0", _COMMON_CFG)
    if arch == "internvl":
        model, processor, tok, ref = _load_internvl(mcfg, lcfg, bnb, ref_dev, load_reference=load_reference)
    else:
        model, processor, tok, ref = _load_llava(mcfg, lcfg, bnb, ref_dev, load_reference=load_reference)
    logger.info("模型加载完成: arch=%s", arch)
    return model, processor, tok, ref



def find_connector_module(model):
    llava_connector = find_llava_connector_module(model)
    if llava_connector is not None:
        return llava_connector

    internvl_connector = find_internvl_connector_module(model)
    if internvl_connector is not None:
        return internvl_connector

    for n, m in model.named_modules():
        name = n.lower()
        if "projector" in name or "vision_proj" in name or name.endswith("mlp1"):
            return m
    raise ValueError("找不到多模态 connector!")


def get_embedding_layer(model):
    llava_embedding = get_llava_embedding_layer(model)
    if llava_embedding is not None:
        return llava_embedding

    internvl_embedding = get_internvl_embedding_layer(model)
    if internvl_embedding is not None:
        return internvl_embedding

    for n, m in model.named_modules():
        if isinstance(m, nn.Embedding) and ("embed_tokens" in n or "tok_embeddings" in n):
            return m
    raise ValueError("找不到 embedding 层!")


class EviGuardModel(nn.Module):
    def __init__(self, model, ref_model, processor, tokenizer, cfg):
        super().__init__()
        self.model, self.ref_model = model, ref_model
        self.processor, self.tokenizer, self.cfg = processor, tokenizer, cfg
        self.arch = cfg["model"]["arch"]
        self.connector = find_connector_module(model)
        self.ref_connector = find_connector_module(ref_model) if ref_model is not None else None
        self.embed_layer = get_embedding_layer(model)
        hd = self._detect_hidden_dim()

        self.rsf = RSFModule(
            embed_dim=hd,
            bottleneck_dim=cfg["rsf"]["bottleneck_dim"],
            dropout=cfg["rsf"]["dropout"],
            gate_hidden_dim=get_common_value("rsf.gate_hidden_dim", _COMMON_CFG["rsf"]["gate_hidden_dim"], cfg),
        ).to(self._dev(model))
        self.evidence_encoder = EvidenceEncoder(tokenizer, cfg["rsf"]["evidence_max_tokens"])

        self.hook = ConnectorInterventionHook()
        self.hook.register(self.connector)
        self.ref_hook = ConnectorHookWithGrad() if self.ref_connector is not None else None
        if self.ref_hook is not None:
            self.ref_hook.register(self.ref_connector)

        logger.info("EviGuardModel v4 ok.  hidden_dim=%d  hook=InterventionHook", hd)

    @staticmethod
    def _dev(m):
        for t in list(m.parameters()) + list(m.buffers()): return t.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _to_dev(val, dev):
        if isinstance(val, torch.Tensor): return val.to(dev, non_blocking=True)
        if isinstance(val, dict): return {k: EviGuardModel._to_dev(v, dev) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return type(val)(EviGuardModel._to_dev(v, dev) for v in val)
        return val

    def _detect_hidden_dim(self):
        for c in (getattr(self.model, "config", None),
                  getattr(getattr(self.model, "base_model", None), "config", None),
                  getattr(getattr(self.model, "model", None), "config", None)):
            if c is None: continue
            lc = getattr(c, "llm_config", None)
            if lc and hasattr(lc, "hidden_size"): return int(lc.hidden_size)
            tc = getattr(c, "text_config", None)
            if tc and hasattr(tc, "hidden_size"): return int(tc.hidden_size)
            if hasattr(c, "hidden_size"): return int(c.hidden_size)
        return int(get_common_value("rsf.default_embed_dim", _COMMON_CFG["rsf"]["default_embed_dim"], self.cfg))

    def _forward_base(self, **kw):
        kw.pop("num_patches_list", None)
        if self.arch == "internvl":
            from src.models.internvl_utils import internvl_forward, _resolve_img_context_token_id
            kw.setdefault("img_context_token_id", _resolve_img_context_token_id(self.tokenizer))
            return internvl_forward(self.model, **kw)
        return llava_forward(self.model, **kw)

    def _forward_ref(self, **kw):
        kw.pop("num_patches_list", None)
        if self.arch == "internvl":
            from src.models.internvl_utils import internvl_forward, _resolve_img_context_token_id
            kw.setdefault("img_context_token_id", _resolve_img_context_token_id(self.tokenizer))
            return internvl_forward(self.ref_model, **kw)
        return llava_forward(self.ref_model, **kw)


    def _patch_counts(self, H, num_patches_list, sample_count=None):
        if H.ndim != 3:
            raise ValueError("connector activations must have shape [patches, tokens, hidden]")
        if self.arch != "internvl":
            if sample_count is not None and H.shape[0] != sample_count:
                raise ValueError("connector sample count does not match evidence/reliabilities")
            return [1] * H.shape[0]
        if num_patches_list is None:
            raise ValueError("InternVL requires num_patches_list for sample/patch correspondence")
        counts = list(num_patches_list)
        if any(isinstance(n, bool) or int(n) != n or n < 0 for n in counts):
            raise ValueError("num_patches_list must contain nonnegative integer patch counts")
        counts = [int(n) for n in counts]
        if sample_count is not None and len(counts) != sample_count:
            raise ValueError("patch list sample count does not match evidence/reliabilities")
        if sum(counts) != H.shape[0]:
            raise ValueError("sum(num_patches_list) does not match connector patch count")
        return counts

    def _rsf_modulation(self, H, evidence_strings, reliabilities, num_patches_list=None):
        if isinstance(evidence_strings, str):
            raise ValueError("evidence_strings must contain one string per sample")
        rel = torch.as_tensor(reliabilities, device=H.device)
        if rel.ndim != 1 or len(evidence_strings) != len(rel):
            raise ValueError("evidence/reliabilities must have the same sample count")
        counts = self._patch_counts(H, num_patches_list, len(evidence_strings))
        # InternVL batches concatenate patches across images. RSF uses one pooled
        # representation per example; text-only examples get zero visual evidence.
        rsf_H = self.pool_connector(H, counts).unsqueeze(1) if self.arch == "internvl" else H
        ev_emb = self.encode_evidence(evidence_strings, H.device)
        if ev_emb.shape != (len(counts), H.shape[-1]):
            raise ValueError("encoded evidence does not match sample count/hidden dimension")
        _, risk_scores, r_raw, delta = self.rsf(rsf_H, ev_emb, rel)
        addition = (r_raw[:, None] * delta).repeat_interleave(
            torch.tensor(counts, device=H.device), dim=0).unsqueeze(1)
        return addition, risk_scores, r_raw, delta

    def forward_with_rsf(self, evidence_strings, reliabilities, **model_kwargs):
        if model_kwargs.get("pixel_values") is None:
            self.hook.clear()
            self.hook.clear_replacement()
            out = self._forward_base(**model_kwargs)
            return {"logits": out.logits, "H_original": None, "H_fused": None,
                    "risk_scores": None, "r_raw": None, "logits_clean": None}

        self.hook.clear()
        self.hook.clear_replacement()
        with torch.no_grad():
            _ = self._forward_base(**model_kwargs)
        H_pass1 = self.hook.get()

        if H_pass1 is None:
            raise RuntimeError("visual input did not produce connector activations")
        addition, risk_scores, r_raw, _ = self._rsf_modulation(
            H_pass1.detach(), evidence_strings, reliabilities, model_kwargs.get("num_patches_list"))
        self.hook.clear()
        activations = {}

        def add_modulation(module, inputs, out):
            if out.shape != H_pass1.shape:
                raise ValueError("Pass2 connector shape differs from Pass1 patch mapping")
            activations["clean"] = out
            activations["fused"] = out + addition.to(device=out.device, dtype=out.dtype)
            return activations["fused"]

        # The existing capture hook sees the clean Pass2 tensor. Add the detached
        # evidence's trainable modulation to this freshly computed tensor (Eq.14).
        handle = self.connector.register_forward_hook(add_modulation)
        try:
            out_p2 = self._forward_base(**model_kwargs)
        finally:
            handle.remove()
            self.hook.clear()
        if "clean" not in activations:
            raise RuntimeError("Pass2 did not execute the multimodal connector")

        return {"logits": out_p2.logits, "H_original": activations["clean"], "H_fused": activations["fused"],
                "risk_scores": risk_scores, "r_raw": r_raw, "logits_clean": None}

    def forward_current(self, **kw):
        self.hook.clear()
        self.hook.clear_replacement()
        out = self._forward_base(**kw)
        return {"logits": out.logits, "connector_activation": self.hook.get()}

    def forward_reference(self, **kw):
        if self.ref_model is None or self.ref_hook is None:
            return {"logits": None, "connector_activation": None}
        self.ref_hook.clear()
        rd, td = self._dev(self.ref_model), self._dev(self.model)
        rkw = {k: self._to_dev(v, rd) for k, v in kw.items()}
        with torch.no_grad():
            out = self._forward_ref(**rkw)
        rH = self.ref_hook.get()
        logits = out.logits if out.logits.device == td else out.logits.to(td)
        if rH is not None and rH.device != td:
            rH = rH.to(td)
        return {"logits": logits, "connector_activation": rH}

    def encode_evidence(self, ev_str, dev):
        return self.evidence_encoder.encode(ev_str, self.embed_layer, dev)

    def compute_risk_scores(self, H, ev_str, rel, num_patches_list=None):
        counts = self._patch_counts(H, num_patches_list, len(ev_str))
        ev = self.encode_evidence(ev_str, H.device)
        rsf_H = self.pool_connector(H, counts)[:, None] if self.arch == "internvl" else H
        _, scores, raw, delta = self.rsf(rsf_H, ev, torch.as_tensor(rel, device=H.device))
        addition = (raw[:, None] * delta).repeat_interleave(torch.tensor(counts,device=H.device),dim=0)[:, None]
        return H + addition.to(H), scores, raw, delta

    def pool_connector(self, H, num_patches_list=None):
        counts = self._patch_counts(H, num_patches_list)
        if self.arch != "internvl":
            return H.mean(dim=1)
        pooled, offset = [], 0
        for count in counts:
            pooled.append(H[offset:offset + count].mean((0, 1)) if count else H.new_zeros(H.shape[-1]))
            offset += count
        return torch.stack(pooled)

    @contextmanager
    def inference_intervention(self, evidence_strings=None, reliabilities=None,
                               num_patches_list=None, num_beams=1):
        """Apply restored RSF during generate/chat, even after InternVL unwrapping.

        In eval/no_grad mode the evidence and injected connector are deterministic,
        so both can be computed from the current connector call without a second
        language-model forward. The hook is removed on success and on failure.
        """
        evidence_strings = [""] if evidence_strings is None else list(evidence_strings)
        if reliabilities is None:
            reliabilities = torch.full((len(evidence_strings),), float(
                get_common_value("rsf.default_reliability", .5, self.cfg)))

        def intervene(module, inputs, out):
            ev, rel, counts = evidence_strings, reliabilities, num_patches_list
            if self.arch == "internvl" and counts is None and len(ev) == 1:
                counts = [out.shape[0]]
            if self.arch != "internvl" and out.shape[0] == len(ev) * num_beams and num_beams > 1:
                ev = [s for s in ev for _ in range(num_beams)]
                rel = torch.as_tensor(rel).repeat_interleave(num_beams)
            addition, _, _, _ = self._rsf_modulation(out.detach(), ev, rel, counts)
            return out + addition.to(out)

        self.hook.clear()
        handle = self.connector.register_forward_hook(intervene)
        try:
            yield
        finally:
            handle.remove()
            self.hook.clear()

    @contextmanager
    def _generation_precision(self):
        if self._dev(self.model).type == "cuda":
            dtype=getattr(torch,self.cfg.get("model",{}).get("torch_dtype","float16"))
            with torch.autocast("cuda",dtype=dtype):
                yield
        else:
            yield

    @torch.no_grad()
    def generate(self, *, evidence_strings=None, reliabilities=None, num_patches_list=None, **kwargs):
        with self.inference_intervention(evidence_strings, reliabilities, num_patches_list,
                                         kwargs.get("num_beams", 1)):
            with self._generation_precision():
                return self.model.generate(**kwargs)

    @torch.no_grad()
    def generate_response(self, image, question, evidence_strings=None, reliabilities=None, **kwargs):
        if self.arch == "internvl":
            from src.models.internvl_utils import internvl_generate_response
            kwargs.pop("num_beams", None)  # Existing InternVL chat API exposes no beam option.
            with self._generation_precision(), self.inference_intervention(evidence_strings, reliabilities):
                return internvl_generate_response(self.model, self.tokenizer, image, question,
                                                   device=self._dev(self.model), **kwargs)
        system = get_common_value("runtime.llava.system_msg", "", self.cfg)
        prompt = f"{system}USER: {'<image>' + chr(10) if image is not None else ''}{question} ASSISTANT: "
        inputs = (self.processor(text=prompt, images=image, return_tensors="pt") if image is not None
                  else self.tokenizer(prompt, return_tensors="pt"))
        inputs = self._to_dev(dict(inputs), self._dev(self.model))
        output = self.generate(**inputs, evidence_strings=evidence_strings, reliabilities=reliabilities, **kwargs)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad] + \
               [p for p in self.rsf.parameters() if p.requires_grad]

    def save_rsf(self, path): torch.save(self.rsf.state_dict(), path)
    def load_rsf(self, path): self.rsf.load_state_dict(torch.load(path, map_location="cpu"))
