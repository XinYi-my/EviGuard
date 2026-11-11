import logging
import math
import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from src.utils.helpers import load_common_config, get_common_value

logger = logging.getLogger("eviguard.internvl_utils")
_COMMON_CFG = load_common_config()


def _direct_getattr(obj, name, default=None):
    if obj is None:
        return default
    try:
        return object.__getattribute__(obj, name)
    except Exception:
        pass

    if isinstance(obj, nn.Module):
        for store_name in ("_modules", "_parameters", "_buffers"):
            try:
                store = object.__getattribute__(obj, store_name)
            except Exception:
                store = None
            if isinstance(store, dict) and name in store:
                return store[name]

    return default


def _direct_hasattr(obj, name):
    sentinel = object()
    return _direct_getattr(obj, name, sentinel) is not sentinel


def _resolve_img_context_token_id(tokenizer):
    token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id

    added_vocab = {}
    if hasattr(tokenizer, "get_added_vocab"):
        try:
            added_vocab = tokenizer.get_added_vocab() or {}
        except Exception:
            added_vocab = {}
    token_id = added_vocab.get(IMG_CONTEXT_TOKEN)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id

    vocab = {}
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab() or {}
        except Exception:
            vocab = {}
    token_id = vocab.get(IMG_CONTEXT_TOKEN)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id

    raise ValueError(f"Tokenizer 中找不到 {IMG_CONTEXT_TOKEN}，无法为 InternVL 设置 img_context_token_id")


IMAGENET_MEAN = tuple(get_common_value("runtime.internvl.imagenet_mean", cfg=_COMMON_CFG))
IMAGENET_STD  = tuple(get_common_value("runtime.internvl.imagenet_std", cfg=_COMMON_CFG))


def build_transform(input_size=None):
    if input_size is None:
        input_size = int(get_common_value("runtime.internvl.input_size", cfg=_COMMON_CFG))
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=None, max_num=None, image_size=None, use_thumbnail=None):
    if min_num is None:
        min_num = int(get_common_value("runtime.internvl.dynamic_min_num", cfg=_COMMON_CFG))
    if max_num is None:
        max_num = int(get_common_value("runtime.internvl.dynamic_max_num", cfg=_COMMON_CFG))
    if image_size is None:
        image_size = int(get_common_value("runtime.internvl.input_size", cfg=_COMMON_CFG))
    if use_thumbnail is None:
        use_thumbnail = bool(get_common_value("runtime.internvl.use_thumbnail", cfg=_COMMON_CFG))
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl(image: Image.Image, max_num=None, input_size=None):
    if max_num is None:
        max_num = int(get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG) or 2)
    if input_size is None:
        input_size = int(get_common_value("runtime.internvl.input_size", cfg=_COMMON_CFG) or 448)
    use_thumbnail = bool(get_common_value("runtime.internvl.use_thumbnail", cfg=_COMMON_CFG))
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)  # [N, 3, 448, 448]
    return pixel_values, len(pixel_values)


class _FloatingOutputCastWrapper(nn.Module):

    def __init__(self, module: nn.Module, target_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.module = module
        self.target_dtype = target_dtype
        self._eviguard_output_cast_wrapper = True

    def _cast(self, x):
        if torch.is_tensor(x):
            if x.is_floating_point() and x.dtype != self.target_dtype:
                return x.to(dtype=self.target_dtype)
            return x
        if isinstance(x, tuple):
            return tuple(self._cast(v) for v in x)
        if isinstance(x, list):
            return [self._cast(v) for v in x]
        if isinstance(x, dict):
            return {k: self._cast(v) for k, v in x.items()}
        return x

    def _module_input_dtype(self):
        try:
            for p in self.module.parameters():
                if torch.is_floating_point(p):
                    return p.dtype
        except Exception:
            pass
        return self.target_dtype

    def forward(self, *args, **kwargs):
        in_dtype = self._module_input_dtype()
        old_dtype = self.target_dtype
        self.target_dtype = in_dtype
        args = self._cast(args)
        kwargs = self._cast(kwargs)
        self.target_dtype = old_dtype
        return self._cast(self.module(*args, **kwargs))


def _infer_internvl_lm_dtype(model, fallback=None, input_ids=None):
    if fallback is None:
        fallback = torch.float16
    core = unwrap_internvl_model(model) if model is not None else None
    for obj in (core, model):
        if obj is None:
            continue
        lm = _direct_getattr(obj, 'language_model')
        if lm is None or not hasattr(lm, 'get_input_embeddings'):
            continue
        try:
            emb = lm.get_input_embeddings()
            if emb is None:
                continue
            if input_ids is not None:
                probe = input_ids[:1, :1] if getattr(input_ids, 'ndim', 0) >= 2 else input_ids[:1]
                with torch.no_grad():
                    probe_out = emb(probe)
                if torch.is_tensor(probe_out) and probe_out.is_floating_point():
                    return probe_out.dtype
            if hasattr(emb, 'weight'):
                return emb.weight.dtype
        except Exception:
            pass
    return fallback


def _patch_internvl_extract_feature_dtype(core, target_dtype):
    if core is None:
        return False

    setattr(core, '_eviguard_extract_target_dtype', target_dtype)
    if getattr(core, '_eviguard_extract_feature_patched', False):
        return True

    orig_extract_feature = _direct_getattr(core, 'extract_feature')
    if not callable(orig_extract_feature):
        return False

    setattr(core, '_eviguard_orig_extract_feature', orig_extract_feature)

    def _patched_extract_feature(self, pixel_values, *args, **kwargs):
        fn = _direct_getattr(self, '_eviguard_orig_extract_feature')
        tgt = _direct_getattr(self, '_eviguard_extract_target_dtype', target_dtype)
        device_type = None
        if torch.is_tensor(pixel_values):
            device_type = pixel_values.device.type
        amp_enabled = bool(device_type in ('cuda', 'cpu') and tgt in (torch.float16, torch.bfloat16))
        if amp_enabled:
            with torch.autocast(device_type=device_type, dtype=tgt):
                out = fn(pixel_values, *args, **kwargs)
        else:
            out = fn(pixel_values, *args, **kwargs)
        if torch.is_tensor(out) and out.is_floating_point() and out.dtype != tgt:
            out = out.to(dtype=tgt)
        return out

    setattr(core, 'extract_feature', types.MethodType(_patched_extract_feature, core))
    setattr(core, '_eviguard_extract_feature_patched', True)
    return True


def ensure_internvl_connector_output_dtype(model, target_dtype=None, input_ids=None):
    core = unwrap_internvl_model(model)
    if target_dtype is None:
        target_dtype = _infer_internvl_lm_dtype(core, input_ids=input_ids)

    wrapped = False
    for attr in ('mlp1', 'vision_proj'):
        mod = _direct_getattr(core, attr)
        if mod is None:
            continue
        if getattr(mod, '_eviguard_output_cast_wrapper', False):
            mod.target_dtype = target_dtype
            wrapped = True
            continue
        setattr(core, attr, _FloatingOutputCastWrapper(mod, target_dtype=target_dtype))
        wrapped = True

    patched = _patch_internvl_extract_feature_dtype(core, target_dtype)
    if wrapped or patched:
        logger.debug('InternVL visual feature dtype fixed to %s', str(target_dtype).replace('torch.', ''))
    return wrapped or patched


INTERNVL_SYSTEM_MSG = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN   = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def get_num_image_tokens(model):
    if hasattr(model, 'num_image_token'):
        return model.num_image_token
    return 256


def build_internvl_prompt(question: str, num_patches: int, num_image_token: int,
                          has_image: bool = True, system_msg: str = None):
    sys_msg = system_msg or INTERNVL_SYSTEM_MSG

    if has_image:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
        user_content = f"{image_tokens}\n{question}"
    else:
        user_content = question

    prompt = (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def build_internvl_full_text(question: str, answer: str, num_patches: int,
                              num_image_token: int, has_image: bool = True,
                              system_msg: str = None):
    prompt = build_internvl_prompt(question, num_patches, num_image_token,
                                    has_image, system_msg)
    return prompt, prompt + answer + "<|im_end|>"



_INTERNVL_MIXED_WARNED = False


def prepare_model_inputs_internvl(batch, model, tokenizer, max_length=None,
                                   device=None):
    if max_length is None:
        max_length = int(get_common_value("runtime.internvl.prepare_max_length", 2048, _COMMON_CFG))
    if device is None:
        device = torch.device(get_common_value("runtime.internvl.default_device", "cuda", _COMMON_CFG))
    global _INTERNVL_MIXED_WARNED

    images = batch["images"]
    questions = batch["questions"]
    answers = batch["answers"]

    base_model = model
    for attr in ["base_model", "model"]:
        if hasattr(base_model, attr):
            candidate = getattr(base_model, attr)
            if hasattr(candidate, "num_image_token"):
                base_model = candidate
                break
    num_image_token = getattr(base_model, "num_image_token", 256)
    max_dynamic_patches = int(getattr(base_model, "eviguard_max_dynamic_patches", get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG)) or get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG))

    img_context_token_id = _resolve_img_context_token_id(tokenizer)
    _set_internvl_img_context_token_id_by_id(model, img_context_token_id)

    all_pixel_values = []
    num_patches_list = []
    valid_image_flags = []

    for img in images:
        if img is not None:
            try:
                pv, np_ = load_image_internvl(img, max_num=max_dynamic_patches)
                all_pixel_values.append(pv)
                num_patches_list.append(np_)
                valid_image_flags.extend([1] * np_)
            except Exception as e:
                raise ValueError("InternVL visual preprocessing failed; refusing text-only substitution") from e
        else:
            num_patches_list.append(0)

    prompts = []
    full_texts = []
    for i, (q, a, img) in enumerate(zip(questions, answers, images)):
        has_image = img is not None and num_patches_list[i] > 0
        np_ = num_patches_list[i] if has_image else 0
        prompt, full_text = build_internvl_full_text(
            q, a, np_, num_image_token, has_image=has_image)
        prompts.append(prompt)
        full_texts.append(full_text)

    tokenizer.padding_side = 'right'
    tok_out = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = tok_out["input_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)

    if all_pixel_values:
        pixel_values = torch.cat(all_pixel_values, dim=0).to(device=device, dtype=torch.float16)
        image_flags = torch.tensor(valid_image_flags, dtype=torch.long, device=device)
    else:
        pixel_values = None
        image_flags = None

    labels = input_ids.clone()
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        pl = min(prompt_ids.shape[1], labels.shape[1])
        labels[i, :pl] = -100
    labels[attention_mask == 0] = -100
    if ((labels[:,1:] != -100).sum(1) == 0).any():
        raise ValueError("InternVL answer targets empty or truncated; increase max_seq_length")

    keep_indices = list(range(len(images)))
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)

    mi = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_flags": image_flags,
        "num_patches_list": num_patches_list,
        "img_context_token_id": img_context_token_id,
        "forget_mask": batch["forget_mask"][keep_tensor].to(device),
        "retain_mask": batch["retain_mask"][keep_tensor].to(device),
        "pd_mask": batch["pd_mask"][keep_tensor].to(device),
        "reliabilities": batch["reliabilities"][keep_tensor].to(device),
        "evidence_strings": [batch["evidence_strings"][i] for i in keep_indices],
        "original_answers": [batch["original_answers"][i] for i in keep_indices],
        "splits": [batch["splits"][i] for i in keep_indices],
        "kept_indices": keep_indices,
    }
    return mi


def _collect_linear_leaf_names(model):
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            names.add(n.split(".")[-1])
    return sorted(names)


def build_internvl_lora_config(model, lcfg, bias="none"):
    from peft import LoraConfig

    available = _collect_linear_leaf_names(model)
    available_set = set(available)

    legacy_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    internlm2_targets = ["wqkv", "wo", "w1", "w2", "w3"]

    raw_targets = list(lcfg.get("target_modules", legacy_targets))
    raw_targets = [t for t in raw_targets if t not in ("mlp.fc1", "mlp.fc2")]

    targets = [t for t in raw_targets if t in available_set]

    if not targets:
        if any(t in available_set for t in internlm2_targets):
            targets = [t for t in internlm2_targets if t in available_set]
        elif any(t in available_set for t in legacy_targets):
            targets = [t for t in legacy_targets if t in available_set]

    if not targets:
        raise ValueError(
            "InternVL LoRA target_modules 解析失败。当前模型可见的 Linear 叶子模块名包括: "
            f"{available[:40]}"
        )

    from src.models.llava_utils import language_lora_targets
    targets = language_lora_targets(model, targets)
    logger.info("InternVL LoRA targets resolved to: %s", targets)
    return LoraConfig(
        r=lcfg["rank"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=targets,
        bias=bias,
    )


def unwrap_internvl_model(model):
    seen = set()

    def _walk(obj):
        if obj is None or id(obj) in seen:
            return None
        seen.add(id(obj))

        cls_name = getattr(type(obj), "__name__", "")
        if cls_name == "InternVLChatModel":
            return obj
        if _direct_hasattr(obj, "language_model") and _direct_hasattr(obj, "vision_model") and _direct_hasattr(obj, "num_image_token"):
            return obj

        gbm = _direct_getattr(obj, "get_base_model")
        if callable(gbm):
            try:
                found = _walk(gbm())
                if found is not None:
                    return found
            except Exception:
                pass

        for attr in ("base_model", "model"):
            child = _direct_getattr(obj, attr)
            if child is not None:
                found = _walk(child)
                if found is not None:
                    return found
        return None

    return _walk(model) or model


def _set_internvl_img_context_token_id_by_id(model, img_context_token_id):
    seen = set()

    def _walk(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))

        cls_name = getattr(type(obj), "__name__", "")
        if cls_name == "InternVLChatModel" or (_direct_hasattr(obj, "language_model") and _direct_hasattr(obj, "vision_model")):
            try:
                setattr(obj, "img_context_token_id", img_context_token_id)
            except Exception:
                pass

        gbm = _direct_getattr(obj, "get_base_model")
        if callable(gbm):
            try:
                _walk(gbm())
            except Exception:
                pass

        for attr in ("base_model", "model"):
            child = _direct_getattr(obj, attr)
            if child is not None:
                _walk(child)

    _walk(model)
    return img_context_token_id


def find_internvl_connector_module(model):
    for path in (
        "base_model.model.mlp1",
        "model.mlp1",
        "mlp1",
        "base_model.model.vision_proj",
        "vision_proj",
    ):
        mod = model
        try:
            for p in path.split("."):
                mod = getattr(mod, p)
            return mod
        except AttributeError:
            pass
    for name, module in model.named_modules():
        lname = name.lower()
        if lname.endswith("mlp1") or "vision_proj" in lname:
            return module
    return None


def get_internvl_embedding_layer(model):
    for path in (
        "base_model.model.language_model.model.embed_tokens",
        "model.language_model.model.embed_tokens",
        "base_model.model.language_model.model.tok_embeddings",
        "model.language_model.model.tok_embeddings",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
    ):
        mod = model
        try:
            for p in path.split("."):
                mod = getattr(mod, p)
            if isinstance(mod, nn.Embedding):
                return mod
        except AttributeError:
            pass

    for obj in (model, _direct_getattr(model, "base_model"), _direct_getattr(model, "model")):
        if obj is None:
            continue
        lm = _direct_getattr(obj, "language_model")
        if lm is not None and hasattr(lm, "get_input_embeddings"):
            try:
                emb = lm.get_input_embeddings()
            except Exception:
                emb = None
            if isinstance(emb, nn.Embedding):
                return emb

    for n, m in model.named_modules():
        if isinstance(m, nn.Embedding) and ("embed_tokens" in n or "tok_embeddings" in n):
            return m
    return None


def _set_internvl_img_context_token_id(model, tokenizer):
    return _set_internvl_img_context_token_id_by_id(model, _resolve_img_context_token_id(tokenizer))


def internvl_forward(model, input_ids, attention_mask, pixel_values=None,
                     image_flags=None, labels=None, num_patches_list=None,
                     img_context_token_id=None, **kwargs):
    core_model = unwrap_internvl_model(model)

    if img_context_token_id is None:
        img_context_token_id = _direct_getattr(core_model, "img_context_token_id")
    if img_context_token_id is None:
        img_context_token_id = _direct_getattr(model, "img_context_token_id")
    if img_context_token_id is None:
        raise ValueError("InternVL 的 img_context_token_id 尚未设置，请先在 prepare_model_inputs_internvl 中注入该值")
    _set_internvl_img_context_token_id_by_id(core_model, img_context_token_id)
    ensure_internvl_connector_output_dtype(core_model, input_ids=input_ids)

    if pixel_values is None:
        language_model = _direct_getattr(core_model, "language_model")
        if language_model is None:
            raise ValueError("InternVL text-only forward failed: language_model not found")
        lm_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "use_cache": False,
        }
        if labels is not None:
            lm_kwargs["labels"] = labels
        return language_model(**lm_kwargs)

    forward_kwargs = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if image_flags is not None:
        if image_flags.dim() == 1:
            forward_kwargs["image_flags"] = image_flags.unsqueeze(-1)
        elif image_flags.dim() == 2:
            forward_kwargs["image_flags"] = image_flags
        else:
            forward_kwargs["image_flags"] = image_flags.view(-1, 1)
    else:
        forward_kwargs["image_flags"] = torch.ones(
            pixel_values.shape[0], 1, dtype=torch.long, device=pixel_values.device
        )

    return core_model(**forward_kwargs)



def internvl_generate_response(model, tokenizer, image, question,
                                max_new_tokens=None, do_sample=None,
                                temperature=None, top_p=None,
                                device=None):
    if max_new_tokens is None:
        max_new_tokens = int(get_common_value("runtime.internvl.generate_max_new_tokens", 512, _COMMON_CFG))
    if do_sample is None:
        do_sample = bool(get_common_value("runtime.internvl.generate_do_sample", False, _COMMON_CFG))
    if temperature is None:
        temperature = float(get_common_value("runtime.internvl.generate_temperature", 0.0, _COMMON_CFG))
    if top_p is None:
        top_p = float(get_common_value("runtime.generation.top_p", 0.9, _COMMON_CFG))
    if device is None:
        device = torch.device(get_common_value("runtime.internvl.default_device", "cuda", _COMMON_CFG))
    gen_config = {"max_new_tokens": max_new_tokens}
    if do_sample and temperature > 0:
        gen_config["do_sample"] = True
        gen_config["temperature"] = temperature
        gen_config["top_p"] = top_p
    else:
        gen_config["do_sample"] = False

    base_model = unwrap_internvl_model(model)
    core_model = base_model

    if hasattr(core_model, "chat"):
        try:
            if image is not None:
                max_dynamic_patches = int(getattr(base_model, "eviguard_max_dynamic_patches", get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG)) or get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG))
                pv, _ = load_image_internvl(image, max_num=max_dynamic_patches)
                pv = pv.to(device=device, dtype=torch.float16)
            else:
                pv = None
            _set_internvl_img_context_token_id(model, tokenizer)
            response = core_model.chat(tokenizer, pv, question, gen_config)
            return response
        except Exception as e:
            logger.warning("InternVL chat() failed, falling back to manual generate: %s", e)

    num_image_token = getattr(base_model, "num_image_token", 256)
    max_dynamic_patches = int(getattr(base_model, "eviguard_max_dynamic_patches", get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG)) or get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG))

    if image is not None:
        max_dynamic_patches = int(getattr(base_model, "eviguard_max_dynamic_patches", get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG)) or get_common_value("runtime.defaults.max_dynamic_patches", cfg=_COMMON_CFG))
        pv, num_patches = load_image_internvl(image, max_num=max_dynamic_patches)
        pv = pv.to(device=device, dtype=torch.float16)
    else:
        pv = None
        num_patches = 0

    has_image = pv is not None
    prompt = build_internvl_prompt(question, num_patches, num_image_token, has_image)

    _set_internvl_img_context_token_id(model, tokenizer)

    tok_out = tokenizer(prompt, return_tensors="pt")
    input_ids = tok_out["input_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
    }
    if do_sample and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    if pv is not None:
        gen_kwargs["pixel_values"] = pv

    if not hasattr(core_model, "generate"):
        raise AttributeError(
            "InternVL generate fallback unavailable: core_model has no .generate(); "
            "please use chat() path or load with a transformers version exposing GenerationMixin."
        )
    output_ids = core_model.generate(**gen_kwargs)
    response = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    return response



def filter_pixels_internvl(inp, mask):
    pv = inp.get("pixel_values")
    if pv is None:
        return None, None

    npl = inp.get("num_patches_list")
    if npl is None:
        return pv, inp.get("image_flags")

    selected_pv = []
    selected_flags = []
    offset = 0
    image_flags = inp.get("image_flags")

    for i, np_ in enumerate(npl):
        if mask[i]:
            if np_ > 0:
                selected_pv.append(pv[offset:offset + np_])
                if image_flags is not None:
                    selected_flags.append(image_flags[offset:offset + np_])
        if np_ > 0:
            offset += np_

    if not selected_pv:
        return None, None

    new_pv = torch.cat(selected_pv, dim=0)
    new_flags = torch.cat(selected_flags, dim=0) if selected_flags else None

    new_npl = [npl[i] for i in range(len(mask)) if mask[i]]

    return new_pv, new_flags, new_npl



def internvl_generate(*args, **kwargs):
    return internvl_generate_response(*args, **kwargs)
