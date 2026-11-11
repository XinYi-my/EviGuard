import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from src.utils.helpers import load_common_config, get_common_value

logger = logging.getLogger("eviguard.evidence")
_COMMON_CFG = load_common_config()


def normalize_evidence_key(image_path: str) -> str:
    key = str(image_path or "").replace("\\", "/").strip()
    while key.startswith("./"):
        key = key[2:]
    return key


def evidence_key_candidates(image_path: str) -> List[str]:
    key = normalize_evidence_key(image_path)
    if not key:
        return [""]
    cands = [key]
    markers = _COMMON_CFG["evidence"]["normalize_markers"]
    for m in markers:
        pos = key.find(m)
        if pos >= 0:
            cands.append(key[pos:])
    # Basename-only matches can alias unrelated images; require a namespaced key.
    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


class OCRExtractor:
    def __init__(self, languages=None, gpu=None):
        if languages is None:
            languages = _COMMON_CFG["evidence"]["ocr_languages"]
        if gpu is None:
            gpu = _COMMON_CFG["evidence"]["ocr_use_gpu"]
        self.languages = list(languages)
        self.gpu = bool(gpu) and torch.cuda.is_available()
        self._reader = None

    def _lazy_init(self):
        if self._reader is None:
            import easyocr
            logger.info(f"初始化 EasyOCR (langs={self.languages}, gpu={self.gpu})")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def extract(self, image_path: str) -> Dict:
        self._lazy_init()
        try:
            results = self._reader.readtext(image_path)
            texts = [r[1] for r in results]
            confs = [float(r[2]) for r in results]
            return {
                "texts": texts,
                "confidences": confs,
                "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            }
        except Exception as e:
            logger.warning(f"OCR失败 {image_path}: {e}")
            return {"texts": [], "confidences": [], "mean_confidence": 0.0, "error": str(e)}


class TagExtractor:
    def __init__(self, model_path: str, max_tags=None):
        if max_tags is None:
            max_tags = _COMMON_CFG["evidence"]["tag_max_tags"]
        self.model_path = model_path
        self.max_tags = int(max_tags)
        self._model = None
        self._processor = None
        self._disabled = False
        self._first_error_logged = False

    def _empty(self) -> Dict:
        return {"caption": "", "tags": [], "tag_confidence": 0.0}

    def _lazy_init(self):
        if self._disabled or self._model is not None:
            return

        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        logger.info(f"加载 BLIP-2: {self.model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._processor = Blip2Processor.from_pretrained(
            self.model_path,
            use_fast=False,
            local_files_only=True,
        )
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )

        num_query_tokens = (
            getattr(self._processor, "num_query_tokens", None)
            or getattr(self._model.config, "num_query_tokens", None)
            or getattr(getattr(self._model.config, "qformer_config", None), "query_length", None)
            or _COMMON_CFG["evidence"]["blip2_num_query_tokens"]
        )
        num_query_tokens = int(num_query_tokens)
        self._processor.num_query_tokens = num_query_tokens
        self._model.config.num_query_tokens = num_query_tokens

        tokenizer = self._processor.tokenizer
        vocab = tokenizer.get_vocab()
        if "<image>" not in vocab:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        if image_token_id is None:
            raise RuntimeError("无法为 BLIP-2 tokenizer 获取 <image> token id")

        try:
            self._model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        except TypeError:
            self._model.resize_token_embeddings(len(tokenizer))

        self._model.config.image_token_index = int(image_token_id)

        bos = getattr(self._model.config.text_config, "bos_token_id", None)
        if bos is None:
            bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.cls_token_id
        if bos is None:
            bos = tokenizer.eos_token_id
        if bos is None:
            raise RuntimeError("BLIP-2 bos_token_id 仍然为 None，无法生成 caption")

        self._model.config.text_config.bos_token_id = int(bos)

        gen_cfg = getattr(self._model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.image_token_index = int(image_token_id)
            gen_cfg.bos_token_id = int(bos)

            if getattr(gen_cfg, "eos_token_id", None) is None:
                eos = getattr(self._model.config.text_config, "eos_token_id", None)
                if eos is None:
                    eos = tokenizer.eos_token_id
                gen_cfg.eos_token_id = eos

            if getattr(gen_cfg, "pad_token_id", None) is None:
                pad = getattr(self._model.config.text_config, "pad_token_id", None)
                if pad is None:
                    pad = tokenizer.pad_token_id
                if pad is None:
                    pad = gen_cfg.eos_token_id
                gen_cfg.pad_token_id = pad

        self._model = self._model.to(device)
        self._model.eval()

    def _move_inputs(self, inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        moved = {}
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                moved[k] = v
                continue
            if k == "pixel_values":
                target_dtype = torch.float16 if device.type == "cuda" else torch.float32
                moved[k] = v.to(device=device, dtype=target_dtype)
            else:
                moved[k] = v.to(device=device)
        return moved

    def _parse_tag_text(self, text: str) -> List[str]:
        raw = (text or "").strip().lower()
        if not raw:
            return []
        norm = raw.replace("\n", ",").replace(";", ",").replace("|", ",")
        parts = []
        for chunk in norm.split(","):
            chunk = " ".join(chunk.strip().split())
            if not chunk:
                continue
            chunk = re.sub(r"^(objects?|tags?|scene|main objects?)\s*[:\-]\s*", "", chunk)
            parts.append(chunk)
        tags = []
        seen = set()
        for part in parts:
            cand = part.strip(" .")
            if not cand or cand in seen:
                continue
            seen.add(cand)
            tags.append(cand)
            if len(tags) >= self.max_tags:
                break
        return tags

    def _caption_to_tags(self, caption: str) -> List[str]:
        text = (caption or "").lower()
        text = text.replace("\n", " ").replace("\t", " ").replace("\\", " ")

        for ch in [".", ";", ":", "!", "?", "，", "。", "；", "：", "|", "/", "(", ")", "[", "]", "{", "}"]:
            text = text.replace(ch, ",")

        stopwords = {
            "a", "an", "the", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being",
            "with", "and", "or", "of", "in", "on", "at", "to", "for", "from",
            "by", "as", "it", "its", "into", "over", "under", "near",
            "image", "picture", "photo", "showing", "shows", "background", "foreground"
        }

        chunks = []
        for part in text.split(","):
            part = " ".join(part.strip().split())
            if not part or len(part) <= 1:
                continue
            if part in stopwords:
                continue
            chunks.append(part)

        tags = []
        seen = set()

        for part in chunks:
            cands = [part]
            words = part.split()
            if len(words) >= 2:
                cands.append(" ".join(words[-2:]))
                cands.append(words[-1])
            elif len(words) == 1:
                cands.append(words[0])

            for cand in cands:
                cand = cand.strip()
                if not cand or cand in stopwords or cand.isdigit():
                    continue
                if cand not in seen:
                    seen.add(cand)
                    tags.append(cand)
                if len(tags) >= self.max_tags:
                    return tags

        return tags[:self.max_tags]

    def extract(self, image_path: str) -> Dict:
        if self._disabled:
            return self._empty()

        self._lazy_init()

        try:
            image = Image.open(image_path).convert("RGB")
            device = next(self._model.parameters()).device

            tag_prompt = get_common_value("evidence.tag_prompt", "Question: List the main objects and scene tags in this image as a short comma-separated list. Answer:", _COMMON_CFG)
            inputs = self._processor(images=image, text=tag_prompt, return_tensors="pt")
            inputs = self._move_inputs(inputs, device)

            with torch.no_grad():
                ids = self._model.generate(
                    **inputs,
                    max_new_tokens=get_common_value("evidence.tag_max_new_tokens", 40, _COMMON_CFG),
                    do_sample=False,
                    use_cache=True,
                    num_beams=get_common_value("evidence.generation_num_beams", 1, _COMMON_CFG),
                )

            tag_text = self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            tags = self._parse_tag_text(tag_text)
            if not tags:
                cap_inputs = self._processor(images=image, return_tensors="pt")
                cap_inputs = self._move_inputs(cap_inputs, device)
                with torch.no_grad():
                    cap_ids = self._model.generate(
                        **cap_inputs,
                        max_new_tokens=get_common_value("evidence.caption_max_new_tokens", 50, _COMMON_CFG),
                        do_sample=False,
                        use_cache=True,
                        num_beams=get_common_value("evidence.generation_num_beams", 1, _COMMON_CFG),
                    )
                caption = self._processor.batch_decode(cap_ids, skip_special_tokens=True)[0].strip()
                tags = self._caption_to_tags(caption)
            else:
                caption = tag_text

            conf = get_common_value("evidence.tag_confidence_base", 0.2, _COMMON_CFG) + get_common_value("evidence.tag_confidence_scale", 0.6, _COMMON_CFG) * (1.0 - np.exp(-len(tags) / get_common_value("evidence.tag_confidence_divisor", 6.0, _COMMON_CFG)))
            return {"caption": caption, "tags": tags, "tag_confidence": conf}

        except Exception as exc:
            if not self._first_error_logged:
                logger.exception(f"标签提取失败，将记录本图提取错误: {image_path}")
                self._first_error_logged = True
            return {**self._empty(), "error": str(exc)}


class EvidenceCache:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = _COMMON_CFG["evidence"]["default_cache_dir"]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / get_common_value("evidence.cache_filename", "evidence_cache.json", _COMMON_CFG)
        self._cache = self._load()

    def _load(self):
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                data = json.load(f)
            logger.info(f"加载证据缓存: {len(data)} 条")
            return data
        return {}

    def save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2, default=str)

    def get(self, image_path: str):
        for key in evidence_key_candidates(image_path):
            if key in self._cache:
                return self._cache[key]
        return None

    def put(self, image_path: str, ev, cache_key: Optional[str] = None):
        key = normalize_evidence_key(cache_key if cache_key is not None else image_path)
        self._cache[key] = ev

    def __len__(self):
        return len(self._cache)


class EvidencePipeline:
    def __init__(
        self,
        cache_dir,
        use_ocr,
        use_tags,
        blip2_model,
        default_reliability=0.5,
    ):
        self.default_reliability = default_reliability
        self.cache = EvidenceCache(cache_dir)
        self.ocr = OCRExtractor() if use_ocr else None
        self.tagger = TagExtractor(model_path=blip2_model) if use_tags else None

    def _calibrate_reliability(self, c_ocr: float, c_tag: float) -> float:
        rel_cfg = get_common_value("evidence.reliability", {}, _COMMON_CFG)
        dual_cfg = rel_cfg.get("dual_source", {})
        single_cfg = rel_cfg.get("single_source", {})
        clamp_min = float(rel_cfg.get("clamp_min", 0.05))
        clamp_max = float(rel_cfg.get("clamp_max", 0.95))
        ocr_weight = float(dual_cfg.get("ocr_weight", 0.6))
        tag_weight = float(dual_cfg.get("tag_weight", 0.4))
        disagreement_penalty = float(dual_cfg.get("disagreement_penalty", 0.15))
        source_weight = float(single_cfg.get("source_weight", 0.85))
        prior_weight = float(single_cfg.get("prior_weight", 0.15))

        c_ocr = float(max(0.0, min(1.0, c_ocr)))
        c_tag = float(max(0.0, min(1.0, c_tag)))
        if c_ocr <= 0.0 and c_tag <= 0.0:
            return float(self.default_reliability)
        if c_ocr > 0.0 and c_tag > 0.0:
            rel = ocr_weight * c_ocr + tag_weight * c_tag - disagreement_penalty * abs(c_ocr - c_tag)
        else:
            only = c_ocr if c_ocr > 0.0 else c_tag
            rel = source_weight * only + prior_weight * float(self.default_reliability)
        return float(max(clamp_min, min(clamp_max, rel)))

    def extract_single(self, image_path: str, force=False, cache_key: Optional[str] = None) -> Dict:
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Evidence image not found: {image_path}")
        if not force:
            cached = self.cache.get(cache_key if cache_key is not None else image_path)
            if cached:
                return cached

        result = {"image_path": image_path}

        if self.ocr:
            result["ocr"] = self.ocr.extract(image_path)
        else:
            result["ocr"] = {"texts": [], "confidences": [], "mean_confidence": 0.0}

        if self.tagger:
            result["tags"] = self.tagger.extract(image_path)
        else:
            result["tags"] = {"caption": "", "tags": [], "tag_confidence": 0.0}

        ocr_text = " ".join(result["ocr"]["texts"])
        tag_text = ", ".join(result["tags"].get("tags", []))
        result["extraction_errors"] = {k: result[k]["error"] for k in ("ocr", "tags") if "error" in result[k]}
        result["evidence_string"] = f"[OCR] {ocr_text} [TAGS] {tag_text}"

        c_ocr = result["ocr"]["mean_confidence"]
        c_tag = result["tags"].get("tag_confidence", 0.0)
        result["reliability"] = self._calibrate_reliability(c_ocr, c_tag)

        self.cache.put(image_path, result, cache_key=cache_key)
        return result

    def precompute_all(self, image_paths: List[str]):
        logger.info(f"预计算证据: {len(image_paths)} 张图片...")
        for i, path in enumerate(image_paths):
            self.extract_single(path)
            if (i + 1) % 100 == 0:
                logger.info(f"  已处理: {i+1}/{len(image_paths)}")
                self.cache.save()
        self.cache.save()
        logger.info("证据预计算完成。")

    def get_evidence_string(self, image_path: str) -> Tuple[str, float]:
        ev = self.extract_single(image_path)
        return ev["evidence_string"], ev["reliability"]
