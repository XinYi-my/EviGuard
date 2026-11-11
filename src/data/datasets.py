import hashlib, json, os, logging, random
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from src.utils.helpers import load_common_config, get_common_value

logger = logging.getLogger("eviguard.data")
_COMMON_CFG = load_common_config()

IDK_POOL = get_common_value("autogen.refusal_templates", [], _COMMON_CFG)

def normalize_evidence_key(image_ref: str) -> str:
    if not image_ref: return ""
    key = str(image_ref).replace("\\", "/").strip()
    while key.startswith("./"): key = key[2:]
    return key


def resolve_image_path(image_ref: str, search_dirs: List[str]) -> Optional[str]:
    if not image_ref: return None
    if os.path.isabs(image_ref) and os.path.exists(image_ref): return image_ref
    for d in search_dirs:
        full = os.path.join(d, image_ref)
        if os.path.exists(full): return full
    basename = os.path.basename(image_ref)
    matches = set()
    for d in search_dirs:
        for root, dirs, files in os.walk(d):
            if basename in files: matches.add(str(Path(root, basename).resolve()))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image basename: {basename}")
    return next(iter(matches), None)


def require_records(raw, source="dataset"):
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source} must be a non-empty list of records")
    if any(not isinstance(row, dict) for row in raw):
        raise ValueError(f"{source} contains a non-object record")
    return raw


def load_image(image_path):
    """Read a required visual input; missing/corrupt images never become text samples."""
    if not image_path or not Path(image_path).is_file():
        raise FileNotFoundError(f"Required image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode image: {image_path}") from exc


def evaluation_collate_fn(batch):
    """Keep PIL images and optional fields intact for evaluation loaders."""
    return batch


def load_evidence_cache(cache_dir):
    if not cache_dir:
        return {}
    cache = {}
    for name in ("evidence_cache.json", "vlguard_evidence_cache.json"):
        path = Path(cache_dir) / name
        if path.is_file():
            with path.open(encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise ValueError("Invalid evidence cache")
            cache.update(payload)
    return cache


def cached_evidence(cache, image_path, image_ref="", required=False):
    keys = [normalize_evidence_key(str(Path(image_path).resolve()))] if image_path else []
    keys += [str(Path(image_path).resolve())] if image_path else []
    if image_ref:
        keys.append(normalize_evidence_key(image_ref))
    evidence = next((cache[key] for key in keys if key in cache), None)
    if evidence is None:
        if required and image_path:
            raise ValueError(f"Required cached evidence missing for image: {image_path}")
        return {"evidence_string": "", "reliability": float(get_common_value("data.default_reliability", 0.5, _COMMON_CFG))}
    import math
    if not isinstance(evidence, dict) or evidence.get("extraction_errors"):
        raise ValueError(f"Invalid or failed cached evidence for image: {image_path}")
    if not isinstance(evidence.get("evidence_string"), str):
        raise ValueError("Cached evidence is missing evidence_string")
    reliability = evidence.get("reliability")
    if not isinstance(reliability, (int, float)) or not math.isfinite(reliability) or not 0 <= reliability <= 1:
        raise ValueError("Cached evidence reliability must lie between zero and one")
    return evidence


def flatten_safeeraser(raw_data: list, search_dirs: list, seed: int) -> List[Dict]:
    require_records(raw_data, "SafeEraser")
    samples = []
    sid = 0

    # Maintained-release protocol: hold out whole source images, rather than
    # different questions for the same image. Hash ordering is seed-dependent
    # and independent of JSON record order. This is not a recovered paper split.
    group_keys = {}
    for rec in raw_data:
        image_ref = rec.get("image_path", rec.get("image_id", ""))
        path = resolve_image_path(image_ref, search_dirs)
        if not path:
            raise FileNotFoundError(f"Required SafeEraser image not found: {image_ref}")
        canonical = Path(path).resolve()
        relative_names = []
        for directory in search_dirs:
            try:
                relative_names.append(canonical.relative_to(Path(directory).resolve()).as_posix())
            except ValueError:
                continue
        group_keys[id(rec)] = (min(relative_names, key=len) if relative_names else canonical.name).casefold()
    groups = sorted(set(group_keys.values()), key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
    ratio = float(get_common_value("data.safeeraser_split.forget_test_ratio", 0.2, _COMMON_CFG))
    if not 0 < ratio < 1:
        raise ValueError("SafeEraser forget_test_ratio must lie between zero and one")
    n_test = min(len(groups) - 1, max(1, int(len(groups) * ratio))) if len(groups) > 1 else 0
    heldout_groups = set(groups[:n_test])

    for rec in raw_data:
        cat = rec.get("category", "unknown")
        kw = rec.get("keyword", "")
        image_id = rec.get("image_path", rec.get("image_id", ""))
        sd_image = rec.get("SDImage_path", "")
        forget_img = resolve_image_path(image_id, search_dirs)
        pd_img = resolve_image_path(sd_image, search_dirs)
        if sd_image and not pd_img:
            raise FileNotFoundError(f"Required PD image not found: {sd_image}")
        heldout = group_keys[id(rec)] in heldout_groups

        unsafe_pairs = rec.get("unsafe_pairs") or []

        for pair in unsafe_pairs:
            if not isinstance(pair, dict): continue
            q = pair.get("question", "")
            orig_answer = pair.get("answer",
                pair.get("Answer",
                    pair.get("response",
                        pair.get("model_response", ""))))
            if not q: continue

            digest = hashlib.sha256(f"{seed}|{image_id}|{q}".encode()).hexdigest()
            idk_answer = IDK_POOL[int(digest, 16) % len(IDK_POOL)]
            split_label = "forget_test" if heldout else "forget"

            samples.append({
                "id": f"forget_{sid}",
                "image_ref": image_id,
                "image_path": forget_img,
                "question": q,
                "answer": idk_answer,
                "original_answer": orig_answer,
                "split": split_label,
                "modality": "image",
                "category": cat,
                "keyword": kw,
            })
            sid += 1

        if pd_img and not heldout:
            for pair in unsafe_pairs:
                if not isinstance(pair, dict): continue
                q = pair.get("question", "")
                pd_answer = pair.get("sd_response",
                    pair.get("SDResponse",
                        pair.get("safe_response", "")))
                if not q or not pd_answer: continue

                samples.append({
                    "id": f"pd_{sid}",
                    "image_ref": sd_image,
                    "image_path": pd_img,
                    "question": q,
                    "answer": pd_answer,
                    "split": "pd",
                    "modality": "image",
                    "category": cat,
                    "keyword": kw,
                })
                sid += 1

        for field in ["UnharmPair_image1", "UnharmPair_image2"]:
            p = rec.get(field, {})
            if not isinstance(p, dict): continue
            q = p.get("Question", p.get("question", ""))
            a = p.get("Prediction", p.get("Answer", p.get("answer", "")))
            if not q: continue
            samples.append({
                "id": f"retain_img_{sid}",
                "image_ref": image_id,
                "image_path": forget_img,
                "question": q,
                "answer": a,
                "original_answer": "",
                "split": "retain_test" if heldout else "retain",
                "modality": "image",
                "category": cat,
                "keyword": kw,
            })
            sid += 1

        for field in ["UnharmPair_text1", "UnharmPair_text2"]:
            p = rec.get(field, {})
            if not isinstance(p, dict): continue
            q = p.get("Question", p.get("question", ""))
            a = p.get("Prediction", p.get("Answer", p.get("answer", "")))
            if not q: continue
            samples.append({
                "id": f"retain_txt_{sid}",
                "image_ref": "",
                "image_path": None,
                "question": q,
                "answer": a,
                "original_answer": "",
                "split": "retain_test" if heldout else "retain",
                "modality": "text",
                "category": cat,
                "keyword": kw,
            })
            sid += 1

    if not samples:
        raise ValueError("SafeEraser contains no usable samples")
    seen = set()
    for sample in samples:
        identity = (normalize_evidence_key(sample["image_ref"]), sample["question"], sample["split"])
        if identity in seen:
            raise ValueError("SafeEraser contains a duplicate image/question sample")
        seen.add(identity)
        sample["id"] = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
    sc = {}
    for s in samples:
        sc[s["split"]] = sc.get(s["split"], 0) + 1
    logger.info(f"展开: {len(raw_data)} 原始 → {len(samples)} 样本, 分布: {sc}")
    return samples


class SafeEraserDataset(Dataset):
    def __init__(self, json_path, image_search_dirs, split_filter="all",
                 evidence_cache_dir=None, seed=None, require_evidence=False):
        if seed is None:
            seed = _COMMON_CFG["runtime"]["default_seed"]
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        all_s = flatten_safeeraser(raw, image_search_dirs, seed)
        self.data = [s for s in all_s if s["split"] == split_filter] if split_filter != "all" else all_s
        self._split_indices = {}
        self._split_modality_indices = {}
        for i, s in enumerate(self.data):
            self._split_indices.setdefault(s["split"], []).append(i)
            md = s.get("modality", "image" if s.get("image_path") else "text")
            self._split_modality_indices.setdefault((s["split"], md), []).append(i)
        self._evidence = load_evidence_cache(evidence_cache_dir)
        self.require_evidence = require_evidence
        logger.info(f"Dataset: {len(self.data)}, filter='{split_filter}', "
                     f"splits={dict((k,len(v)) for k,v in self._split_indices.items())}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        image = load_image(s["image_path"]) if s.get("modality") == "image" else None
        ev = cached_evidence(self._evidence, s.get("image_path"), s.get("image_ref", ""), self.require_evidence)
        default_reliability = get_common_value("data.default_reliability", cfg=_COMMON_CFG)
        return {
            "id": s["id"],
            "image": image,
            "image_path": s["image_path"] or "",
            "question": s["question"],
            "answer": s["answer"],
            "original_answer": s.get("original_answer", ""),
            "split": s["split"],
            "modality": s.get("modality", get_common_value("data.default_modality", "image", _COMMON_CFG) if s.get("image_path") else get_common_value("data.default_text_modality", "text", _COMMON_CFG)),
            "category": s["category"],
            "evidence_string": ev.get("evidence_string", "[OCR]  [TAGS] "),
            "reliability": ev.get("reliability", default_reliability),
        }

    def get_split_indices(self, name, modality=None):
        if modality is None:
            return self._split_indices.get(name, [])
        return self._split_modality_indices.get((name, modality), [])


class SIUODataset(Dataset):
    def __init__(self, json_path, image_dir, evidence_cache_dir=None, require_evidence=False):
        self.image_dir = image_dir
        self._evidence = load_evidence_cache(evidence_cache_dir)
        self.require_evidence = require_evidence
        with open(json_path, "r", encoding="utf-8") as f: raw = json.load(f)
        if isinstance(raw, dict):
            for k in ["data","samples","items"]:
                if k in raw and isinstance(raw[k], list): raw = raw[k]; break
        self.data = require_records(raw, "SIUO")
        for item in self.data:
            if not item.get("question", item.get("prompt", item.get("instruction", ""))):
                raise ValueError("SIUO record is missing its question")
            if not resolve_image_path(item.get("image", item.get("image_path", "")), [self.image_dir]):
                raise FileNotFoundError("Required SIUO image not found")
        logger.info(f"SIUO: {len(self.data)} 条")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ir = item.get("image", item.get("image_path", ""))
        ip = resolve_image_path(ir, [self.image_dir])
        img = load_image(ip)
        q = item.get("question", item.get("prompt", item.get("instruction", "")))
        evidence = cached_evidence(self._evidence, ip, ir, self.require_evidence)
        return {
            "id": str(item.get("id", item.get("question_id", idx))),
            "question_id": item.get("question_id", item.get("id", idx)),
            "image": img,
            "image_path": ip or "",
            "question": q,
            "category": item.get("category", item.get("harm_type", "unknown")),
            "reference_answer": item.get("reference_answer", item.get("reference", item.get("answer", ""))),
            "safety_warning": item.get("safety_warning", item.get("warning", "")),
            "evidence_string": evidence["evidence_string"],
            "reliability": evidence["reliability"],
        }


class MixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=None, ratio=None, shuffle=True, seed=None,
                 text_batch_every=None, text_batch_size=None):
        if batch_size is None:
            batch_size = sum(get_common_value("data.train_mix_ratio_safeeraser", [1,1,1], _COMMON_CFG))
        if seed is None:
            seed = get_common_value("runtime.default_seed", 42, _COMMON_CFG)
        self.ratio = ratio or get_common_value("data.train_mix_ratio_safeeraser", cfg=_COMMON_CFG)
        if len(self.ratio) != 3 or any(r < 0 for r in self.ratio) or not any(self.ratio):
            raise ValueError("SafeEraser batch ratio requires three nonnegative domain weights")
        if batch_size < sum(r > 0 for r in self.ratio):
            raise ValueError("batch_size needs at least one slot per requested domain")
        self.shuffle = shuffle
        self.seed = seed
        if text_batch_every is None:
            text_batch_every = int(get_common_value("training.retain_text_batch_every", cfg=_COMMON_CFG) or 1)
        self.text_batch_every = max(0, int(text_batch_every))
        self.text_batch_size = max(1, int(text_batch_size or batch_size))

        self.forget_ids = dataset.get_split_indices("forget", modality="image")
        self.retain_img_ids = dataset.get_split_indices("retain", modality="image")
        self.retain_txt_ids = dataset.get_split_indices("retain", modality="text")
        self.pd_ids = dataset.get_split_indices("pd", modality="image")

        counts = [int(r > 0) for r in self.ratio]
        remaining = batch_size - sum(counts)
        raw = [remaining * r / sum(self.ratio) for r in self.ratio]
        extra = [int(value) for value in raw]
        order = sorted(range(len(raw)),key=lambda i:raw[i]-extra[i],reverse=True)
        for i in order[:remaining-sum(extra)]:extra[i]+=1
        counts = [base+value for base,value in zip(counts,extra)]
        self.per_batch = counts
        image_group_info = [(self.forget_ids, self.per_batch[0]), (self.retain_img_ids, self.per_batch[1]), (self.pd_ids, self.per_batch[2])]
        for ids, need in image_group_info:
            if need > 0 and len(ids) < need:
                raise ValueError("Training mixture has an empty or undersized required image split")
        image_non_empty = [len(ids) // need for ids, need in image_group_info if need > 0]
        self.num_image_batches = min(image_non_empty) if image_non_empty else 0
        self.num_text_batches = (len(self.retain_txt_ids) + self.text_batch_size - 1) // self.text_batch_size if self.text_batch_every else 0

        logger.info(
            "MixedBatch: f=%d r_img=%d r_txt=%d p=%d -> image_batches=%d text_batches=%d",
            len(self.forget_ids), len(self.retain_img_ids), len(self.retain_txt_ids), len(self.pd_ids),
            self.num_image_batches, self.num_text_batches,
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        def permute(ids):
            if not ids: return []
            if self.shuffle:
                p = torch.randperm(len(ids), generator=g).tolist()
                return [ids[i] for i in p]
            return list(ids)

        f_perm = permute(self.forget_ids)
        r_img_perm = permute(self.retain_img_ids)
        p_perm = permute(self.pd_ids)
        r_txt_perm = permute(self.retain_txt_ids) if self.text_batch_every else []

        fi = ri = pi = ti = 0
        emitted_images = 0

        for _ in range(self.num_image_batches):
            batch = []
            for _ in range(self.per_batch[0]):
                batch.append(f_perm[fi % len(f_perm)]); fi += 1
            for _ in range(self.per_batch[1]):
                batch.append(r_img_perm[ri % len(r_img_perm)]); ri += 1
            for _ in range(self.per_batch[2]):
                batch.append(p_perm[pi % len(p_perm)]); pi += 1
            yield batch
            emitted_images += 1

            if self.num_text_batches > 0 and emitted_images % self.text_batch_every == 0 and ti < len(r_txt_perm):
                yield r_txt_perm[ti:ti + self.text_batch_size]
                ti += self.text_batch_size

        while ti < len(r_txt_perm):
            yield r_txt_perm[ti:ti + self.text_batch_size]
            ti += self.text_batch_size

    def __len__(self):
        inserted = min(self.num_text_batches, self.num_image_batches // self.text_batch_every) if self.num_image_batches and self.text_batch_every else 0
        remaining = max(0, self.num_text_batches - inserted)
        return self.num_image_batches + inserted + remaining


def eviguard_collate_fn(batch):
    return {
        "ids": [b["id"] for b in batch],
        "images": [b["image"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b.get("answer", "") for b in batch],
        "original_answers": [b.get("original_answer", "") for b in batch],
        "splits": [b["split"] for b in batch],
        "modalities": [b.get("modality", "image" if b.get("image") is not None else "text") for b in batch],
        "categories": [b.get("category", "unknown") for b in batch],
        "evidence_strings": [b["evidence_string"] for b in batch],
        "reliabilities": torch.tensor([b["reliability"] for b in batch], dtype=torch.float32),
        "forget_mask": torch.tensor([b["split"] == "forget" for b in batch], dtype=torch.bool),
        "retain_mask": torch.tensor([b["split"] == "retain" for b in batch], dtype=torch.bool),
        "pd_mask": torch.tensor([b["split"] == "pd" for b in batch], dtype=torch.bool),
    }

LLAVA_SYSTEM_MSG = get_common_value("data.llava_system_msg", "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ", _COMMON_CFG)
ANSWER_TAG = get_common_value("data.answer_tag", " ASSISTANT: ", _COMMON_CFG)

_MIXED_MODAL_BATCH_WARNED = False


def prepare_model_inputs(batch, processor, tokenizer, max_length=None,
                         device=None, arch="llava", model=None):
    if max_length is None:
        max_length = get_common_value("runtime.llava.prepare_max_length", 512, _COMMON_CFG)
    if device is None:
        device = torch.device(get_common_value("subspace.default_device", "cpu", _COMMON_CFG))
    if arch == "internvl":
        from src.models.internvl_utils import prepare_model_inputs_internvl
        return prepare_model_inputs_internvl(
            batch, model=model, tokenizer=tokenizer,
            max_length=max_length, device=device)

    return _prepare_model_inputs_llava(batch, processor, tokenizer, max_length, device)


def _prepare_model_inputs_llava(batch, processor, tokenizer, max_length=None, device=None):
    if max_length is None:
        max_length = get_common_value("runtime.llava.prepare_max_length", 512, _COMMON_CFG)
    if device is None:
        device = torch.device(get_common_value("subspace.default_device", "cpu", _COMMON_CFG))
    global _MIXED_MODAL_BATCH_WARNED

    images = batch["images"]
    questions = batch["questions"]
    answers = batch["answers"]

    image_indices = [i for i, img in enumerate(images) if img is not None]
    text_only_indices = [i for i, img in enumerate(images) if img is None]

    keep_indices = list(range(len(images)))

    if image_indices and text_only_indices:
        raise ValueError("Mixed image/text batch is unsupported; use homogeneous modality batches")

    images = [images[i] for i in keep_indices]
    questions = [questions[i] for i in keep_indices]
    answers = [answers[i] for i in keep_indices]

    prompts = []
    full_texts = []
    for q, a, img in zip(questions, answers, images):
        if img is not None:
            p = f"{LLAVA_SYSTEM_MSG}USER: <image>\n{q}{ANSWER_TAG}"
        else:
            p = f"{LLAVA_SYSTEM_MSG}USER: {q}{ANSWER_TAG}"
        prompts.append(p)
        full_texts.append(p + a)

    valid_images = [img for img in images if img is not None]
    if valid_images:
        if processor is None:
            raise ValueError("A visual processor is required for image samples")
        inputs = processor(
            text=full_texts, images=valid_images, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
    else:
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    mi = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    labels = mi["input_ids"].clone()
    if getattr(tokenizer,"padding_side","right") != "right":
        raise ValueError("Training tokenization requires right padding")
    for i, prompt in enumerate(prompts):
        if images[i] is not None:
            prefix = processor(text=prompt, images=images[i], return_tensors="pt",
                               truncation=True, max_length=max_length)
        else:
            prefix = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        prefix_length = prefix["input_ids"].shape[1]
        labels[i, :min(prefix_length, labels.shape[1])] = -100
    labels[mi["attention_mask"] == 0] = -100
    if ((labels[:,1:] != -100).sum(1) == 0).any():
        raise ValueError("Answer targets are empty or entirely truncated; increase max_seq_length")
    mi["labels"] = labels

    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    mi["forget_mask"] = batch["forget_mask"][keep_tensor].to(device)
    mi["retain_mask"] = batch["retain_mask"][keep_tensor].to(device)
    mi["pd_mask"] = batch["pd_mask"][keep_tensor].to(device)
    mi["reliabilities"] = batch["reliabilities"][keep_tensor].to(device)
    mi["evidence_strings"] = [batch["evidence_strings"][i] for i in keep_indices]
    mi["original_answers"] = [batch["original_answers"][i] for i in keep_indices]
    mi["splits"] = [batch["splits"][i] for i in keep_indices]
    mi["kept_indices"] = keep_indices
    return mi


def create_data_loaders(cfg):
    dc = cfg["data"]
    tc = cfg["training"]
    sd = [dc["safeeraser_root"], dc["img_dir"], dc["pdset_img_dir"]]
    require_evidence = dc.get("require_evidence", True)
    tds = SafeEraserDataset(dc["train_json"], sd, "all", dc.get("evidence_cache_dir"), tc["seed"], require_evidence=require_evidence)
    bs = tc.get("batch_size", sum(get_common_value("data.train_mix_ratio_safeeraser", [1,1,1], _COMMON_CFG)))
    smp = MixedBatchSampler(
        tds,
        batch_size=bs,
        ratio=tc["batch_mix_ratio"],
        seed=tc["seed"],
        text_batch_every=tc.get("retain_text_batch_every", 1),
        text_batch_size=tc.get("retain_text_batch_size", bs),
    )
    tl = DataLoader(tds, batch_sampler=smp, collate_fn=eviguard_collate_fn,
                    num_workers=tc.get("num_workers", 4), pin_memory=True)
    vl = None
    val_path = dc.get("val_json")
    if val_path:
        if not os.path.isfile(val_path):
            raise FileNotFoundError(f"Validation JSON does not exist: {val_path}")
        if Path(val_path).resolve() == Path(dc["train_json"]).resolve():
            vds = SafeEraserDataset(val_path, sd, "forget_test", dc.get("evidence_cache_dir"), tc["seed"], require_evidence=require_evidence)
        else:
            vds = SafeEraserDataset(val_path, sd, "all", dc.get("evidence_cache_dir"), tc["seed"], require_evidence=require_evidence)
            trained_images = {str(Path(s["image_path"]).resolve()).casefold() for s in tds.data if s.get("image_path") and s["split"] in ("forget", "retain", "pd")}
            if any(s.get("image_path") and str(Path(s["image_path"]).resolve()).casefold() in trained_images for s in vds.data):
                raise ValueError("Training and validation images overlap")
        if not len(vds):
            raise ValueError("Validation split is empty")
        vl = DataLoader(vds, batch_size=1, shuffle=False, collate_fn=eviguard_collate_fn)
    return tl, vl, None
