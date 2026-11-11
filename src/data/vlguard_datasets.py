import json, os, logging, random
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from src.utils.helpers import load_common_config, get_common_value
from src.data.datasets import resolve_image_path, require_records, load_image, evaluation_collate_fn, load_evidence_cache, cached_evidence

logger = logging.getLogger("eviguard.data.vlguard")
_COMMON_CFG = load_common_config()

IDK_POOL = _COMMON_CFG["autogen"]["refusal_templates"]

def resolve_vlguard_image_path(image_ref: str, image_dir: str) -> Optional[str]:
    path = resolve_image_path(image_ref, [image_dir])
    if not path:
        raise FileNotFoundError(f"Required VLGuard image not found: {image_ref}")
    return path


def flatten_vlguard_train(raw_data: list, image_dir: str, seed: int) -> List[Dict]:
    require_records(raw_data, "VLGuard train")
    rng = random.Random(seed)
    samples = []
    sid = 0

    for rec in raw_data:
        is_safe = rec.get("safe", True)
        image_ref = rec.get("image", "")
        image_path = resolve_vlguard_image_path(image_ref, image_dir)
        category = rec.get("harmful_category", "unknown")
        subcategory = rec.get("harmful_subcategory", "")
        instr_resp_list = rec.get("instr-resp", [])

        if not is_safe:
            # ---- Unsafe → forget set ----
            for ir in instr_resp_list:
                q = ir.get("instruction", "")
                orig_answer = ir.get("response", "")
                if not q:
                    continue
                idk_answer = rng.choice(IDK_POOL)
                samples.append({
                    "id": f"vlg_forget_{sid}",
                    "image_ref": image_ref,
                    "image_path": image_path,
                    "question": q,
                    "answer": idk_answer,
                    "original_answer": orig_answer,
                    "split": "forget",
                    "modality": "image",
                    "category": category,
                    "subcategory": subcategory,
                })
                sid += 1
        else:
            for ir in instr_resp_list:
                if "safe_instruction" in ir:
                    q = ir["safe_instruction"]
                    a = ir.get("response", "")
                    if not q:
                        continue
                    samples.append({
                        "id": f"vlg_retain_{sid}",
                        "image_ref": image_ref,
                        "image_path": image_path,
                        "question": q,
                        "answer": a,
                        "original_answer": "",
                        "split": "retain",
                        "modality": "image",
                        "category": "safe",
                        "subcategory": "",
                    })
                    sid += 1

    sc = {}
    for s in samples:
        sc[s["split"]] = sc.get(s["split"], 0) + 1
    logger.info("VLGuard train: %d raw → %d samples, splits: %s", len(raw_data), len(samples), sc)
    return samples


def flatten_vlguard_test(raw_data: list, image_dir: str) -> Dict[str, List[Dict]]:
    require_records(raw_data, "VLGuard test")
    subsets = {"unsafes": [], "safe_unsafes": [], "safe_safes": []}
    sid = 0

    for rec in raw_data:
        is_safe = rec.get("safe", True)
        image_ref = rec.get("image", "")
        image_path = resolve_vlguard_image_path(image_ref, image_dir)
        category = rec.get("harmful_category", "unknown") if not is_safe else "safe"
        instr_resp_list = rec.get("instr-resp", [])

        if not is_safe:
            for ir in instr_resp_list:
                q = ir.get("instruction", "")
                a = ir.get("response", "")
                if not q:
                    continue
                subsets["unsafes"].append({
                    "id": f"vlg_test_unsafe_{sid}",
                    "image_ref": image_ref,
                    "image_path": image_path,
                    "question": q,
                    "answer": a,
                    "category": category,
                })
                sid += 1
        else:
            for ir in instr_resp_list:
                if "unsafe_instruction" in ir:
                    q = ir["unsafe_instruction"]
                    a = ir.get("response", "")
                    if q:
                        subsets["safe_unsafes"].append({
                            "id": f"vlg_test_safeunsafe_{sid}",
                            "image_ref": image_ref,
                            "image_path": image_path,
                            "question": q,
                            "answer": a,
                            "category": "safe_unsafe",
                        })
                        sid += 1
                if "safe_instruction" in ir:
                    q = ir["safe_instruction"]
                    a = ir.get("response", "")
                    if q:
                        subsets["safe_safes"].append({
                            "id": f"vlg_test_safesafe_{sid}",
                            "image_ref": image_ref,
                            "image_path": image_path,
                            "question": q,
                            "answer": a,
                            "category": "safe_safe",
                        })
                        sid += 1

    for k, v in subsets.items():
        logger.info("VLGuard test subset '%s': %d samples", k, len(v))
    return subsets


class VLGuardTrainDataset(Dataset):

    def __init__(self, json_path, image_dir, split_filter="all",
                 evidence_cache_dir=None, seed=None, require_evidence=False):
        if seed is None:
            seed = _COMMON_CFG["runtime"]["default_seed"]
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        all_s = flatten_vlguard_train(raw, image_dir, seed)
        self.data = [s for s in all_s if s["split"] == split_filter] if split_filter != "all" else all_s

        self._split_indices = {}
        self._split_modality_indices = {}
        for i, s in enumerate(self.data):
            self._split_indices.setdefault(s["split"], []).append(i)
            md = s.get("modality", "image")
            self._split_modality_indices.setdefault((s["split"], md), []).append(i)

        self._evidence = load_evidence_cache(evidence_cache_dir)
        self.require_evidence = require_evidence

        logger.info("VLGuardTrainDataset: %d samples, filter='%s', splits=%s",
                     len(self.data), split_filter,
                     {k: len(v) for k, v in self._split_indices.items()})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        image = load_image(s["image_path"])
        ev = cached_evidence(self._evidence, s.get("image_path"), s.get("image_ref", ""), self.require_evidence)
        return {
            "id": s["id"],
            "image": image,
            "image_path": s["image_path"] or "",
            "question": s["question"],
            "answer": s["answer"],
            "original_answer": s.get("original_answer", ""),
            "split": s["split"],
            "modality": s.get("modality", "image"),
            "category": s["category"],
            "evidence_string": ev.get("evidence_string", "[OCR]  [TAGS] "),
            "reliability": ev.get("reliability", get_common_value("rsf.default_reliability", cfg=_COMMON_CFG) or 0.5),
        }

    def get_split_indices(self, name, modality=None):
        if modality is None:
            return self._split_indices.get(name, [])
        return self._split_modality_indices.get((name, modality), [])


class VLGuardTestDataset(Dataset):

    def __init__(self, json_path, image_dir, subset_name, evidence_cache_dir=None, require_evidence=False):
        self._evidence = load_evidence_cache(evidence_cache_dir)
        self.require_evidence = require_evidence
        assert subset_name in ("unsafes", "safe_unsafes", "safe_safes"), \
            f"Invalid subset: {subset_name}. Must be one of: unsafes, safe_unsafes, safe_safes"
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        subsets = flatten_vlguard_test(raw, image_dir)
        self.data = subsets[subset_name]
        if not self.data:
            raise ValueError(f"VLGuard subset {subset_name} is empty")
        self.subset_name = subset_name
        logger.info("VLGuardTestDataset[%s]: %d samples", subset_name, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        image = load_image(s["image_path"])
        evidence = cached_evidence(self._evidence, s.get("image_path"), s.get("image_ref", ""), self.require_evidence)
        return {
            "id": s["id"],
            "image": image,
            "image_path": s["image_path"] or "",
            "question": s["question"],
            "answer": s.get("answer", ""),
            "category": s.get("category", "unknown"),
            "evidence_string": evidence["evidence_string"],
            "reliability": evidence["reliability"],
            "modality": "image",
        }


class VLGuardBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=None, ratio=None, shuffle=True, seed=None):
        if batch_size is None:
            batch_size = sum(get_common_value("data.train_mix_ratio_vlguard", cfg=_COMMON_CFG))
        self.ratio = ratio or get_common_value("data.train_mix_ratio_vlguard", cfg=_COMMON_CFG)
        if len(self.ratio) != 2 or any(r <= 0 for r in self.ratio):
            raise ValueError("VLGuard batch ratio requires two positive domain weights")
        if batch_size < 2:
            raise ValueError("VLGuard batch_size needs at least two domain slots")
        self.shuffle = shuffle
        self.seed = int(seed if seed is not None else get_common_value("runtime.default_seed", cfg=_COMMON_CFG))

        self.forget_ids = dataset.get_split_indices("forget", modality="image")
        self.retain_ids = dataset.get_split_indices("retain", modality="image")

        counts = [int(r > 0) for r in self.ratio]
        remaining = batch_size - sum(counts)
        raw = [remaining * r / sum(self.ratio) for r in self.ratio]
        extra = [int(value) for value in raw]
        order = sorted(range(len(raw)),key=lambda i:raw[i]-extra[i],reverse=True)
        for i in order[:remaining-sum(extra)]:extra[i]+=1
        counts = [base+value for base,value in zip(counts,extra)]
        self.per_batch = counts
        if any(need > len(ids) for need, ids in zip(self.per_batch, [self.forget_ids, self.retain_ids])):
            raise ValueError("VLGuard mixture has an empty or undersized required split")

        non_empty = []
        if self.per_batch[0] > 0 and self.forget_ids:
            non_empty.append(len(self.forget_ids) // self.per_batch[0])
        if self.per_batch[1] > 0 and self.retain_ids:
            non_empty.append(len(self.retain_ids) // self.per_batch[1])
        self.num_batches = min(non_empty) if non_empty else 0

        logger.info("VLGuardBatchSampler: forget=%d, retain=%d, per_batch=%s, batches=%d",
                     len(self.forget_ids), len(self.retain_ids),
                     self.per_batch, self.num_batches)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        def permute(ids):
            if not ids:
                return []
            if self.shuffle:
                p = torch.randperm(len(ids), generator=g).tolist()
                return [ids[i] for i in p]
            return list(ids)

        f_perm = permute(self.forget_ids)
        r_perm = permute(self.retain_ids)
        fi = ri = 0

        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.per_batch[0]):
                batch.append(f_perm[fi % len(f_perm)])
                fi += 1
            for _ in range(self.per_batch[1]):
                batch.append(r_perm[ri % len(r_perm)])
                ri += 1
            yield batch

    def __len__(self):
        return self.num_batches


def vlguard_collate_fn(batch):
    return {
        "ids": [b["id"] for b in batch],
        "images": [b["image"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b.get("answer", "") for b in batch],
        "original_answers": [b.get("original_answer", "") for b in batch],
        "splits": [b["split"] for b in batch],
        "modalities": [b.get("modality", "image") for b in batch],
        "categories": [b.get("category", "unknown") for b in batch],
        "evidence_strings": [b["evidence_string"] for b in batch],
        "reliabilities": torch.tensor([b["reliability"] for b in batch], dtype=torch.float32),
        "forget_mask": torch.tensor([b["split"] == "forget" for b in batch], dtype=torch.bool),
        "retain_mask": torch.tensor([b["split"] == "retain" for b in batch], dtype=torch.bool),
        "pd_mask": torch.tensor([False for _ in batch], dtype=torch.bool),  # ★ VLGuard 无 PD
    }


def create_vlguard_data_loaders(cfg):
    vc = cfg.get("vlguard", {})
    tc = cfg["training"]

    if not vc:
        raise ValueError("Config missing 'vlguard' section. Please add VLGuard config.")

    data_dir = vc["data_dir"]
    train_json = os.path.join(data_dir, vc["train_json"])
    test_json = os.path.join(data_dir, vc["test_json"])
    image_dir_train = os.path.join(data_dir, vc["image_dir_train"])
    image_dir_test = os.path.join(data_dir, vc["image_dir_test"])
    evidence_cache = cfg["data"].get("evidence_cache_dir")

    require_evidence = cfg["data"].get("require_evidence", True)
    train_ds = VLGuardTrainDataset(train_json, image_dir_train, "all", evidence_cache, tc["seed"], require_evidence=require_evidence)
    bs = tc.get("batch_size", int(get_common_value("training.batch_size", cfg=cfg) or 2))
    sampler = VLGuardBatchSampler(
        train_ds, batch_size=bs,
        ratio=tc.get("vlguard_batch_mix_ratio", get_common_value("training.vlguard_batch_mix_ratio", cfg=cfg) or [1, 1]),
        seed=tc["seed"],
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=vlguard_collate_fn,
        num_workers=tc.get("num_workers", int(get_common_value("training.num_workers", cfg=cfg) or 4)), pin_memory=True,
    )

    test_loaders = {}
    if not os.path.isfile(test_json):
        raise FileNotFoundError(f"VLGuard test JSON does not exist: {test_json}")
    if os.path.exists(test_json):
        for subset in ["unsafes", "safe_unsafes", "safe_safes"]:
            test_ds = VLGuardTestDataset(test_json, image_dir_test, subset, evidence_cache, require_evidence=require_evidence)
            test_loaders[subset] = DataLoader(
                test_ds, batch_size=int(cfg["evaluation"].get("batch_size", get_common_value("evaluation.defaults.batch_size", cfg=cfg) or 4)),
                shuffle=False,
                collate_fn=evaluation_collate_fn,
            )

    return train_loader, test_loaders
