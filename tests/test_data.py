import json

import pytest
import torch
from PIL import Image

from src.data.datasets import SafeEraserDataset, SIUODataset, MixedBatchSampler, flatten_safeeraser, prepare_model_inputs, create_data_loaders
from src.data.vlguard_datasets import VLGuardTestDataset, VLGuardTrainDataset, VLGuardBatchSampler, create_vlguard_data_loaders


def write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def records(tmp_path, count=6):
    rows = []
    for i in range(count):
        Image.new("RGB", (2, 2)).save(tmp_path / f"{i}.png")
        Image.new("RGB", (2, 2)).save(tmp_path / f"pd{i}.png")
        rows.append({"image_path": f"{i}.png", "SDImage_path": f"pd{i}.png",
                     "unsafe_pairs": [{"question": f"q{i}-{j}", "answer": "original", "sd_response": "benign"} for j in range(3)]})
    return rows


def test_safeeraser_holdout_is_by_image_and_stable_under_record_order(tmp_path):
    raw = records(tmp_path)
    rows = flatten_safeeraser(raw, [str(tmp_path)], 42)
    heldout = {r["image_ref"] for r in rows if r["split"] == "forget_test"}
    trained = {r["image_ref"] for r in rows if r["split"] == "forget"}
    assert heldout and trained and heldout.isdisjoint(trained)
    again = flatten_safeeraser(list(reversed(raw)), [str(tmp_path)], 42)
    assert {(r["image_ref"], r["question"], r["split"]) for r in rows} == {(r["image_ref"], r["question"], r["split"]) for r in again}
    heldout_questions = {r["question"] for r in rows if r["split"] == "forget_test"}
    assert not any(r["split"] == "pd" and r["question"] in heldout_questions for r in rows)


@pytest.mark.parametrize("kind", ["safeeraser", "siuo", "vlguard_train", "vlguard_test"])
def test_missing_images_fail_instead_of_turning_into_text(kind, tmp_path):
    with pytest.raises((FileNotFoundError, ValueError), match="image|Image"):
        if kind == "safeeraser":
            ds = SafeEraserDataset(write_json(tmp_path, "data.json", [{"image_path": "missing.png", "unsafe_pairs": [{"question": "q"}]}]), [str(tmp_path)])
        elif kind == "siuo":
            ds = SIUODataset(write_json(tmp_path, "data.json", [{"image": "missing.png", "question": "q"}]), str(tmp_path))
        else:
            path = write_json(tmp_path, "data.json", [{"image": "missing.png", "safe": False, "instr-resp": [{"instruction": "q"}]}])
            ds = VLGuardTrainDataset(path, str(tmp_path)) if kind.endswith("train") else VLGuardTestDataset(path, str(tmp_path), "unsafes")
        ds[0]


def test_corrupt_image_is_not_silently_ignored(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises((ValueError, OSError), match="image|Image"):
        ds = SIUODataset(write_json(tmp_path, "data.json", [{"image": "bad.png", "question": "q"}]), str(tmp_path))
        ds[0]


def test_empty_benchmark_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty|nonempty|non-empty"):
        SIUODataset(write_json(tmp_path, "empty.json", []), str(tmp_path))


def test_visual_processor_failure_does_not_retry_as_text():
    class FailingProcessor:
        def __call__(self, **kwargs):
            raise RuntimeError("vision preprocessing failed")
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}
    batch = {"images": [Image.new("RGB", (2, 2))], "questions": ["q"], "answers": ["a"],
             "forget_mask": torch.tensor([True]), "retain_mask": torch.tensor([False]),
             "pd_mask": torch.tensor([False]), "reliabilities": torch.tensor([0.5]),
             "evidence_strings": [""], "original_answers": ["original"], "splits": ["forget"]}
    with pytest.raises(RuntimeError, match="vision preprocessing failed"):
        prepare_model_inputs(batch, FailingProcessor(), Tokenizer(), device="cpu")


class SplitDataset:
    def get_split_indices(self, name, modality=None):
        return {("forget", "image"): [0, 1, 2], ("retain", "image"): [3, 4, 5],
                ("pd", "image"): [6, 7, 8], ("retain", "text"): [9, 10]}.get((name, modality), [])


def test_sampler_cannot_silently_omit_requested_domains():
    with pytest.raises(ValueError, match="batch|slot|domain"):
        MixedBatchSampler(SplitDataset(), batch_size=1, ratio=[1, 1, 1])


def test_zero_text_frequency_disables_text_batches():
    sampler = MixedBatchSampler(SplitDataset(), batch_size=3, ratio=[1, 1, 1], text_batch_every=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3
    assert all(len(batch) == 3 and not any(index >= 9 for index in batch) for batch in batches)


def test_vlguard_sampler_respects_small_batch_error():
    with pytest.raises(ValueError, match="batch|slot|domain"):
        VLGuardBatchSampler(SplitDataset(), batch_size=1, ratio=[1, 1])


def test_training_requires_cached_evidence_and_reads_absolute_keys(tmp_path):
    raw = records(tmp_path)
    path = write_json(tmp_path, "data.json", raw)
    with pytest.raises((ValueError, FileNotFoundError), match="evidence"):
        ds = SafeEraserDataset(path, [str(tmp_path)], evidence_cache_dir=str(tmp_path), require_evidence=True)
        ds[0]
    cache = {str((tmp_path / f"{i}.png").resolve()).replace("\\", "/"): {"evidence_string": "[OCR] test [TAGS] object", "reliability": 0.7} for i in range(6)}
    write_json(tmp_path, "evidence_cache.json", cache)
    ds = SafeEraserDataset(path, [str(tmp_path)], split_filter="forget", evidence_cache_dir=str(tmp_path), require_evidence=True)
    assert ds[0]["evidence_string"] == "[OCR] test [TAGS] object"
    assert ds[0]["reliability"] == 0.7


def test_training_masks_expanded_visual_prefix_and_keeps_answer_eos():
    class Tokenizer:
        pad_token_id=2
        padding_side="right"
        def __call__(self,text,**kwargs):
            return {"input_ids":torch.tensor([[1,3]])}
    class Processor:
        def __call__(self,text,images,**kwargs):
            if isinstance(text,list):
                return {"input_ids":torch.tensor([[1,9,9,3,4,2]]),"attention_mask":torch.ones(1,6,dtype=torch.long),"pixel_values":torch.ones(1,2,2)}
            return {"input_ids":torch.tensor([[1,9,9,3]]),"attention_mask":torch.ones(1,4,dtype=torch.long)}
    batch={"images":[Image.new("RGB",(2,2))],"questions":["q"],"answers":["a"],"forget_mask":torch.tensor([True]),"retain_mask":torch.tensor([False]),"pd_mask":torch.tensor([False]),"reliabilities":torch.tensor([.5]),"evidence_strings":["x"],"original_answers":["original"],"splits":["forget"]}
    result=prepare_model_inputs(batch,Processor(),Tokenizer(),device="cpu")
    assert result["labels"].tolist()==[[-100,-100,-100,-100,4,2]]
