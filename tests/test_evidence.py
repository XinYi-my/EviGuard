import pytest
from src.modules.evidence import EvidenceCache, EvidencePipeline

def test_cache_does_not_alias_equal_basenames(tmp_path):
    cache=EvidenceCache(tmp_path)
    cache.put("a/image.png",{"evidence_string":"a"})
    assert cache.get("b/image.png") is None

def test_pipeline_exposes_release_defaults(tmp_path):
    pipeline=EvidencePipeline(cache_dir=tmp_path,use_ocr=False,use_tags=False,blip2_model="unused")
    assert pipeline.default_reliability==0.5

def test_missing_image_fails(tmp_path):
    pipeline=EvidencePipeline(tmp_path,False,False,"unused",0.5)
    with pytest.raises(FileNotFoundError):
        pipeline.extract_single(str(tmp_path/"missing.png"))
