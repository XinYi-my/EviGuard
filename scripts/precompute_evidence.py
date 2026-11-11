"""Extract frozen visual evidence before training. No benchmark metrics are produced."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.helpers import load_config
from src.modules.evidence import EvidencePipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--no-ocr",action="store_true")
    parser.add_argument("--no-tags",action="store_true")
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--dataset",choices=["train","siuo"],default="train")
    args=parser.parse_args()
    cfg=load_config(args.config)
    if args.dataset == "siuo":
        roots=[Path(cfg["data"]["siuo_image_dir"])]
    elif "vlguard" in cfg:
        v=cfg["vlguard"]; roots=[Path(v["data_dir"])/v["image_dir_train"],Path(v["data_dir"])/v["image_dir_test"]]
    else:
        roots=[Path(cfg["data"]["img_dir"]),Path(cfg["data"]["pdset_img_dir"])]
    records={}
    for root in roots:
        if not root.is_dir(): raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in cfg["data"]["image_extensions"]:
                records[str(path.resolve())]=path
    if not records: raise ValueError("No input images found")
    pipeline=EvidencePipeline(cfg["data"]["evidence_cache_dir"],not args.no_ocr,not args.no_tags,cfg["evidence_models"]["blip2"],cfg["rsf"]["default_reliability"])
    failures=0
    for index,(key,path) in enumerate(records.items(),1):
        result=pipeline.extract_single(str(path),force=args.force,cache_key=key)
        failures+=bool(result.get("extraction_errors"))
        if index%100==0: pipeline.cache.save()
    pipeline.cache.save()
    print(f"Cached {len(records)} images; {failures} have explicitly recorded extractor errors.")
    if failures: raise RuntimeError("Review extraction_errors in the cache before training")

if __name__=="__main__": main()
