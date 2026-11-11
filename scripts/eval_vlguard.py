"""Evaluate VLGuard and optionally SIUO with complete model restoration."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.helpers import load_config,set_seed
from scripts.evaluate import _load_model_for_eval


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/vlguard_llava.yaml")
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint")
    selection.add_argument("--vanilla",action="store_true")
    p.add_argument("--backbone-path")
    p.add_argument("--eval-datasets",nargs="+",choices=["vlguard","siuo"],default=["vlguard"])
    p.add_argument("--tag",default="evaluation")
    a=p.parse_args();cfg=load_config(a.config);set_seed(cfg["training"]["seed"])
    model,processor,tokenizer=_load_model_for_eval(cfg,a.checkpoint,vanilla=a.vanilla,backbone_path=a.backbone_path)
    results={"mode":"vanilla" if a.vanilla else "eviguard"}
    if "vlguard" in a.eval_datasets:
        from src.evaluation.vlguard_evaluator import VLGuardEvaluator
        vc=cfg["vlguard"]
        results["vlguard"]=VLGuardEvaluator(model,processor,tokenizer,cfg).evaluate_all(str(Path(vc["data_dir"])/vc["test_json"]),str(Path(vc["data_dir"])/vc["image_dir_test"]),a.tag)
    if "siuo" in a.eval_datasets:
        from src.data.datasets import SIUODataset
        from src.evaluation.evaluator import Evaluator
        dc=cfg["data"]
        ds=SIUODataset(dc["siuo_gen_json"],dc["siuo_image_dir"],dc["evidence_cache_dir"],require_evidence=not a.vanilla)
        results["siuo"]=Evaluator(model,processor,tokenizer,cfg).evaluate_siuo(ds,a.tag)
    print(json.dumps(results,indent=2,allow_nan=False))

if __name__=="__main__":main()
