"""Evaluate an explicit complete checkpoint or an explicitly selected vanilla model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.helpers import load_config,set_seed


def _load_model_for_eval(cfg, checkpoint=None, *, vanilla=False, backbone_path=None):
    if checkpoint:
        from src.models.checkpoint import load_inference_checkpoint
        model=load_inference_checkpoint(checkpoint,backbone_path=backbone_path)
        if model.arch != cfg["model"]["arch"]:
            raise ValueError("Evaluation config architecture differs from checkpoint")
        return model,model.processor,model.tokenizer
    if not vanilla:
        raise ValueError("Select --checkpoint or --vanilla explicitly")
    from src.models.eviguard_model import load_model_and_processor
    # Fresh zero-initialized LoRA has no delta. No RSF wrapper is applied to vanilla.
    model,processor,tokenizer,_=load_model_and_processor(cfg,load_reference=False)
    model.eval()
    return model,processor,tokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/safeeraser_llava.yaml")
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint",help="Complete maintained-release checkpoint directory")
    selection.add_argument("--vanilla",action="store_true")
    p.add_argument("--backbone-path",help="Explicit relocation of the same frozen backbone")
    p.add_argument("--eval-datasets",nargs="+",choices=["safeeraser","siuo"],default=["safeeraser"])
    p.add_argument("--max-samples",type=int)
    p.add_argument("--tag",default="evaluation")
    a=p.parse_args();cfg=load_config(a.config);set_seed(cfg["training"]["seed"])
    from src.evaluation.evaluator import Evaluator
    model,processor,tokenizer=_load_model_for_eval(cfg,a.checkpoint,vanilla=a.vanilla,backbone_path=a.backbone_path)
    evaluator=Evaluator(model,processor,tokenizer,cfg)
    dc=cfg["data"];required=not a.vanilla
    results={"evaluation_scope":"subset" if a.max_samples else "full","max_samples":a.max_samples,"mode":"vanilla" if a.vanilla else "eviguard"}
    if "safeeraser" in a.eval_datasets:
        from src.data.datasets import SafeEraserDataset
        ds=SafeEraserDataset(dc["train_json"],[dc["safeeraser_root"],dc["img_dir"],dc["pdset_img_dir"]],"all",dc["evidence_cache_dir"],cfg["training"]["seed"],require_evidence=required)
        results["safeeraser"]=evaluator.evaluate_safeeraser(ds,a.tag,a.max_samples)
    if "siuo" in a.eval_datasets:
        from src.data.datasets import SIUODataset
        ds=SIUODataset(dc["siuo_gen_json"],dc["siuo_image_dir"],dc["evidence_cache_dir"],require_evidence=required)
        results["siuo"]=evaluator.evaluate_siuo(ds,a.tag,a.max_samples)
    print(json.dumps(results,indent=2,allow_nan=False))

if __name__=="__main__":main()
