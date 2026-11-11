"""Validate configured local datasets and evidence without loading backbone weights."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.helpers import load_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument("--check-models",action="store_true")
    a=p.parse_args();cfg=load_config(a.config)
    if a.check_models:
        for name in [cfg["model"]["name"],cfg["evidence_models"]["blip2"]]:
            if not Path(name).is_dir():raise FileNotFoundError(name)
    if "vlguard" in cfg:
        from src.data.vlguard_datasets import create_vlguard_data_loaders
        train,tests=create_vlguard_data_loaders(cfg)
        counts={"train":len(train.dataset),**{k:len(v.dataset) for k,v in tests.items()}}
    else:
        from src.data.datasets import create_data_loaders
        train,val,siuo=create_data_loaders(cfg)
        counts={"train":len(train.dataset),"validation":len(val.dataset) if val else None,"siuo":len(siuo.dataset) if siuo else None}
    for index in range(len(train.dataset)):train.dataset[index]
    print(json.dumps({"dataset_sizes":counts,"training_batches":len(train)},indent=2))

if __name__=="__main__":main()
