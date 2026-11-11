import sys, os, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, get_default_config_path, resolve_backbone_model_path, get_arch_override_cfg




def main():
    parser = argparse.ArgumentParser(description="EviGuard on VLGuard — training")
    parser.add_argument("--config", default=get_default_config_path("train_vlguard"))
    parser.add_argument("--model", default=None, choices=["llava", "internvl"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model is not None:
        arch_cfg = get_arch_override_cfg(args.model, cfg)
        cfg["model"]["name"] = resolve_backbone_model_path(args.model)
        cfg["model"]["arch"] = arch_cfg.get("arch", args.model)
        cfg["lora"]["target_modules"] = list(arch_cfg.get("lora_target_modules", cfg["lora"]["target_modules"]))

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    cfg["loss"]["use_pd"] = cfg["loss"]["use_pd"]
    cfg["loss"]["lambda_pd"] = cfg["loss"]["lambda_pd"]

    arch = cfg["model"]["arch"]

    for d in [cfg["paths"]["output_dir"],
              cfg["paths"]["checkpoint_dir"],
              cfg["paths"]["log_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    rcfg = cfg["runtime"]
    logging.basicConfig(
        level=getattr(logging, rcfg["logging"]["level"]),
        format=rcfg["logging"]["format"],
        datefmt=rcfg["logging"]["date_format"],
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(cfg["paths"]["log_dir"], rcfg["files"]["train_vlguard_log"])),
        ])
    logger = logging.getLogger("eviguard")
    logger.info("EviGuard on VLGuard — PO backbone (no PD) + RSF+CLSU+Feedback")
    logger.info("Config: %s  |  Arch: %s  |  Model: %s", args.config, arch, cfg["model"]["name"])
    logger.info("use_pd=False, lambda_pd=0.0")

    from src.utils.helpers import set_seed
    set_seed(cfg["training"]["seed"])

    from src.models.eviguard_model import load_model_and_processor, EviGuardModel
    model, processor, tokenizer, ref_model = load_model_and_processor(cfg)
    eviguard = EviGuardModel(model, ref_model, processor, tokenizer, cfg)

    from src.data.vlguard_datasets import create_vlguard_data_loaders
    train_loader, test_loaders = create_vlguard_data_loaders(cfg)

    from src.training.trainer import EviGuardTrainer
    trainer = EviGuardTrainer(eviguard, train_loader, val_loader=None, cfg=cfg)
    if args.resume is not None:
        resume_path = None if args.resume in (None, "auto", "") else args.resume
        trainer.load_resume_checkpoint(resume_path)
    trainer.train()

    logger.info("VLGuard EviGuard training complete.")
    logger.info("Checkpoint: %s", cfg["paths"]["checkpoint_dir"])


if __name__ == "__main__":
    main()
