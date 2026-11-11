import sys, os, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, get_default_config_path, resolve_backbone_model_path, get_arch_override_cfg


def main():
    parser = argparse.ArgumentParser(
        description="EviGuard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=get_default_config_path("train"),
                        help="配置文件路径")
    parser.add_argument("--model", default=None,
                        choices=["llava", "internvl"],
                        help="选择训练骨干: llava=LLaVA-1.5-7B, internvl=InternVL2.5-8B; "
                             "覆盖 config 中的 model.name")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (覆盖 config 中的 training.seed)")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="从最近 checkpoint 断点续跑; 可选传入具体 ckpt 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model is not None:
        arch_cfg = get_arch_override_cfg(args.model, cfg)
        cfg["model"]["name"] = resolve_backbone_model_path(args.model)
        cfg["model"]["arch"] = arch_cfg.get("arch", args.model)
        cfg["lora"]["target_modules"] = list(arch_cfg.get("lora_target_modules", cfg["lora"]["target_modules"]))

    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

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
            logging.FileHandler(
                os.path.join(cfg["paths"]["log_dir"], rcfg["files"]["train_log"])),
        ])
    logger = logging.getLogger("eviguard")
    logger.info("EviGuard — PO+PD backbone + RSF+CLSU+Feedback")
    logger.info("Config: %s  |  Arch: %s  |  Model: %s",
                args.config, arch, cfg["model"]["name"])
    logger.info("Seed: %d", cfg["training"]["seed"])

    from src.utils.helpers import set_seed
    set_seed(cfg["training"]["seed"])

    from src.models.eviguard_model import load_model_and_processor, EviGuardModel
    model, processor, tokenizer, ref_model = load_model_and_processor(cfg)
    eviguard = EviGuardModel(model, ref_model, processor, tokenizer, cfg)

    from src.data.datasets import create_data_loaders
    train_loader, val_loader, siuo_loader = create_data_loaders(cfg)

    from src.training.trainer import EviGuardTrainer
    trainer = EviGuardTrainer(eviguard, train_loader, val_loader, cfg)
    if args.resume is not None:
        resume_path = None if args.resume in (None, "auto", "") else args.resume
        trainer.load_resume_checkpoint(resume_path)
    trainer.train()

    logger.info("训练完成. checkpoint 保存在: %s", cfg["paths"]["checkpoint_dir"])
    logger.info("请运行 scripts/evaluate.py 进行评估.")


if __name__ == "__main__":
    main()
