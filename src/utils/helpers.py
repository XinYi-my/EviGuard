import os, sys, json, random, logging, hashlib, datetime, re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml, numpy as np, torch


COMMON_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "common.yaml"


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def _expand_env_in_obj(obj):
    if isinstance(obj, dict):
        return {k: _expand_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_obj(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    return obj


def load_common_config() -> Dict[str, Any]:
    if not COMMON_CONFIG_PATH.exists():
        return {}
    with open(COMMON_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env_in_obj(data)


def get_common_value(path: str, default: Any = None, cfg: Optional[Dict[str, Any]] = None) -> Any:
    node = cfg if cfg is not None else load_common_config()
    if not path:
        return node
    cur = node
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def get_default_config_path(script_name: str) -> str:
    return get_common_value(f"runtime.default_configs.{script_name}", "")


def get_arch_override_cfg(arch: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source = cfg or load_common_config()
    arch = (arch or get_common_value("runtime.defaults.arch", "llava", source)).lower()
    return get_common_value(f"runtime.model_overrides.{arch}", {}, source)


def get_script_cli_default(key: str, default: Any = None, cfg: Optional[Dict[str, Any]] = None) -> Any:
    return get_common_value(f"runtime.cli_defaults.{key}", default, cfg)


def get_quantization_kwargs(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source = cfg or load_common_config()
    qcfg = get_common_value("runtime.quantization", {}, source)
    mcfg = source.get("model", {})
    dtype_name = mcfg.get("bnb_4bit_compute_dtype", qcfg.get("bnb_4bit_compute_dtype", "float16"))
    return {
        "load_in_4bit": bool(mcfg.get("load_in_4bit", qcfg.get("load_in_4bit", True))),
        "llm_int8_skip_modules": ["mlp1" if get_common_value("model.arch", "llava", source) == "internvl" else "multi_modal_projector"],
        "bnb_4bit_compute_dtype": getattr(torch, dtype_name),
        "bnb_4bit_quant_type": qcfg.get("bnb_4bit_quant_type", "nf4"),
        "bnb_4bit_use_double_quant": bool(qcfg.get("bnb_4bit_use_double_quant", True)),
    }


def resolve_model_path(model_key: str, fallback: Optional[str] = None) -> str:
    env_name = get_common_value(f"runtime.model_env_vars.{model_key}", "")
    env_value = os.environ.get(env_name, "").strip() if env_name else ""
    if env_value:
        return os.path.expandvars(os.path.expanduser(env_value))
    configured_default = get_common_value(f"runtime.model_local_paths.{model_key}", fallback or "")
    target = fallback or configured_default
    return os.path.expandvars(os.path.expanduser(target)) if isinstance(target, str) else target


def resolve_backbone_model_path(arch: str) -> str:
    arch = (arch or get_common_value("runtime.defaults.arch", "llava")).lower()
    if arch not in ("llava", "internvl"):
        raise ValueError(f"Unsupported arch: {arch}")
    return resolve_model_path(arch)


def apply_runtime_path_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _expand_env_in_obj(cfg)
    model_cfg = cfg.get("model", {})
    default_arch = get_common_value("runtime.defaults.arch", "llava", cfg)
    model_cfg.setdefault("arch", default_arch)
    arch = str(model_cfg["arch"]).lower()
    if "name" in model_cfg:
        model_cfg["name"] = resolve_model_path(arch, model_cfg["name"])
    evidence_cfg = cfg.get("evidence_models", {})
    if isinstance(evidence_cfg, dict) and "blip2" in evidence_cfg:
        evidence_cfg["blip2"] = resolve_model_path("blip2", evidence_cfg.get("blip2"))
    judge_cfg = cfg.get("judge_models", {})
    if isinstance(judge_cfg, dict) and "roberta_toxic" in judge_cfg:
        judge_cfg["roberta_toxic"] = resolve_model_path("roberta_toxic", judge_cfg.get("roberta_toxic"))
    return cfg


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    common_cfg = load_common_config()
    config_path = path or get_default_config_path("train")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    merged = _deep_merge_dicts(common_cfg, cfg)
    merged.setdefault("meta", {})["config_path"] = config_path
    return apply_runtime_path_overrides(merged)


def flatten_config(cfg, prefix=""):
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, key))
        else:
            flat[key] = v
    return flat


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name="eviguard", log_dir=None, level=None, console=None):
    common = load_common_config()
    if log_dir is None:
        log_dir = get_common_value("runtime.helper_paths.metric_log_dir", "./logs", common)
    if level is None:
        level_name = get_common_value("runtime.logging.level", "INFO", common)
        level = getattr(logging, str(level_name).upper(), logging.INFO)
    if console is None:
        console = bool(get_common_value("runtime.logging.console", True, common))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{ts}.log"
    logger = logging.getLogger(name)
    logger.setLevel(level); logger.handlers.clear()
    fmt = logging.Formatter(
        get_common_value("runtime.logging.format", "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", common),
        datefmt=get_common_value("runtime.logging.date_format", "%Y-%m-%d %H:%M:%S", common),
    )
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setLevel(level); fh.setFormatter(fmt); logger.addHandler(fh)
    if console:
        ch = logging.StreamHandler(sys.stdout); ch.setLevel(level); ch.setFormatter(fmt); logger.addHandler(ch)
    logger.info(f"日志文件: {log_file}")
    return logger


class MetricTracker:
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = get_common_value("runtime.helper_paths.metric_log_dir", "./logs")
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []; self._current = {}

    def update(self, step, **kw):
        rec = {"step": step, **kw}; self.history.append(rec); self._current.update(kw)

    @property
    def current(self):
        return dict(self._current)

    def save(self, fn="metrics.json"):
        p = self.log_dir / fn
        with open(p, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        return str(p)

    def get_best(self, metric, mode="min"):
        valid = [r for r in self.history if metric in r]
        if not valid:
            return {}
        return (min if mode == "min" else max)(valid, key=lambda r: r[metric])


class StateSaver:
    def __init__(self, save_dir=None):
        if save_dir is None:
            save_dir = get_common_value("runtime.helper_paths.state_saver_dir", "./outputs/intermediates")
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = []

    def save_tensor(self, name, tensor, step=0):
        p = self.save_dir / f"{name}_step{step}.pt"
        torch.save(tensor.detach().cpu(), p)
        self.manifest.append({"name": name, "step": step, "path": str(p), "shape": list(tensor.shape)})
        return str(p)

    def save_numpy(self, name, arr, step=0):
        p = self.save_dir / f"{name}_step{step}.npy"; np.save(p, arr)
        self.manifest.append({"name": name, "step": step, "path": str(p)}); return str(p)

    def save_json(self, name, data, step=0):
        p = self.save_dir / f"{name}_step{step}.json"
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.manifest.append({"name": name, "step": step, "path": str(p)}); return str(p)

    def save_manifest(self):
        p = self.save_dir / "manifest.json"
        with open(p, "w") as f:
            json.dump(self.manifest, f, indent=2)
        return str(p)


def get_device():
    fallback = get_common_value("subspace.default_device", "cpu")
    if torch.cuda.is_available() and str(fallback).startswith("cuda"):
        return torch.device("cuda")
    return torch.device(fallback)


def print_gpu_info():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} — {p.total_memory/1024**3:.1f} GB")
    else:
        print("  无GPU可用")


def _checkpoint_sort_key(name):
    nums = re.findall(r"(\d+)", name)
    return tuple(int(n) for n in nums) if nums else (-1,)


def find_latest_checkpoint(checkpoint_dir):
    ck = Path(checkpoint_dir)
    hint = ck / get_common_value("runtime.files.latest_checkpoint_hint", "latest_checkpoint.txt")
    if hint.exists():
        target = ck / hint.read_text(encoding="utf-8").strip()
        if target.exists():
            return str(target)
    cands = sorted(ck.glob("*.pt"), key=lambda p: _checkpoint_sort_key(p.name))
    if not cands:
        return None
    return str(cands[-1])


def redact_config(value):
    """Remove credentials before persisting a resolved configuration."""
    if isinstance(value, dict):
        return {k: ("" if any(part in k.lower() for part in ("api_key", "password", "secret", "access_token")) or k.lower() == "token" else redact_config(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_config(v) for v in value]
    return value
