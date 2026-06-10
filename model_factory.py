from pathlib import Path


import yaml


try:
    from model import MERTAASIST, SpecNF, XLSRAASIST
    from sonics.models.model import AudioClassifier
    from sonics.utils.config import dict2cfg
except ModuleNotFoundError:
    from .model import MERTAASIST, SpecNF, XLSRAASIST
    from .sonics.models.model import AudioClassifier
    from .sonics.utils.config import dict2cfg

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PROJECT_ROOT / "configs"


SUPPORTED_MODELS = [
    "spec-nf",
    "Spec-nf",
    "fr-w2v2aasist",
    "fr-mertaasist",
    "ft-w2v2aasist",
    "ft-mertaasist",
    "SpecTTTra-alpha",
    "SpecTTTra-beta",
    "SpecTTTra-gamma",
    "ViT",
    "ConvNeXt",
]


def is_nf_model(model_name):
    return model_name in {"spec-nf", "Spec-nf"}

def _load_cfg(config_name):
    with open(CONFIG_ROOT / config_name, "r", encoding="utf-8") as file:
        cfg_dict = yaml.safe_load(file.read())
    return dict2cfg(cfg_dict)

def build_model(args):
    model_name = args.model

    if model_name in {"spec-nf", "Spec-nf"}:
        return SpecNF(K=args.K, L=args.L, R=args.R).to(args.device)
    if model_name == "fr-w2v2aasist":
        return XLSRAASIST(model_dir=args.xlsr, freeze=True).to(args.device)
    if model_name == "ft-w2v2aasist":
        return XLSRAASIST(model_dir=args.xlsr, freeze=False).to(args.device)
    if model_name == "fr-mertaasist":
        return MERTAASIST(model_dir=args.mert, freeze=True).to(args.device)
    if model_name == "ft-mertaasist":
        return MERTAASIST(model_dir=args.mert, freeze=False).to(args.device)
    if model_name == "SpecTTTra-alpha":
        return AudioClassifier(_load_cfg("spectttra_f1t3-5s.yaml")).to(args.device)
    if model_name == "SpecTTTra-beta":
        return AudioClassifier(_load_cfg("spectttra_f3t5-5s.yaml")).to(args.device)
    if model_name == "SpecTTTra-gamma":
        return AudioClassifier(_load_cfg("spectttra_f5t7-5s.yaml")).to(args.device)
    if model_name == "ViT":
        return AudioClassifier(_load_cfg("vit-5s.yaml")).to(args.device)
    if model_name == "ConvNeXt":
        return AudioClassifier(_load_cfg("convnext-5s.yaml")).to(args.device)

    raise ValueError(f"Unsupported model: {model_name}")
