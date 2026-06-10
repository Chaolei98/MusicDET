from sonics.models.spectttra import SpecTTTra
from sonics.models.vit import ViT
from sonics.layers.feature import FeatureExtractor
from sonics.layers.augment import AugmentLayer
import torch.nn as nn
import torch.nn.functional as F
import timm


def use_global_pool(model_name):
    """
    Check if the model requires global pooling or not.
    """
    no_global_pool = ["timm"]
    return False if any(x in model_name for x in no_global_pool) else True

def get_embed_dim(model_name, encoder):
    """
    Get the embedding dimension of the encoder.
    """
    if "timm" in model_name:
        return encoder.head_hidden_size
    else:
        return encoder.embed_dim

def use_init_weights(model_name):
    """
    Check if the model requires initialization of weights or not.
    """
    has_init_weights = ["timm"]
    return False if any(x in model_name for x in has_init_weights) else True

class AudioClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model_name = cfg.model.name
        self.input_shape = cfg.model.input_shape
        self.num_classes = cfg.num_classes
        self.ft_extractor = FeatureExtractor(cfg)
        self.augment = AugmentLayer(cfg)
        self.encoder = self.get_encoder(cfg)
        self.embed_dim = get_embed_dim(self.model_name, self.encoder)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.use_init_weights = getattr(cfg.model, "use_init_weights", True)

        (
            self.initialize_weights()
            if self.use_init_weights and use_init_weights(self.model_name)
            else None
        )

    def get_encoder(self, cfg):
        if cfg.model.name == "SpecTTTra":
            model = SpecTTTra(
                input_spec_dim=cfg.model.input_shape[0],
                input_temp_dim=cfg.model.input_shape[1],
                embed_dim=cfg.model.embed_dim,
                t_clip=cfg.model.t_clip,
                f_clip=cfg.model.f_clip,
                num_heads=cfg.model.num_heads,
                num_layers=cfg.model.num_layers,
                pre_norm=cfg.model.pre_norm,
                pe_learnable=cfg.model.pe_learnable,
                pos_drop_rate=getattr(cfg.model, "pos_drop_rate", 0.0),
                attn_drop_rate=getattr(cfg.model, "attn_drop_rate", 0.0),
                proj_drop_rate=getattr(cfg.model, "proj_drop_rate", 0.0),
                mlp_ratio=getattr(cfg.model, "mlp_ratio", 4.0),
            )
        elif cfg.model.name == "ViT":
            model = ViT(
                image_size=cfg.model.input_shape,
                patch_size=cfg.model.patch_size,
                embed_dim=cfg.model.embed_dim,
                num_heads=cfg.model.num_heads,
                num_layers=cfg.model.num_layers,
                pe_learnable=cfg.model.pe_learnable,
                patch_norm=getattr(cfg.model, "patch_norm", False),
                pos_drop_rate=getattr(cfg.model, "pos_drop_rate", 0.0),
                attn_drop_rate=getattr(cfg.model, "attn_drop_rate", 0.0),
                proj_drop_rate=getattr(cfg.model, "proj_drop_rate", 0.0),
                mlp_ratio=getattr(cfg.model, "mlp_ratio", 4.0),
            )
        elif "timm" in cfg.model.name:
            model_name = cfg.model.name.replace("timm-", "")
            model = timm.create_model(
                model_name,
                pretrained=cfg.model.pretrained,
                in_chans=1,
                num_classes=0,
            )
        else:
            raise ValueError(f"Model {cfg.model.name} not supported in V1.")
        return model

    def forward(self, audio, y=None):
        audio = audio.float()
        spec = self.ft_extractor(audio)
        if self.training:
            spec, y = self.augment(spec, y)
        spec = spec.unsqueeze(1)
        spec = F.interpolate(spec, size=tuple(self.input_shape), mode="bilinear")
        features = self.encoder(spec)
        embeds = features.mean(dim=1) if use_global_pool(self.model_name) else features
        preds = self.classifier(embeds)
        return embeds, preds

    def initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.startswith("classifier"):
                    nn.init.zeros_(module.weight)
                    nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, std=1e-6)
            elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif hasattr(module, "init_weights"):
                module.init_weights()
