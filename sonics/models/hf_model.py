import os
import json
import torch
import torch.nn as nn
from .model import AudioClassifier
from ..utils.config import dict2cfg, cfg2dict
from huggingface_hub import HfApi, create_repo, hf_hub_download


class HFAudioClassifier(AudioClassifier):
    """Hugging Face compatible AudioClassifier model"""

    def __init__(self, config):
        if isinstance(config, dict):
            self.config = dict2cfg(config)
        super().__init__(self.config)

    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None, map_location="cpu", strict=False):
        is_local = os.path.exists(model_id)

        if is_local:
            config_file = os.path.join(model_id, "config.json")
            model_file = os.path.join(model_id, "pytorch_model.bin")
        else:
            config_file = hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=cache_dir)
            model_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin", cache_dir=cache_dir)

        config = None
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

        model = cls(config)

        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=torch.device(map_location))
            model.load_state_dict(state_dict, strict=strict)
            model.eval()
        else:
            raise FileNotFoundError(f"Model weights not found at {model_file}")

        return model

    def push_to_hub(self, repo_id, token=None, commit_message=None, private=False):
        """Push model and config to Hugging Face Hub.

        Args:
            repo_id (str): Repository ID on HuggingFace Hub (e.g., 'username/model-name')
            token (str, optional): HuggingFace token. If None, will use token from ~/.huggingface/token
            commit_message (str, optional): Commit message for the push
            private (bool, optional): Whether to make the repository private
        """

        api = HfApi()
        try:
            create_repo(repo_id, private=private, token=token, exist_ok=True)
        except Exception as e:
            print(f"Repository creation failed: {e}")
            return

        config = cfg2dict(self.config)
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        torch.save(self.cpu().state_dict(), "pytorch_model.bin")
        self.to(self.device if hasattr(self, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')

        files_to_push = ["config.json", "pytorch_model.bin"]
        for file in files_to_push:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or f"Upload {file}"
            )
            os.remove(file)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model weights and configuration to a directory.

        Args:
            save_directory (str): Directory to save files in
            **kwargs: Additional arguments passed to save functions
        """
        os.makedirs(save_directory, exist_ok=True)

        config = cfg2dict(self.config)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.cpu().state_dict(), model_file)
        self.to(self.device if hasattr(self, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
