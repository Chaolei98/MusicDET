import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


class XLSR(nn.Module):
    """Wrapper around wav2vec 2.0 XLS-R feature extraction."""

    def __init__(self, model_dir, device="cuda", sampling_rate=16000, freeze=True, visual=False):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, do_normalize=False)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)
        self.freeze = freeze
        self.visual = visual

        self.model.config.output_hidden_states = True
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

    def _prepare_inputs(self, audio_data: torch.Tensor) -> torch.Tensor:
        if audio_data.dim() == 3 and audio_data.size(-1) == 1:
            audio_data = audio_data.squeeze(-1)
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        return self.processor(
            audio_data.detach().cpu().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_values.to(self.device)

    def forward(self, audio_data):
        feat = self._prepare_inputs(audio_data)

        if self.visual:
            outputs = self.model(feat, output_attentions=True)
            return outputs.last_hidden_state, outputs.attentions

        if self.freeze:
            with torch.no_grad():
                outputs = self.model(feat)
        else:
            outputs = self.model(feat)
        return outputs.last_hidden_state

    def extract_features(self, audio_data, layers=None, aggregate="list"):
        """
        Extract hidden states from selected transformer layers.

        Args:
            audio_data: Input waveform tensor.
            layers: One-based transformer layer indices. `0` selects the
                projected input state. If omitted, the final layer is used.
            aggregate: One of `list`, `concat`, `mean`, `sum`, or `stack`.

        Returns:
            A list of tensors when `aggregate="list"`, otherwise a single tensor.
        """
        feat = self._prepare_inputs(audio_data)

        if self.freeze:
            with torch.no_grad():
                outputs = self.model(feat, output_hidden_states=True)
        else:
            outputs = self.model(feat, output_hidden_states=True)

        all_hs = outputs.hidden_states
        num_layers = self.model.config.num_hidden_layers

        if not layers:
            selected = [all_hs[-1]]
        else:
            selected = []
            for layer_idx in layers:
                if layer_idx == 0:
                    selected.append(all_hs[0])
                    continue
                if not (1 <= layer_idx <= num_layers):
                    raise ValueError(f"Layer {layer_idx} is out of range. Expected [1, {num_layers}] or 0.")
                selected.append(all_hs[layer_idx])

        if aggregate == "list":
            return selected
        if aggregate == "concat":
            return torch.cat(selected, dim=-1)
        if aggregate == "mean":
            return torch.stack(selected, dim=0).mean(dim=0)
        if aggregate == "sum":
            return torch.stack(selected, dim=0).sum(dim=0)
        if aggregate == "stack":
            return torch.stack(selected, dim=0)
        raise ValueError(f"Unsupported aggregate mode: {aggregate}")


class MERT(nn.Module):
    """Wrapper around MERT feature extraction."""

    def __init__(self, model_dir, device="cuda", sampling_rate=16000, freeze=True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = AutoFeatureExtractor.from_pretrained(
            model_dir,
            sampling_rate=sampling_rate,
            do_normalize=False,
        )
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
        self.freeze = freeze

        self.model.config.output_hidden_states = True
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()

    def _prepare_inputs(self, audio_data: torch.Tensor) -> torch.Tensor:
        if audio_data.dim() == 3 and audio_data.size(-1) == 1:
            audio_data = audio_data.squeeze(-1)
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        return self.processor(
            audio_data.detach().cpu().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).input_values.to(self.device)

    def forward(self, audio_data):
        """Return the final hidden state with shape `[B, T, C]`."""
        inputs = self._prepare_inputs(audio_data)
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
        else:
            outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
        return outputs.last_hidden_state

    def extract_features(self, audio_data, layers=None, aggregate="list"):
        """Extract features from selected hidden layers."""
        inputs = self._prepare_inputs(audio_data)

        if self.freeze:
            with torch.no_grad():
                outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
        else:
            outputs = self.model(inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states
        if layers is None:
            return hidden_states[-1]

        selected = []
        num_states = len(hidden_states)
        for layer_idx in layers:
            if layer_idx < 0:
                layer_idx = num_states + layer_idx
            layer_idx = max(0, min(layer_idx, num_states - 1))
            selected.append(hidden_states[layer_idx])

        if aggregate == "stack":
            return torch.stack(selected, dim=1)
        if aggregate == "cat":
            return torch.cat(selected, dim=-1)
        return selected
