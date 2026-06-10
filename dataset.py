import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
SONICS_ROOT = DATASETS_ROOT / "sonics"
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "numba"))


import librosa
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


TARGET_SR = 16000
TARGET_LEN = 64600
EPS = 1e-8


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_sonics_filepath(path):
    path = str(path)
    if "real_songs_wav/" in path:
        return str(SONICS_ROOT / "real_songs_wav" / Path(path).name)
    if "fake_songs_wav/" in path:
        return str(SONICS_ROOT / "fake_songs_wav" / Path(path).name)
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(resolve_project_path(candidate))


def load_audio_mono_resample(path, target_sr=TARGET_SR, duration=None):
    audio, _ = librosa.load(path, sr=target_sr, mono=True, duration=duration)
    return torch.from_numpy(audio).float()

def crop_or_pad_1d(audio, target_len=TARGET_LEN, random_sampling=True):
    assert audio.dim() == 1
    length = audio.shape[0]

    if length == target_len:
        return audio

    if length > target_len:
        diff = length - target_len
        start = random.randint(0, diff) if random_sampling else diff // 2
        return audio[start:start + target_len]

    diff = target_len - length
    pad_left = random.randint(0, diff) if random_sampling else diff // 2
    pad_right = diff - pad_left
    return F.pad(audio, (pad_left, pad_right), mode="constant", value=0.0)

def rms_normalize(audio, eps=EPS):
    rms = torch.sqrt(torch.mean(audio ** 2) + eps)
    return audio / rms

def preprocess_waveform_from_path(path, target_sr=TARGET_SR, target_len=TARGET_LEN, random_sampling=True):
    audio = load_audio_mono_resample(path, target_sr=target_sr)
    audio = crop_or_pad_1d(audio, target_len=target_len, random_sampling=random_sampling)
    return rms_normalize(audio)

class FakeMusicCapsDataset(Dataset):
    def __init__(
        self,
        path_to_audio,
        path_to_protocol,
        audio_length=TARGET_LEN,
        only_real=False,
        random_sampling=True,
    ):
        super().__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label_map = {"fake": 1, "real": 0}
        self.random_sampling = random_sampling
        self.is_train = random_sampling

        with open(self.path_to_protocol, "r", encoding="utf-8") as file:
            audio_info = [line.strip().split() for line in file if line.strip()]

        if only_real:
            audio_info = [info for info in audio_info if info[1] == "real"]

        self.all_files = audio_info

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, label_str = self.all_files[idx][:2]
        filepath = resolve_project_path(Path(self.path_to_audio) / filename)
        waveform = preprocess_waveform_from_path(
            str(filepath),
            target_sr=TARGET_SR,
            target_len=self.audio_length,
            random_sampling=self.random_sampling,
        )
        return waveform, filename, self.label_map[label_str]

    @staticmethod
    def _normalize_to_base_id(filename):
        file_id = os.path.splitext(os.path.basename(filename))[0]
        for prefix in ("TTM01_", "TTM02_", "TTM03_", "TTM04_", "TTM05_"):
            if file_id.startswith(prefix):
                return file_id[len(prefix):]
        return file_id

class SonicsDataset(Dataset):
    def __init__(
        self,
        filepaths,
        labels,
        max_len=TARGET_LEN,
        random_sampling=False,
        train=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.labels = labels
        self.max_len = max_len
        self.train = train
        self.random_sampling = random_sampling if train else False

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio = preprocess_waveform_from_path(
            self.filepaths[idx],
            target_sr=TARGET_SR,
            target_len=self.max_len,
            random_sampling=self.random_sampling,
        )
        target = int(self.labels[idx])
        audio_fn = self.filepaths[idx]
        return audio, audio_fn, target

def load_sonics_dataframe(csv_path, only_real=False):
    dataframe = pd.read_csv(csv_path)
    if only_real:
        dataframe = dataframe.loc[dataframe["target"] == 0].copy()
    if "filepath" in dataframe.columns:
        dataframe["filepath"] = dataframe["filepath"].map(normalize_sonics_filepath)
    dataframe = dataframe.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return dataframe
