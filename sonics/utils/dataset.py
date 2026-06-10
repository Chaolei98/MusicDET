from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import librosa


class AudioDataset(Dataset):
    def __init__(
        self,
        filepaths,
        labels,
        skip_times=None,
        num_classes=1,
        normalize="std",
        max_len=32000,
        random_sampling=True,
        train=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filepaths = filepaths
        self.labels = labels
        self.skip_times = skip_times
        self.num_classes = num_classes
        self.random_sampling = random_sampling
        self.normalize = normalize
        self.max_len = max_len
        self.train = train
        if not self.train:
            assert (
                not self.random_sampling
            ), "Ensure random_sampling is disabled for val"

    def __len__(self):
        return len(self.filepaths)

    def crop_or_pad(self, audio, max_len, random_sampling=True):
        audio_len = audio.shape[0]
        if random_sampling:
            diff_len = abs(max_len - audio_len)
            if audio_len < max_len:
                pad1 = np.random.randint(0, diff_len)
                pad2 = diff_len - pad1
                audio = np.pad(audio, (pad1, pad2), mode="constant")
            elif audio_len > max_len:
                idx = np.random.randint(0, diff_len)
                audio = audio[idx : (idx + max_len)]
        else:
            if audio_len < max_len:
                audio = np.pad(audio, (0, max_len - audio_len), mode="constant")
            elif audio_len > max_len:

                idx = int((audio_len - max_len) / 4 * 3)
                audio = audio[idx : (idx + max_len)]
        return audio

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.filepaths[idx], sr=None, duration=10)
        target = np.array([self.labels[idx]])

        if self.skip_times is not None:
            skip_time = self.skip_times[idx]
            audio = audio[int(skip_time*sr):]

        audio = self.crop_or_pad(audio, self.max_len, self.random_sampling)

        if self.normalize == "std":
            audio /= np.maximum(np.std(audio), 1e-6)
        elif self.normalize == "minmax":
            audio -= np.min(audio)
            audio /= np.maximum(np.max(audio), 1e-6)

        audio = torch.from_numpy(audio).float()
        target = torch.from_numpy(target).float().squeeze()
        return {
            "audio": audio,
            "target": target,
        }

def get_dataloader(
    filepaths,
    labels,
    skip_times=None,
    batch_size=8,
    num_classes=1,
    max_len=32000,
    random_sampling=True,
    normalize="std",
    train=False,
    pin_memory=True,
    worker_init_fn=None,
    collate_fn=None,
    num_workers=0,
    distributed=False,
):
    dataset = AudioDataset(
        filepaths,
        labels,
        skip_times=skip_times,
        num_classes=num_classes,
        max_len=max_len,
        random_sampling=random_sampling,
        normalize=normalize,
        train=train,
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=train, drop_last=not train
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None) and train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return dataloader
