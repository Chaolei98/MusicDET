import json
import os
import random
import shutil
from collections import defaultdict


import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from tqdm import tqdm


import config
import eval_metrics as em
from dataset import FakeMusicCapsDataset, SonicsDataset, load_sonics_dataframe
from model_factory import build_model, is_nf_model


torch.set_default_dtype(torch.float32)
torch.multiprocessing.set_start_method("spawn", force=True)
torch.set_num_threads(5)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_params():
    parser = config.init_params()
    args = parser.parse_args()
    setup_seed(args.seed)

    if not args.continue_training:
        if os.path.exists(args.out_fold):
            shutil.rmtree(args.out_fold)
        os.makedirs(os.path.join(args.out_fold, "checkpoint"), exist_ok=True)

        with open(os.path.join(args.out_fold, "args.json"), "w", encoding="utf-8") as file:
            json.dump(vars(args), file, indent=4)
        with open(os.path.join(args.out_fold, "train_loss.log"), "w", encoding="utf-8") as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, "dev_loss.log"), "w", encoding="utf-8") as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args

def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def compute_loss(nll, labels=None):
    if labels is None:
        loss = torch.mean(nll)
    else:
        unique_classes, counts = labels.unique(return_counts=True)
        total = counts.sum().float()
        freqs = counts.float() / total
        weights = (1.0 / freqs)
        weights = weights / weights.sum() * len(unique_classes)

        class_weights = torch.ones(int(labels.max()) + 1, device=nll.device)
        class_weights[unique_classes] = weights.to(nll.device)
        loss = torch.mean(nll * class_weights[labels])
    return loss

def build_dataloaders(args):
    if args.task == "fakemusiccaps":
        train_set = FakeMusicCapsDataset(
            args.fakemusiccaps_audio,
            args.fakemusiccaps_train_label,
            audio_length=args.audio_len,
            only_real=args.only_real,
            random_sampling=True,
        )
        dev_set = FakeMusicCapsDataset(
            args.fakemusiccaps_audio,
            args.fakemusiccaps_dev_label,
            audio_length=args.audio_len,
            only_real=False,
            random_sampling=False,
        )
    elif args.task == "sonics":
        train_df = load_sonics_dataframe(args.train_dataframe, only_real=args.only_real)
        valid_df = load_sonics_dataframe(args.valid_dataframe, only_real=False)
        train_set = SonicsDataset(
            train_df.filepath.tolist(),
            train_df.target.tolist(),
            max_len=args.audio_len,
            random_sampling=True,
            train=True,
        )
        dev_set = SonicsDataset(
            valid_df.filepath.tolist(),
            valid_df.target.tolist(),
            max_len=args.audio_len,
            random_sampling=False,
            train=False,
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    if len(train_set) == 0 or len(dev_set) == 0:
        raise ValueError("Dataset is empty. Please check the configured paths.")

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(train_set))),
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        shuffle=False,
    )
    return train_loader, dev_loader

def train(args):
    model = build_model(args)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005,
    )

    train_loader, dev_loader = build_dataloaders(args)
    prev_loss = float("inf")
    criterion = torch.nn.CrossEntropyLoss()

    epoch_bar = tqdm(range(args.epochs), desc="Epochs", position=0)
    for epoch_num in epoch_bar:
        adjust_learning_rate(args, optimizer, epoch_num)
        model.train()
        train_loss_log = defaultdict(list)

        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Train {epoch_num + 1}/{args.epochs}",
            position=1,
            leave=False,
        )
        for step, batch in train_bar:
            waveform, _, labels = batch
            waveform = waveform.to(args.device)
            labels = labels.to(args.device)

            if is_nf_model(args.model):
                _, nll = model(waveform, labels)
                if nll is None:
                    continue
                loss = compute_loss(nll, labels)
            else:
                _, logits = model(waveform)
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

            train_loss_log["base_loss"].append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            with open(os.path.join(args.out_fold, "train_loss.log"), "a", encoding="utf-8") as file:
                file.write(f"{epoch_num}\t{step}\t{loss.item()}\n")

        train_loss = float(np.nanmean(train_loss_log["base_loss"])) if train_loss_log["base_loss"] else float("nan")
        print(f"Epoch {epoch_num + 1}/{args.epochs} | Train | loss={train_loss:.4f}")

        model.eval()
        dev_scores = []
        dev_labels = []
        dev_losses = []
        val_score_path = os.path.join(args.out_fold, "val_scores.txt")

        with torch.no_grad(), open(val_score_path, "w", encoding="utf-8") as score_file:
            dev_bar = tqdm(
                dev_loader,
                total=len(dev_loader),
                desc=f"Valid {epoch_num + 1}/{args.epochs}",
                position=1,
                leave=False,
            )
            for waveform, audio_fn, labels in dev_bar:
                waveform = waveform.to(args.device)
                labels = labels.to(args.device)
                if is_nf_model(args.model):
                    true_label = torch.zeros(len(labels), dtype=torch.long, device=args.device)
                    _, nll = model(waveform, true_label)
                    loss = compute_loss(nll)
                    score = -nll.detach()
                else:
                    feats, logits = model(waveform)
                    loss = criterion(logits, labels)
                    score = torch.softmax(logits, dim=1)[:, 0].detach()

                for fn, sc, lb in zip(audio_fn, score, labels):
                    clean_fn = os.path.splitext(os.path.basename(fn))[0]
                    label_str = "fake" if lb.item() == 1 else "real"
                    score_file.write(f"{clean_fn} {float(sc)} {label_str}\n")

                dev_scores.append(score.cpu())
                dev_labels.append(labels.cpu())
                dev_losses.append(loss.item())
                dev_bar.set_postfix(loss=f"{loss.item():.4f}")

        scores = torch.cat(dev_scores, 0).numpy()
        labels = torch.cat(dev_labels, 0).numpy()
        val_loss = float(np.nanmean(dev_losses))
        val_eer = min(
            em.compute_eer(scores[labels == 0], scores[labels == 1])[0],
            em.compute_eer(-scores[labels == 0], -scores[labels == 1])[0],
        )

        with open(os.path.join(args.out_fold, "dev_loss.log"), "a", encoding="utf-8") as file:
            file.write(f"{epoch_num}\t{val_loss}\t{val_eer}\n")

        print(f"Epoch {epoch_num + 1}/{args.epochs} | Valid | loss={val_loss:.4f} | eer={val_eer:.4f}")

        if val_loss < prev_loss:
            torch.save(model.state_dict(), os.path.join(args.out_fold, "anti-spoofing_feat_model.pt"))
            prev_loss = val_loss

        epoch_bar.set_postfix(val_loss=f"{val_loss:.4f}", val_eer=f"{val_eer:.4f}")

    return model

if __name__ == "__main__":
    arguments = init_params()
    train(arguments)
