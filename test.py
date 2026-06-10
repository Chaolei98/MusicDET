import argparse
import json
import os
from pathlib import Path


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataset import FakeMusicCapsDataset, SonicsDataset, load_sonics_dataframe
from model_factory import build_model as build_architecture, is_nf_model


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "label" / "fakemusiccaps" / "openset"
DEFAULT_SONICS_EVAL = [
    PROJECT_ROOT / "label" / "sonics" / "test" / "real_suno2.0.csv",
    PROJECT_ROOT / "label" / "sonics" / "test" / "real_suno3.0.csv",
    PROJECT_ROOT / "label" / "sonics" / "test" / "real_suno3.5.csv",
    PROJECT_ROOT / "label" / "sonics" / "test" / "real_udio30s.csv",
    PROJECT_ROOT / "label" / "sonics" / "test" / "real_udio120.csv",
]


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model folder")
    parser.add_argument("--task", type=str, default="fakemusiccaps", choices=["fakemusiccaps", "sonics"], help="Task type")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--fakemusiccaps_audio",
        type=str,
        default=str(DATASETS_ROOT / "fakemusiccaps" / "all_audio_wav"),
        help="Path to FakeMusicCaps audio directory",
    )
    parser.add_argument(
        "--fakemusiccaps_eval_labels_dir",
        type=str,
        default=str(DEFAULT_EVAL_DIR),
        help="Directory containing TTM01-05 eval protocol files",
    )
    parser.add_argument(
        "--sonics_eval_csvs",
        nargs="*",
        default=[str(path) for path in DEFAULT_SONICS_EVAL],
        help="List of SONICS evaluation csv files",
    )

    temp_args, _ = parser.parse_known_args()
    with open(os.path.join(temp_args.model_path, "args.json"), "r", encoding="utf-8") as file:
        json_args = json.load(file)

    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = json_args.get("batch_size", 16)

    args.model = json_args.get("model", "spec-nf")
    args.audio_len = json_args.get("audio_len", 64600)
    args.K = json_args.get("K", 2)
    args.L = json_args.get("L", 1)
    args.R = json_args.get("R", 5.0)
    args.xlsr = json_args.get("xlsr", str(PRETRAINED_ROOT / "xlsr"))
    args.mert = json_args.get("mert", str(PRETRAINED_ROOT / "mert"))
    args.task = args.task or json_args.get("task", "fakemusiccaps")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args

def build_model(args):
    model = build_architecture(args)
    checkpoint = torch.load(
        os.path.join(args.model_path, "anti-spoofing_feat_model.pt"),
        map_location=args.device,
    )
    if all(key.startswith("module.") for key in checkpoint.keys()):
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def compute_scores(model, waveform, device, model_name):
    waveform = waveform.to(device)
    if is_nf_model(model_name):
        true_label = torch.zeros(waveform.shape[0], dtype=torch.long, device=device)
        _, nll = model(waveform, true_label)
        return -nll.detach().cpu()
    _, logits = model(waveform)
    return torch.softmax(logits, dim=1)[:, 0].detach().cpu()

def test_on_fakemusiccaps(model, args):
    result_dir = os.path.join(args.model_path, "result")
    os.makedirs(result_dir, exist_ok=True)

    eval_root = Path(args.fakemusiccaps_eval_labels_dir)
    eval_labels = [eval_root / f"TTM0{i}_eval.txt" for i in range(1, 6)]

    with torch.no_grad():
        for eval_label in eval_labels:
            file_path = os.path.join(result_dir, eval_label.name)
            test_set = FakeMusicCapsDataset(
                args.fakemusiccaps_audio,
                str(eval_label),
                audio_length=args.audio_len,
                random_sampling=False,
            )
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

            with open(file_path, "w", encoding="utf-8") as score_file:
                for waveform, filename, labels in tqdm(test_loader):
                    scores = compute_scores(model, waveform, args.device, args.model)

                    for fn, score, label in zip(filename, scores, labels):
                        audio_fn = fn.strip().split(".")[0]
                        label_str = "fake" if int(label) == 1 else "real"
                        score_file.write(f"{audio_fn} {float(score)} {label_str}\n")

def test_on_sonics(model, args):
    result_dir = os.path.join(args.model_path, "result")
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for csv_path in args.sonics_eval_csvs:
            csv_path = Path(csv_path)
            file_path = os.path.join(result_dir, csv_path.stem + ".txt")

            test_df = load_sonics_dataframe(str(csv_path), only_real=False)
            if "target" not in test_df.columns or "filepath" not in test_df.columns:
                raise ValueError(f"SONICS csv must contain filepath and target columns: {csv_path}")

            sonics_testset = SonicsDataset(
                test_df.filepath.tolist(),
                test_df.target.tolist(),
                max_len=args.audio_len,
                random_sampling=False,
                train=False,
            )
            test_loader = DataLoader(sonics_testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            with open(file_path, "w", encoding="utf-8") as score_file:
                for waveform, filename, labels in tqdm(test_loader):
                    scores = compute_scores(model, waveform, args.device, args.model)
                    for fn, score, label in zip(filename, scores, labels):
                        audio_fn = os.path.basename(fn)
                        label_str = "fake" if int(label) == 1 else "real"
                        score_file.write(f"{audio_fn} {float(score)} {label_str}\n")

if __name__ == "__main__":
    arguments = init()
    model = build_model(arguments)
    if arguments.task == "fakemusiccaps":
        test_on_fakemusiccaps(model, arguments)
    elif arguments.task == "sonics":
        test_on_sonics(model, arguments)
    else:
        raise ValueError(f"Unsupported task: {arguments.task}")
