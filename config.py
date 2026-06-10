import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
PRETRAINED_ROOT = PROJECT_ROOT / "pretrained"
FAKEMUSICCAPS_LABEL_ROOT = PROJECT_ROOT / "label" / "fakemusiccaps" / "openset"
SONICS_LABEL_ROOT = PROJECT_ROOT / "label" / "sonics"


def init_params():
    parser = argparse.ArgumentParser(description="MusicDET configuration")

    parser.add_argument("--seed", type=int, default=688, help="Random seed")
    parser.add_argument(
        "--task",
        "--train_task",
        dest="task",
        type=str,
        default="fakemusiccaps",
        choices=["fakemusiccaps", "sonics"],
        help="Training or evaluation task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="spec-nf",
        choices=[
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
        ],
        help="Model architecture",
    )

    parser.add_argument(
        "--fakemusiccaps_audio",
        type=str,
        default=str(DATASETS_ROOT / "fakemusiccaps" / "all_audio_wav"),
        help="Path to FakeMusicCaps audio directory",
    )
    parser.add_argument(
        "--fakemusiccaps_train_label",
        type=str,
        default=str(FAKEMUSICCAPS_LABEL_ROOT / "TTM01_train.txt"),
        help="Path to the FakeMusicCaps training protocol",
    )
    parser.add_argument(
        "--fakemusiccaps_dev_label",
        type=str,
        default=str(FAKEMUSICCAPS_LABEL_ROOT / "TTM01_dev.txt"),
        help="Path to the FakeMusicCaps validation protocol",
    )
    parser.add_argument(
        "--fakemusiccaps_eval_labels_dir",
        type=str,
        default=str(FAKEMUSICCAPS_LABEL_ROOT),
        help="Directory containing TTM01-05 eval protocol files",
    )

    parser.add_argument(
        "--train_dataframe",
        type=str,
        default=str(SONICS_LABEL_ROOT / "train" / "real_suno-v3.5.csv"),
        help="Path to the SONICS training csv",
    )
    parser.add_argument(
        "--valid_dataframe",
        type=str,
        default=str(SONICS_LABEL_ROOT / "dev" / "real_suno-v3.5.csv"),
        help="Path to the SONICS validation csv",
    )
    parser.add_argument(
        "--test_dataframe",
        type=str,
        default=str(SONICS_LABEL_ROOT / "test" / "real_suno3.5.csv"),
        help="Path to the SONICS test csv",
    )
    parser.add_argument(
        "--sonics_eval_csvs",
        nargs="*",
        default=[
            str(SONICS_LABEL_ROOT / "test" / "real_suno2.0.csv"),
            str(SONICS_LABEL_ROOT / "test" / "real_suno3.0.csv"),
            str(SONICS_LABEL_ROOT / "test" / "real_suno3.5.csv"),
            str(SONICS_LABEL_ROOT / "test" / "real_udio30s.csv"),
            str(SONICS_LABEL_ROOT / "test" / "real_udio120.csv"),
        ],
        help="List of SONICS evaluation csv files",
    )

    parser.add_argument("-o", "--out_fold", type=str, default="./output", help="Output folder")
    parser.add_argument("--audio_len", type=int, default=64600, help="Waveform length")
    parser.add_argument("--epochs", "--num_epochs", dest="epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning rate decay factor")
    parser.add_argument("--interval", type=int, default=10, help="Learning rate decay interval")
    parser.add_argument("--beta_1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta_2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--continue_training", action="store_true", help="Reuse existing output folder")
    parser.add_argument("--only_real", action="store_true", help="Train with real samples only")

    parser.add_argument("--K", type=int, default=2, help="Flow depth parameter")
    parser.add_argument("--L", type=int, default=1, help="Reserved flow parameter")
    parser.add_argument("--R", type=float, default=5.0, help="Flow radius parameter")
    parser.add_argument("--xlsr", type=str, default=str(PRETRAINED_ROOT / "xlsr"))
    parser.add_argument("--mert", type=str, default=str(PRETRAINED_ROOT / "mert"))

    parser.add_argument(
        "--jazz_txt_path",
        type=str,
        default=str(DATASETS_ROOT / "metadata" / "piano_music_names.txt"),
        help="Optional id list used by the original MusicDET split logic",
    )

    return parser
