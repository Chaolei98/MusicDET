import argparse
import os
from statistics import mean


import numpy as np


import eval_metrics as em


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=str, help="Path to the model output folder")
    return parser.parse_args()

def compute_eer(cm_score_file):
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    if cm_data.size == 0:
        print(f"{cm_score_file}\n   skipped        = empty score file")
        return None
    if cm_data.ndim == 1:
        cm_data = np.expand_dims(cm_data, axis=0)
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 1].astype(float)
    other_scores = -cm_scores

    eer = em.compute_eer(cm_scores[cm_keys == "real"], cm_scores[cm_keys == "fake"])[0]
    other_eer = em.compute_eer(other_scores[cm_keys == "real"], other_scores[cm_keys == "fake"])[0]

    best_eer = min(eer, other_eer)
    print(cm_score_file)
    print(f"   EER            = {best_eer * 100:8.2f} % (Equal error rate for countermeasure)")
    return best_eer

def traverse_and_compute_eer(root_dir):
    result_dir = os.path.join(root_dir, "result")
    all_eers = []
    for root, _, files in os.walk(result_dir):
        for file in sorted(files):
            if file.endswith(".txt"):
                eer = compute_eer(os.path.join(root, file))
                if eer is not None:
                    all_eers.append(eer * 100)

    if not all_eers:
        print("No .txt files found.")
        return

    print(f"Average EER is {mean(all_eers):.2f}%")

if __name__ == "__main__":
    args = init()
    traverse_and_compute_eer(args.p)
