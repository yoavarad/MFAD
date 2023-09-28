import argparse
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from evaluate import gaussian_video
from video_dataset import VideoDatasetWithFlows


def random_frames(X, y, test_clip_lengths, p=0.02, sigma=3):
    aucs = []
    for _ in range(100):
        while True:
            train_idx = random.sample(list(range(len(X))), int(p * len(X)))
            test_idx = np.setdiff1d(list(range(len(X))), train_idx)

            if len(np.unique(y[train_idx])) > 1:
                break

        model = LogisticRegression()
        model.fit(X[train_idx], y[train_idx])

        final_scores = model.predict_proba(X)[:, 1]

        final_scores = gaussian_video(final_scores, test_clip_lengths, sigma=sigma)

        auc = roc_auc_score(y[test_idx], final_scores[test_idx])
        aucs.append(auc)

    print(f"Random Frame {100 * p}% LR AUC: {(100 * np.mean(aucs)):.1f}%±{(100 * np.std(aucs)):.1f}%")


def calc_scores(dataset_name):
    vid_names = os.listdir(f"data/{dataset_name}/testing/frames")
    vid_names.sort()

    test_clip_lengths = np.load(f"data/{dataset_name}/test_clip_lengths.npy")

    test_dataset = VideoDatasetWithFlows(dataset_name=dataset_name, root='data/', train=False)
    y = test_dataset.all_gt

    P, V, IE, VE = np.load(f"final_scores/{dataset_name}.npy")
    max_feature = np.max(np.stack((P, V, IE, VE)), axis=0)

    X = np.stack((P, V, IE, VE)).T
    X_max = np.stack((*X.T, np.max(X, axis=1))).T

    sigma = (3 if dataset_name in ['ped2', 'avenue'] else 7)

    final_scores = gaussian_video(P + V + IE + VE + max_feature, test_clip_lengths, sigma=sigma)
    auc = roc_auc_score(y, final_scores)
    print(f'P+V+IE+VE+max {auc * 100:.1f}%')

    p = 0.02
    random_frames(X_max, y, test_clip_lengths, p=p, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ped2', help='dataset name')
    args = parser.parse_args()

    calc_scores(args.dataset_name)
