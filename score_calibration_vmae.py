import numpy as np
import argparse
import os
from tqdm import tqdm
import faiss
import glob


def compute_calibration_parameters(args, root="data/"):
    train_clip_lengths = np.load(os.path.join(root, args.dataset_name, "train_clip_lengths.npy"))
    vmae_root = "../VideoMAEv2/extracted_features/continuous"
    train_deep_features_files = glob.glob(f"{vmae_root}/{args.dataset_name}/training/*.npy")
    train_deep_features_files.sort()
    train_deep_features = np.concatenate([np.load(f) for f in
                                          tqdm(train_deep_features_files, desc="Loading training deep features")],
                                         axis=0)

    all_ranges = np.arange(0, len(train_deep_features))
    features_scores = []

    print(train_clip_lengths)
    print(len(train_deep_features))

    prev = 0
    for idx in tqdm(range(len(train_clip_lengths)), desc="Computing calibration parameters"):
        cur = train_clip_lengths[idx]
        print(prev, cur)
        print(prev - 15 * idx, cur - 15 * (idx + 1))

        cur_video_range = np.arange(prev - 15 * idx, cur - 15 * (idx + 1))
        complement_indices = np.setdiff1d(all_ranges, cur_video_range)

        rest_deep_features = train_deep_features[complement_indices]
        cur_deep_features = train_deep_features[cur_video_range]

        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(rest_deep_features.shape[1])
        index_deep_features = faiss.index_cpu_to_gpu(res, 0, index)
        index_deep_features.add(rest_deep_features.astype(np.float32))

        D, I = index_deep_features.search(cur_deep_features.astype(np.float32), 1)
        score_deep_features = np.mean(D, axis=1)
        features_scores.append(score_deep_features)

        prev = cur

    features_scores = np.concatenate(features_scores, 0)

    np.save(f"{vmae_root}/{args.dataset_name}/train_deep_features_scores.npy", features_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="shanghaitech", help="dataset name")
    args = parser.parse_args()
    compute_calibration_parameters(args)
