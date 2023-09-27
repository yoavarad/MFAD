import argparse
import os
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from matplotlib import patches
from evaluate import split_by_video, gaussian_video, first_and_last_seq, min_max_normalize
from video_dataset import VideoDatasetWithFlows


def plot_videos(relevant_videos, dataset_name):
    fig_size = (5, 2)
    font_size = 15

    vid_names = os.listdir(f"data/{dataset_name}/testing/frames")
    vid_names.sort()

    test_clip_lengths = np.load(f"data/{dataset_name}/test_clip_lengths.npy")

    test_dataset = VideoDatasetWithFlows(dataset_name=dataset_name, root='data/', train=False)
    y = test_dataset.all_gt

    X_non_overlapping = np.load(f"final_scores/{dataset_name}.npy").transpose()  # (num_frames, 4)
    # X_sliding_window = np.load(f"final_scores-2/{dataset_name}.npy").transpose()  # (num_frames, 4)
    X = X_non_overlapping

    while True:
        train_idx = random.sample(list(range(len(X))), len(X) // 20)
        test_idx = np.setdiff1d(list(range(len(X))), train_idx)

        if len(np.unique(y[test_idx])) > 1:
            break

    model = LogisticRegression()
    model.fit(X[train_idx], y[train_idx])

    my_final_scores = model.predict_proba(X)[:, 1]
    my_final_scores = gaussian_video(my_final_scores, test_clip_lengths, sigma=7)

    ai_final_scores = gaussian_video(np.sum(X[:, :3], axis=1), test_clip_lengths, sigma=7)

    video_labels = split_by_video(y, test_clip_lengths, vid_names)
    my_video_scores = split_by_video(my_final_scores, test_clip_lengths, vid_names)
    ai_video_scores = split_by_video(ai_final_scores, test_clip_lengths, vid_names)

    video_labels = {v: video_labels[v] for v in relevant_videos}
    video_scores = {v: (my_video_scores[v], ai_video_scores[v]) for v in relevant_videos}

    plt.rcParams.update({'font.size': font_size})

    plot_dir = 'plots-final'
    os.makedirs(plot_dir, exist_ok=True)
    for video_name, (my_score_list, ai_video_scores) in video_scores.items():
        for plot_type, score_list in [('my', my_score_list), ('ai', ai_video_scores)]:
            score_local_list = min_max_normalize(score_list)

            fig, ax = plt.subplots(figsize=fig_size, )

            boundaries = first_and_last_seq(video_labels[video_name])
            rects = []
            for s, e in boundaries:
                rects.append(patches.Rectangle((s / 25, 0), (e - s) / 25, 1, facecolor='red', alpha=0.2))

            for rect in rects:
                ax.add_patch(rect)

            ax.plot(np.linspace(0, len(score_local_list) / 25, num=len(score_local_list)), score_local_list, color='b')

            ax.set_ylim([0, 1])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Anomaly Score')

            plt.savefig(f'{plot_dir}/{plot_type}_{video_name}.png', bbox_inches='tight', transparent=False,
                        pad_inches=0.03)
            plt.close()


def random_frames(X, y, test_clip_lengths, dataset_name, p=0.05, sigma=3):
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


def random_videos(X, y, test_clip_lengths, vid_names):
    test_clip_lengths_single = np.diff([0] + test_clip_lengths.tolist())
    num_learned = max(1, len(test_clip_lengths) // 20)

    yes_gaus, no_gaus = [], []

    X_dict = split_by_video(X, test_clip_lengths, vid_names)
    y_dict = split_by_video(y, test_clip_lengths, vid_names)

    for _ in range(100):
        while True:
            train_idx = random.sample(list(range(len(test_clip_lengths))), num_learned)
            test_idx = np.setdiff1d(list(range(len(test_clip_lengths))), train_idx)

            test_lengths = np.cumsum(test_clip_lengths_single[test_idx])

            X_train = np.concatenate([X_dict[vid_names[i]] for i in train_idx])
            y_train = np.concatenate([y_dict[vid_names[i]] for i in train_idx])
            X_test = np.concatenate([X_dict[vid_names[i]] for i in test_idx])
            y_test = np.concatenate([y_dict[vid_names[i]] for i in test_idx])

            if len(np.unique(y_train)) > 1:
                break

        model = LogisticRegression()
        model.fit(X_train, y_train)

        final_scores = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, final_scores)
        no_gaus.append(auc)

        final_scores = gaussian_video(final_scores, test_lengths, sigma=(3 if args.dataset_name == 'ped2' else 7))
        auc = roc_auc_score(y_test, final_scores)
        yes_gaus.append(auc)

    print(f"no gaus, video selection, AUC: {(100 * np.mean(no_gaus)):.1f}%±{(100 * np.std(no_gaus)):.1f}%")
    print(f"yes gaus, video selection, AUC: {(100 * np.mean(yes_gaus)):.1f}%±{(100 * np.std(yes_gaus)):.1f}%")


def calc_scores(dataset_name):
    vid_names = os.listdir(f"data/{dataset_name}/testing/frames")
    vid_names.sort()

    test_clip_lengths = np.load(f"data/{dataset_name}/test_clip_lengths.npy")

    test_dataset = VideoDatasetWithFlows(dataset_name=dataset_name, root='data/', train=False)
    y = test_dataset.all_gt

    # X = np.load(f"final_scores/{dataset_name}.npy").transpose()  # (num_frames, 4)
    # X_sliding_window = np.load(f"final_scores-2/{dataset_name}.npy").transpose()  # (num_frames, 4)
    P, V, IE, VE = np.load(f"final_scores-3/{dataset_name}.npy")  # .transpose()  # (num_frames, 4)
    #P, V, IE = np.load(f"../MFAD/Accurate-Interpretable-VAD/{dataset_name}.npy")

    X = np.stack((P, V, IE, VE)).T

    # X1 = np.stack((np.sum(X[:, :3], axis=1), X[:, 3])).T
    # X2 = np.stack((*X.T, np.max(X, axis=1))).T
    # X3 = np.stack((np.sum(X[:, :3], axis=1), X[:, 3], np.max(X, axis=1))).T
    # X4 = np.stack((*X.T, np.sum(X[:, :3], axis=1), np.max(X, axis=1))).T

    # random_videos(X_non_overlapping, y, test_clip_lengths, vid_names)

    # for p in [9]: #range(1,9):
    #     print('only X',p*10)
    #     #print('P+V+IE, VE')
    #     #random_frames(X1, y)
    # print()
    # for p in [9]: #range(1,9):
    #     print('X, max(X)',p*10)
    #     random_frames(X2, y, test_clip_lengths, dataset_name,p)
    # print('P+V+IE, VE, max(X)')
    # random_frames(X3, y)
    # print('X, P+V+IE, max(X)')
    # random_frames(X4, y)

    # print('X')
    # final_scores = np.sum(X,axis=1)#+np.max(X, axis=1)
    # final_scores = gaussian_video(final_scores, test_clip_lengths, sigma=(3 if dataset_name == 'ped2' else 7))
    # auc = roc_auc_score(y, final_scores)
    # print(f'auc {auc*100:.1f}%')

    # print('sliding window')
    # random_videos(X_sliding_window, y, test_clip_lengths, vid_names)
    # random_frames(X_sliding_window, y)

    sigma = (3 if dataset_name in ['ped2', 'avenue'] else 7)
    #for sigma in [1,2,3,4,5,6,7,8,9,10]:
    
    print('SIGMA', sigma)
    """
    final_scores = gaussian_video(VE, test_clip_lengths, sigma=sigma)
    auc = roc_auc_score(y, final_scores)
    print(f'VE {auc * 100:.1f}%')
    """
    final_scores = gaussian_video(P + V, test_clip_lengths, sigma=sigma)
    auc = roc_auc_score(y, final_scores)
    print(f'P+V {auc * 100:.1f}%')
    """
    final_scores = gaussian_video(P + V + IE, test_clip_lengths, sigma=sigma)
    auc = roc_auc_score(y, final_scores)
    print(f'P+V+IE {auc * 100:.1f}%')
    
    final_scores = gaussian_video(P + V + VE, test_clip_lengths, sigma=sigma)
    auc = roc_auc_score(y, final_scores)
    print(f'P+V+VE {auc * 100:.1f}%')
    
    final_scores = gaussian_video(P + V + IE + VE, test_clip_lengths, sigma=sigma)
    auc = roc_auc_score(y, final_scores)
    print(f'P+V+IE+VE {auc * 100:.1f}%')

    final_scores = gaussian_video(P + V + IE + VE + np.max(np.stack((P, V, IE, VE)), axis=0), test_clip_lengths,
                                  sigma=sigma)
    auc = roc_auc_score(y, final_scores)
    print(f'P+V+IE+VE+max {auc * 100:.1f}%')
    
    
    print('X')
    for p in [1, 2, 3, 4, 5, 10, 20, 50, 90]:
        random_frames(X, y, test_clip_lengths, dataset_name, p / 100, sigma)
    # random_frames(X, y, test_clip_lengths, dataset_name, 0)
    print('X+max')
    for p in [1, 2, 3, 4, 5, 10, 20, 50, 90]:
        random_frames(np.stack((*X.T, np.max(X, axis=1))).T, y, test_clip_lengths, dataset_name, p / 100, sigma)
    # random_frames(np.stack((*X.T, np.max(X, axis=1))).T, y, test_clip_lengths, dataset_name, 0.05)
    """ 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ped2', help='dataset name')
    parser.add_argument('--relevant_videos', nargs='+', default=["03_0032", "03_0039", "07_0008"],
                        help='relevant videos')
    args = parser.parse_args()

    # plot_videos(args.relevant_videos, args.dataset_name)
    for dataset_name in ['HMDB51_AD', 'HMDB-Violence']:  # 'ped2', 'avenue', 'shanghaitech', 'HMDB51_AD', 'HMDB-Violence']:
        print(dataset_name)
        calc_scores(dataset_name)
        print('\n---------------------------\n')
