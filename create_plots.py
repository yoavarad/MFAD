"""
Use this script to create plots for the results of the anomaly detection.
"""

import os

import numpy as np
from matplotlib import pyplot as plt, patches

def min_max_normalize(x: np.ndarray, x_min=None, x_max=None):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    res = (x - x_min) / (x_max - x_min)
    return res


def first_and_last_seq(x, n=1):
    a = np.r_[n - 1, x, n - 1]
    a = a == n
    start = np.r_[False, ~a[:-1] & a[1:]]
    end = np.r_[a[:-1] & ~a[1:], False]
    boundaries: np.ndarray = np.where(start | end)[0] - 1
    if np.any(start == end):
        boundaries = np.append(boundaries, (np.where(start & end)[0] - 1))
        boundaries = np.sort(boundaries)
    return boundaries.reshape(-1, 2)


def split_by_video(scores, lengths, video_names):
    video_scores = {}
    for i, video_name in enumerate(video_names):
        if i == 0:
            prev, cur = 0, lengths[0]
        else:
            prev, cur = lengths[i - 1:i + 1]
        video_scores[video_name] = scores[prev: cur]
    return video_scores


def create_plots(scores, gt, lengths, dataset_name):
    video_names = os.listdir(f'data/{dataset_name}/testing/frames')
    video_names.sort()

    video_labels = split_by_video(gt, lengths, video_names)
    video_scores = split_by_video(scores, lengths, video_names)

    plt.rcParams.update({'font.size': 22})

    plot_dir = f'plots/{dataset_name}'
    os.makedirs(plot_dir, exist_ok=True)
    for video_name, score_list in video_scores.items():
        score_local_list = min_max_normalize(score_list)

        fig, ax = plt.subplots(figsize=(20, 10), )

        boundaries = first_and_last_seq(video_labels[video_name])
        rects = []
        for s, e in boundaries:
            rects.append(patches.Rectangle((s / 25, 0), (e - s) / 25, 1, facecolor='red', alpha=0.2))

        for rect in rects:
            ax.add_patch(rect)

        ax.plot(np.linspace(0, len(score_local_list) / 25, num=len(score_local_list)), score_local_list, color='b')

        ax.set_ylim([0, 1])
        ax.set_title(f'{dataset_name}/{video_name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Anomaly Score')

        plt.savefig(f'{plot_dir}/{video_name}.png')
        plt.close()
