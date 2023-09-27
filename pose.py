import argparse
import json
import os
from collections import defaultdict
import numpy as np
import glob

from tqdm import tqdm

Hpose, Wpose = 0, 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HMDB-Violence')
args = parser.parse_args()

train_json_filenames = glob.glob(f'pose_features/{args.dataset}/training/**/*.json', recursive=True)

train_keypoints = []
total_train_pose = 0
for json_filename in tqdm(sorted(train_json_filenames), total=len(train_json_filenames), desc='train'):
    vid_keypoints = defaultdict(list)
    vid_name = json_filename.split('/')[-2]

    vid_len = len(os.listdir(f'../VAD-Datasets/{args.dataset}/training/frames/{vid_name}'))

    pose_json = json.load(open(json_filename))
    for pose in pose_json:
        img_id = pose['image_id']
        if '(1)' in img_id:  # skip duplicate frames (copy bug)
            continue

        keypoints = np.array(pose['keypoints'])
        keypoints = keypoints.reshape((keypoints.shape[0] // 3, 3))
        keypoints = keypoints[:, :2]

        x, y, w, h = pose['box']

        Hpose += h
        Wpose += w
        total_train_pose += 1

        keypoints = (keypoints - np.array([x, y])) * np.array([1 / w, 1 / h])  # normalize

        img_num = int(img_id.split('.')[0])
        vid_keypoints[img_num].append(keypoints)

    for img_num in range(vid_len):
        if vid_keypoints[img_num] == []:
            img_keypoints = np.array([])
        else:
            img_keypoints = np.stack(vid_keypoints[img_num])  # (num_persons, num_keypoints, 2)
        train_keypoints.append(img_keypoints)  # (num_frames, ) of (num_persons, num_keypoints, 2)

Hpose /= total_train_pose
Wpose /= total_train_pose

train_keypoints = [vid_keypoints * np.array([Wpose, Hpose]) if vid_keypoints.shape != (0,) else vid_keypoints
                   for vid_keypoints in train_keypoints]
train_keypoints = [vid_keypoints.reshape(vid_keypoints.shape[0], -1) if vid_keypoints.shape != (0,) else vid_keypoints
                   for vid_keypoints in train_keypoints]
print(len(train_keypoints))

train_keypoints = np.array(train_keypoints, dtype=object)  # (num_frames, ) of (num_persons, num_keypoints*2)

out_dir = f'extracted_features/{args.dataset}/train'
os.makedirs(out_dir, exist_ok=True)
np.save(f'{out_dir}/pose.npy', train_keypoints)
print(f'Saved {out_dir}/pose.npy of shape {train_keypoints.shape} (inside shape: {train_keypoints[0].shape})')
del train_keypoints

test_json_filenames = glob.glob(f'pose_features/{args.dataset}/testing/**/*.json', recursive=True)
test_keypoints = []
for json_filename in tqdm(sorted(test_json_filenames), total=len(test_json_filenames), desc='test'):
    vid_keypoints = defaultdict(list)
    vid_name = json_filename.split('/')[-2]
    vid_len = len(os.listdir(f'../VAD-Datasets/{args.dataset}/testing/frames/{vid_name}'))

    pose_json = json.load(open(json_filename))
    for pose in pose_json:
        img_id = pose['image_id']
        if '(1)' in img_id:  # skip duplicate frames (copy bug)
            continue

        keypoints = np.array(pose['keypoints'])
        keypoints = keypoints.reshape((keypoints.shape[0] // 3, 3))
        keypoints = keypoints[:, :2]

        x, y, w, h = pose['box']

        keypoints = (keypoints - np.array([x, y])) * np.array([Wpose / w, Hpose / h])  # normalize

        img_num = int(img_id.split('.')[0])
        vid_keypoints[img_num].append(keypoints)

    for img_num in range(vid_len):
        if vid_keypoints[img_num] == []:
            img_keypoints = np.array([])
        else:
            img_keypoints = np.stack(vid_keypoints[img_num])  # (num_persons, num_keypoints, 2)
        test_keypoints.append(img_keypoints)  # (num_frames, ) of (num_persons, num_keypoints, 2)

test_keypoints = [vid_keypoints.reshape(vid_keypoints.shape[0], -1) if vid_keypoints.shape != (0,) else vid_keypoints
                  for vid_keypoints in test_keypoints]
print(len(test_keypoints))
test_keypoints = np.array(test_keypoints, dtype=object)  # (num_frames, ) of (num_persons, num_keypoints*2)

out_dir = f'extracted_features/{args.dataset}/test'
os.makedirs(out_dir, exist_ok=True)
np.save(f'{out_dir}/pose.npy', test_keypoints)
print(f'Saved {out_dir}/pose.npy of shape {test_keypoints.shape} (inside shape: {test_keypoints[0].shape})')
