import argparse
import multiprocessing

import cv2
import os
from pathlib import *

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help='dataset name')
args = parser.parse_args()

path_videos = f"{args.dataset_name}/training/videos"
path_frames = f"{args.dataset_name}/training/frames"

films = list()
files = (x for x in Path(path_videos).iterdir() if x.is_file())
for file in files:
    films.append(file)

pool = multiprocessing.Pool()


def handle_vid(film):
    count = 0
    vidcap = cv2.VideoCapture(str(film))
    success, image = vidcap.read()
    vid_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    mapp = str(film.name).split(".")[0]
    os.makedirs(f"{path_frames}/{mapp}", exist_ok=True)
    while success:
        if vid_length < 100:
            name = f"{path_frames}/{mapp}/{count:02d}.jpg"
        elif vid_length < 1000:
            name = f"{path_frames}/{mapp}/{count:03d}.jpg"
        else:
            name = f"{path_frames}/{mapp}/{count:04d}.jpg"
        if not os.path.isfile(name):
            cv2.imwrite(name, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

outer_pool = multiprocessing.Pool()
outer_pool.map(handle_vid, films)

print("Done!")
