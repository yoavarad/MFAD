import glob
import os


def rename_videos(dataset):
    all_video_files = glob.glob(f'{dataset}/**/*.avi', recursive=True)
    all_video_files.sort()

    for video_file in all_video_files:
        split_name = video_file.split('/')
        new_filename = f'{split_name[-2]}_{split_name[-1]}'
        split_name[-1] = new_filename
        split_name.pop(-2)
        new_path = '/'.join(split_name)
        os.rename(video_file, new_path)

def move_normal_videos(dataset):
    from normal_test_videos import normal_test_videos

    for vid_to_move in normal_test_videos:
        os.rename(f'{dataset}/training/videos/{vid_to_move}.avi',
                  f'{dataset}/testing/videos/{vid_to_move}.avi')


if __name__ == '__main__':
    for dataset in ['HMDB-AD', 'HMDB-Violence']:
        rename_videos(dataset)
        move_normal_videos(dataset)
