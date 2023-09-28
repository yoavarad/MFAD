# MFAD Setup

## 1. Clone Repositories

Clone the VideoMAEv2 and AI-VAD repositories to your desired folder:

```bash
git clone https://github.com/OpenGVLab/VideoMAEv2.git
git clone https://github.com/talreiss/Accurate-Interpretable-VAD.git
```

## 2. Modify AI-VAD

Replace original `evaluate.py`, `video_dataset.py`, `score_calibration_vmae.py` with the modified 
versions in the AI-VAD directory in this repository.

Move `final_scores.py` to `Accurate-Interpretable-VAD/`.

Use `score_calibration_vmae.py` in addition to `score_calibration.py` to calibrate the scores for 
the video encoding features.

In `pre_processing/bboxes.py`, extend the `DATASET_CFGS` dictionary with:

```python
DATASET_CFGS['HMDB-AD'] = DATASET_CFGS['HMDB-Violence'] = \
    {"confidence_threshold": 0.6,
     "min_area": 8 * 8,
     "cover_threshold": 0.6,
     "binary_threshold": 15,
     "gauss_mask_size": 5,
     "contour_min_area": 40 * 40}
```

In `pre_processing/flows.py`, extend the `DATASET_CFGS` dictionary with:

```python
FLOWNET_INPUT_WIDTH['HMDB-AD'] = FLOWNET_INPUT_WIDTH['HMDB-Violence'] = 1024
FLOWNET_INPUT_HEIGHT['HMDB-AD'] = FLOWNET_INPUT_HEIGHT['HMDB-Violence'] = 640
```

## 3. Modify VideoMAEv2

Replace original `extract_tad_features.py` with the modified version in the VideoMAEv2 directory in 
this repository.

To extract features run:
```bash 
python extract_tad_feature.py --data_set '<dataset_name>' \
                              --data_path 'data/<dataset_name>/<testing|training>/videos' \
                              --save_path 'extracted_features/vit_g_hybrid_pt_1200e_ssv2_ft/<dataset_name>/<testing|training>' \
                              --model 'vit_giant_patch14_224' \
                              --ckpt_path 'model_weights/vit_g_hybrid_pt_1200e_ssv2_ft.pth'
```

## 4. Create HMDB-AD and HMDB-Violence

Download the HMDB51 dataset from: http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

Run `Datasets/create_hmdb_datasets.sh`. Use `Accurate-Interpretable-VAD/data/count_frames.py` to create the 
`*_clip_lengths.npy` files.

## 5. Validate 'data' Folder

Validate that your 'data' folder structure is as follows, following the format used in AI-VAD:

```
data/
-- <dataset_name>/
   |-- test_clip_lengths.npy
   |-- train_clip_lengths.npy
   |-- training/
   |    |-- frames/
   |    |    |-- <vid_1>/
   |    |    |    |-- 000.jpg
   |    |    |    |-- 001.jpg
   |    |    |    |-- ...
   |    |    |-- <vid_2>/
   |    |    |    |-- ...
   |    |    |-- ...
   |    |-- videos/
   |    |    |-- <vid_1>.avi
   |    |    |-- <vid_2>.avi
   |    |    |-- ...
   |
   |-- testing/
        |-- ...
```