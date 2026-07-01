# Surgery video reconstruction for AnyGroundBench
The surgery domain combines videos from EgoSurgery and CholecTrack20 videos.

## Preparation

Download the source videos from the official dataset pages.

[![CholecTrack20](https://img.shields.io/badge/Dataset-CholecTrack20-FDBD6C.svg)](https://www.synapse.org/Synapse:syn53182642/wiki/628404)

[![EgoSurgery](https://img.shields.io/badge/Dataset-EgoSurgery-FDBD6C.svg)](https://github.com/Fujiry0/EgoSurgery)

EgoSurgery files used here are available from the Google Drive folder:
`https://drive.google.com/drive/u/1/folders/1wiqxozwIvdmcMMyxRA7YeJETlPmergud`.



The expected preparation is:

1. Download CholecTrack20 from Synapse after completing the official access steps. The public CholecTrack20 README says the release is a single zip file with three split directories; each split contains one directory per video sequence, and each sequence directory contains the raw `.mp4`, sampled frames, and labels JSON. Extract that zip under `data/CholecTrack20/`.
2. Keep the downloaded AnyGroundBench `data/surgery/` files in place. The reconstruction scripts read `data/surgery/meta-data/st_*.json` as the authoritative clip metadata.
3. Download `images.zip` from the EgoSurgery Google Drive folder above, and extract it under `data/egosurgery/`. The zip contains the top-level `images/` directory, so the expected frame root is `data/egosurgery/images/`.

The exact CholecTrack20 split directory names are not assumed here. The reconstruction script recursively searches for Testing mp4 files and also supports Training/Validation `Frames/*.png` directories under `data/CholecTrack20/`.

```text
data/CholecTrack20/
├── <official-split-dir>/
│   └── <video-sequence-dir>/
│       ├── VID02.mp4
│       ├── <1fps-frame-dir>/
│       └── <labels>.json
├── <official-split-dir>/
│   └── <video-sequence-dir>/
│       └── VID06.mp4
data/egosurgery/
└── images/
    ├── 01/
    │   ├── 01_1_0001.jpg
    │   ├── 01_1_0002.jpg
    │   └── ...
    ├── 02/
    │   └── 02_1_0001.jpg
    └── ...

data/surgery/
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   └── <empty>
    └── clips4spatial/
        └── <empty>
```


## Confirmed Clip Rules

The surgery `prepare_*videos.py` scripts do not recompute the N-split selection from scratch. They use the already-confirmed AnyGroundBench `data/surgery/meta-data/st_*.json` metadata and regenerate `data/surgery/videos/clips`.

EgoSurgery:

- split: N=50
- source: `data/egosurgery/images/{case}/{source_stem}_%04d.jpg` extracted image sequences
- metadata: `data/surgery/meta-data/st_*.json`; `meta_info.source_video_name`, `clip_start_sec`, and `clip_end_sec` are authoritative
- clip rule: first render a temporary long 0.5 fps source mp4 from the image sequence with `ffmpeg -framerate 0.5 -start_number 1 -i <source_stem>_%04d.jpg -an -c:v libx264 -profile:v baseline -preset ultrafast -crf 23 -pix_fmt yuv420p`, then re-encode the clip with `ffmpeg -ss <clip_start_sec> -i <temporary_source_video> -t <clip_end_sec - clip_start_sec> -an -c:v libx264 -profile:v baseline -preset ultrafast -crf 23 -pix_fmt yuv420p`
- verification: this two-stage image-based rule reproduced all 150 EgoSurgery clips with decoded raw-pixel SHA equality against the existing `data/egosurgery/videos/0` clips
- script: `create_anygroundbench/surgery/prepare_egosurgery_videos.py`

CholecTrack20:

- split: N=20
- source: `data/CholecTrack20`
- metadata: `data/surgery/meta-data/st_*.json`
- Training/Validation source: `Frames/*.png`; frame IDs are selected from `floor(start_sec * source_fps)` through `ceil(end_sec * source_fps)`, inclusive.
- Testing source: mp4; clips are extracted with the historical `ffmpeg -ss/-t` second-based rule.
- script: `create_anygroundbench/surgery/prepare_cholectrack20_videos.py`

## Step 1: Videos

Run from the repository root to create the CholecTrack20 output clips.

```bash
uv run --project envs/download create_anygroundbench/surgery/prepare_cholectrack20_videos.py
```

Then create the EgoSurgery output clips.

```bash
uv run --project envs/download create_anygroundbench/surgery/prepare_egosurgery_videos.py
```


## Step 2: Spatial clips

Create the spatial clips after `surgery/videos/clips` has been reconstructed. EgoSurgery spatial clips are cut from `data/surgery/videos/clips`; CholecTrack20 spatial clips are cut from the raw CholecTrack20 source because the reference clips4spatial are not byte-equivalent when reconstructed from the prepared clips.

If CholecTrack20 is extracted under the default `data/CholecTrack20`, run:

```bash
uv run --project envs/download create_anygroundbench/surgery/prepare_spatial_clips.py
```

If CholecTrack20 is stored elsewhere, pass the raw root explicitly:

```bash
uv run --project envs/download create_anygroundbench/surgery/prepare_spatial_clips.py --cholectrack20-raw-root /path/to/CholecTrack20
```


## Output


```text
data/surgery/
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   ├── 01_1_12.mp4
    │   ├── 01_1_13.mp4
    │   ├── 01_1_14.mp4
    │   ├── VID02_11.mp4
    │   ├── VID02_13.mp4
    │   ├── VID02_14.mp4
    │   └── ...
    └── clips4spatial/
        ├── CholecTrack20_test_0.mp4
        ├── CholecTrack20_test_1.mp4
        ├── CholecTrack20_test_10.mp4
        ├── egosurgery_test_0.mp4
        ├── egosurgery_test_1.mp4
        ├── egosurgery_test_10.mp4
        └── ...
```
