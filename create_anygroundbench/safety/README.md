# Safety video reconstruction for AnyGroundBench

The safety domain combines videos from two existing datasets: DoTA and UCA.


## Preparation

Download the source videos or frames from the official dataset pages.

[![DoTA](https://img.shields.io/badge/Dataset-DoTA-8F6F55.svg)](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)

[![UCA](https://img.shields.io/badge/Dataset-UCA-8F6F55.svg)](https://github.com/Xuange923/Surveillance-Video-Understanding)

[![UCF-Crime](https://img.shields.io/badge/Dataset-UCF--Crime-8F6F55.svg)](https://www.crcv.ucf.edu/projects/real-world/)


The expected preparation is:

1. Prepare DoTA frames under `data/DoTA/frames/<clip_id>/images/*.jpg`.
2. Download and extract UCF-Crime videos under `data/UCF_Crimes/`.
3. Keep the downloaded AnyGroundBench `data/safety/` files in place.

```text
data/DoTA/
└── frames/
    ├── 0RJPQ_97dcs_000897/
    │   └── images/
    │       ├── 000001.jpg
    │       ├── 000002.jpg
    │       └── ...
    ├── 0RJPQ_97dcs_002194/
    └── ...

data/UCF_Crimes/
└── ...
    ├── Abuse045_x264.mp4
    ├── Arrest019_x264.mp4
    ├── Arrest036_x264.mp4
    └── ...

data/safety/
├── filename_mapping.json
├── data_splits/
│   ├── test.txt
│   └── train.txt
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

## Step 1: Videos

Run from the repository root to create the DoTA output clips.

```bash
uv run --project envs/download create_anygroundbench/safety/prepare_dota_videos.py
```

Then create the UCA/UCF-Crime output clips.

```bash
uv run --project envs/download create_anygroundbench/safety/prepare_uca_videos.py
```


## Step 2: Spatial clips

After `safety/videos/clips` has been reconstructed, create DoTA and UCA spatial clips.
Create the clips.

```bash
uv run --project envs/download create_anygroundbench/safety/prepare_spatial_clips.py
```


## Output


```text
data/safety/
├── filename_mapping.json
├── data_splits/
│   ├── test.txt
│   └── train.txt
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   ├── 0RJPQ_97dcs_000897.mp4
    │   ├── 0RJPQ_97dcs_002194.mp4
    │   ├── 0RJPQ_97dcs_002604.mp4
    │   ├── Abuse045_x264.mp4
    │   ├── Arrest019_x264.mp4
    │   ├── Arrest036_x264.mp4
    │   └── ...
    └── clips4spatial/
        ├── DoTA_test_10.mp4
        ├── DoTA_test_1001.mp4
        ├── DoTA_test_1004.mp4
        ├── uca_test_0.mp4
        ├── uca_test_1.mp4
        ├── uca_test_10.mp4
        └── ...
```
