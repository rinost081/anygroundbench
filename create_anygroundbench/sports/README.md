# Sports video reconstruction for AnyGroundBench

The sports domain combines videos from MultiSports and newly captured american football videos.


## Preparation

Download the source videos from the official dataset pages.

[![MultiSports](https://img.shields.io/badge/Dataset-MultiSports-FD8BA8.svg)](https://huggingface.co/datasets/MCG-NJU/SportsAction/tree/main/data)


The expected preparation is:

1. Download the MCG-NJU/SportsAction dataset from Hugging Face and place its `data/` directory under `data/MultiSports/`.
2. Keep the downloaded AnyGroundBench `data/sports/` files in place. The `american_football_*` videos are newly captured and are already included when you download AnyGroundBench.

```text
data/MultiSports/
└── data/
    ├── trainval/
    │   ├── aerobic_gymnastics/
    │   │   ├── v_-hyYa8ijq-8_c001.mp4
    │   │   ├── v_-hyYa8ijq-8_c002.mp4
    │   │   └── v_-hyYa8ijq-8_c003.mp4
    │   ├── basketball/
    │   ├── football/
    │   └── volleyball/
    └── test/
        ├── aerobic_gymnastics/
        ├── basketball/
        ├── football/
        └── volleyball/

data/sports/
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   ├── kc_wide_clip002.mp4
    │   ├── kc_wide_clip004.mp4
    │   ├── kc_wide_clip005.mp4
    │   └── ...
    └── clips4spatial/
        ├── american_football_test_0.mp4
        ├── american_football_test_1.mp4
        ├── american_football_test_10.mp4
        └── ...
```

## Step 1: Videos

Run from the repository root to create the MultiSports output videos.

```bash
uv run --project envs/download create_anygroundbench/sports/prepare_videos.py
```


## Step 2: Spatial clips

After `sports/videos/clips` has been reconstructed, create MultiSports spatial clips.
Create the clips.

```bash
uv run --project envs/download create_anygroundbench/sports/prepare_spatial_clips.py
```


## Output


```text
data/sports/
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   ├── kc_wide_clip002.mp4
    │   ├── kc_wide_clip004.mp4
    │   ├── kc_wide_clip005.mp4
    │   ├── v_-6Os86HzwCs_c002.mp4
    │   ├── v_-6Os86HzwCs_c004.mp4
    │   ├── v_-6Os86HzwCs_c007.mp4
    │   └── ...
    └── clips4spatial/
        ├── MultiSports_test_0.mp4
        ├── MultiSports_test_1.mp4
        ├── MultiSports_test_10.mp4
        ├── american_football_test_0.mp4
        ├── american_football_test_1.mp4
        ├── american_football_test_10.mp4
        └── ...
```
