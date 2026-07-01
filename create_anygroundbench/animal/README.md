# Animal video reconstruction for AnyGroundBench

The animal domain combines videos from Animal Kingdom and newly captured mouse videos.


## Preparation

Download the source videos from the official dataset pages.

[![Animal Kingdom](https://img.shields.io/badge/Dataset-Animal_Kingdom-017B9B.svg)](https://sutdcv.github.io/Animal-Kingdom/)



The expected preparation is:

1. Download Animal Kingdom `dataset.tar.gz` from `video grounding` folder and extract it under `data/animal_kingdom/`. The expected video root is `data/animal_kingdom/dataset/`.
2. Keep the downloaded AnyGroundBench `data/animal/` files in place. The `mouse_*` videos are newly captured and are already included when you download AnyGroundBench.

```text
data/animal_kingdom/
└── dataset/
    ├── AABGBPZC.mp4
    ├── AADJBFXO.mp4
    ├── AALAJYTZ.mp4
    └── ...

data/animal/
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   ├── o__20220201_mouse4__front1__p00.mp4
    │   ├── o__20220201_mouse4__front1__p01.mp4
    │   ├── o__20220201_mouse4__front2__p00.mp4
    │   └── ...
    └── clips4spatial/
        ├── mouse_test_0.mp4
        ├── mouse_test_1.mp4
        ├── mouse_test_10.mp4
        └── ...
```

## Step 1: Videos

Run from the repository root to create the Animal Kingdom output videos.

```bash
uv run --project envs/download create_anygroundbench/animal/prepare_videos.py
```

## Step 2: Spatial clips

After `animal/videos/clips` has been reconstructed, create Animal Kingdom spatial clips.
Create the clips.

```bash
uv run --project envs/download create_anygroundbench/animal/prepare_spatial_clips.py
```


## Output


```text
data/animal/
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   ├── ACOSBKEW.mp4
    │   ├── AKPXTPIF.mp4
    │   ├── ANSYYSOB.mp4
    │   ├── o__20220201_mouse4__front1__p00.mp4
    │   ├── o__20220201_mouse4__front1__p01.mp4
    │   ├── o__20220201_mouse4__front2__p00.mp4
    │   └── ...
    └── clips4spatial/
        ├── animal_kingdom_test_0.mp4
        ├── animal_kingdom_test_1.mp4
        ├── animal_kingdom_test_10.mp4
        ├── mouse_test_0.mp4
        ├── mouse_test_1.mp4
        ├── mouse_test_10.mp4
        └── ...
```
