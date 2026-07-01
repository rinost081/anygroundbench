# Industry video reconstruction for AnyGroundBench

The industry domain combines videos from two existing datasets: MECCANO and ENIGMA.


## Preparation

Download the source videos from the official dataset pages.

[![MECCANO](https://img.shields.io/badge/Dataset-MECCANO-00C0BA.svg)](https://iplab.dmi.unict.it/MECCANO/)

[![ENIGMA](https://img.shields.io/badge/Dataset-ENIGMA--51-00C0BA.svg)](https://fpv-iplab.github.io/ENIGMA-51/)



The expected preparation is:

1. Download `MECCANO_RGB_Videos.zip` from the official MECCANO page and extract it under `data/MECCANO/`.
2. Download `ENIGMA-51-videos.zip` from the official ENIGMA-51 page and extract it under `data/ENIGMA/videos/`. The zip contains `enigma-data/videos/`, so the default source directory becomes `data/ENIGMA/videos/enigma-data/videos/`.
3. Keep the downloaded AnyGroundBench `data/industry/` files in place.

```text
data/MECCANO/
└── MECCANO_RGB_Videos/
    ├── 0001.mp4
    ├── 0002.mp4
    ├── 0003.mp4
    └── ...

data/ENIGMA/
└── videos/
    └── enigma-data/
        └── videos/
            ├── 44.mp4
            ├── 46.mp4
            ├── 47.mp4
            └── ...

data/industry/
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


Example ENIGMA download commands:

```bash
mkdir -p data/ENIGMA/videos
aria2c -x 16 -s 16 -k 1M \
  -d data/ENIGMA \
  -o ENIGMA-51-videos.zip \
  https://iplab.dmi.unict.it/sharing/ENIGMA-51/ENIGMA-51-videos.zip
unzip data/ENIGMA/ENIGMA-51-videos.zip -d data/ENIGMA/videos
```

With the official zip structure, the default source directory is `data/ENIGMA/videos/enigma-data/videos`. If you move the mp4 files elsewhere, pass `--full-video-dir` to the directory that directly contains them.

## Step 1: Videos

Run from the repository root to create the MECCANO output clips.

```bash
uv run --project envs/download create_anygroundbench/industry/prepare_meccano_videos.py
```

Then create the ENIGMA-51 output clips.

```bash
uv run --project envs/download create_anygroundbench/industry/prepare_enigma_videos.py
```


### Source frame range

The regenerated industry clips intentionally use the same second-based FFmpeg commands as the historical source-generation scripts. MECCANO uses `-ss <clip_start_sec> -i <source> -t <duration> -an -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p -movflags +faststart`. ENIGMA uses `-ss <clip_start_sec> -i <source> -t <duration> -an -c:v libx264 -preset veryfast -crf 18`.

For frame-number accounting, use the source video FPS and the generated clip frame count:

```text
start_frame = ceil(clip_start_sec * source_fps - 1e-9)
end_frame_exclusive = start_frame + generated_clip_nb_frames
source frame range = [start_frame, end_frame_exclusive)
```

Do not derive the exclusive end only from `clip_end_sec`: ENIGMA has boundary cases where `ceil(clip_end_sec * fps)` is one frame shorter than the FFmpeg output. The rule above is the one verified against the historical clips.

Verification performed on 2026-06-30: 10 MECCANO samples and 10 ENIGMA samples were regenerated into `data/industry/videos/clips` and compared against the existing clips in `~/video_icl/data/{MECCANO,ENIGMA}/videos/0` with decoded-frame SHA256 hashes. All 20 sampled clips matched frame-for-frame.

## Step 2: Spatial clips

After `industry/videos/clips` has been reconstructed, create MECCANO and ENIGMA spatial clips.
Create the clips.

```bash
uv run --project envs/download create_anygroundbench/industry/prepare_spatial_clips.py
```


## Output


```text
data/industry/
├── meta-data/
│   ├── s_*.json
│   ├── t_*.json
│   └── st_*.json
├── simrank/
│   └── ...
└── videos/
    ├── clips/
    │   ├── 0001_14.mp4
    │   ├── 0001_19.mp4
    │   ├── 0001_2.mp4
    │   ├── ENIGMA_044_006.mp4
    │   ├── ENIGMA_044_022.mp4
    │   ├── ENIGMA_046_018.mp4
    │   └── ...
    └── clips4spatial/
        ├── ENIGMA_test_0.mp4
        ├── ENIGMA_test_1.mp4
        ├── ENIGMA_test_10.mp4
        ├── MECCANO_test_0.mp4
        ├── MECCANO_test_1.mp4
        ├── MECCANO_test_10.mp4
        └── ...
```
