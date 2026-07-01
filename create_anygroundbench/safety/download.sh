#!/usr/bin/env bash
set -euo pipefail

# DoTA
mkdir -p data/DoTA
rclone copy \
  --drive-root-folder-id 1_WzhwZC2NIpzZIpX7YCvapq66rtBc67n \
  --include 'DoTA_seg.z*' \
  --exclude '*' \
  dota_gdrive: \
  data/DoTA \
  --progress
zip -s 0 data/DoTA/DoTA_seg.zip --out data/DoTA/DoTA_seg_full.zip
unzip data/DoTA/DoTA_seg_full.zip -d data/DoTA

uv run create_anygroundbench/safety/prepare_dota_videos.py

# UCF-Crimes (UCA)
mkdir -p data/UCF_Crimes
aria2c -x 16 -s 16 -k 1M \
  -d data/UCF_Crimes \
  -o UCF_Crimes_dropbox.zip \
  'https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=1&dl=1'
unzip data/UCF_Crimes/UCF_Crimes_dropbox.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Anomaly-Videos-Part-1.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Anomaly-Videos-Part-2.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Anomaly-Videos-Part-3.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Anomaly-Videos-Part-4.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Training-Normal-Videos-Part-1.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Training-Normal-Videos-Part-2.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Testing_Normal_Videos.zip -d data/UCF_Crimes
unzip data/UCF_Crimes/Normal_Videos_for_Event_Recognition.zip -d data/UCF_Crimes

uv run create_anygroundbench/safety/prepare_uca_videos.py

uv run create_anygroundbench/safety/prepare_spatial_clips.py
