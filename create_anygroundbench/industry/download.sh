# ~/AnyGroundBench

# MECCANO download
mkdir -p data/MECCANO
aria2c -x 16 -s 16 -k 1M \
  -d data/MECCANO \
  -o MECCANO_RGB_Videos.zip \
  https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_RGB_Videos.zip
unzip data/MECCANO/MECCANO_RGB_Videos.zip -d data/MECCANO

# extract videos
uv run create_anygroundbench/industry/prepare_enigma_videos.py

# ENIGMA download
mkdir -p data/ENIGMA/videos
aria2c -x 16 -s 16 -k 1M \
  -d data/ENIGMA \
  -o ENIGMA-51-videos.zip \
  https://iplab.dmi.unict.it/sharing/ENIGMA-51/ENIGMA-51-videos.zip
unzip data/ENIGMA/ENIGMA-51-videos.zip -d data/ENIGMA/videos

# extract videos
uv run create_anygroundbench/industry/prepare_meccano_videos.py

# create spatial clips
uv run create_anygroundbench/industry/prepare_spatial_clips.py
