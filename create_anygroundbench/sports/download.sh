#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/MultiSports
uv run --project envs/download hf download MCG-NJU/SportsAction --repo-type dataset --local-dir data/MultiSports --include 'data/*'
tar -xf data/MultiSports/data/trainval/aerobic_gymnastics.tar -C data/MultiSports/data/trainval
tar -xf data/MultiSports/data/trainval/basketball.tar -C data/MultiSports/data/trainval
tar -xf data/MultiSports/data/trainval/football.tar -C data/MultiSports/data/trainval
tar -xf data/MultiSports/data/trainval/volleyball.tar -C data/MultiSports/data/trainval
tar -xf data/MultiSports/data/test/aerobic_gymnastics.tar -C data/MultiSports/data/test
tar -xf data/MultiSports/data/test/basketball.tar -C data/MultiSports/data/test
tar -xf data/MultiSports/data/test/football.tar -C data/MultiSports/data/test
tar -xf data/MultiSports/data/test/volleyball.tar -C data/MultiSports/data/test
uv run create_anygroundbench/sports/prepare_videos.py
