#!/usr/bin/env bash
set -euo pipefail

# Blind test 0 shot
uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_1_pro_preview_blind_test \
  --pred-jsons \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_133626_animal_kingdom.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_153402_american_football.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_123048_mouse.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_124125_dota.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_132418_enigma.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_140111_egosurgery.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_152419_cholectrack20.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_150558_meccano.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_144537_multisports.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_135731_uca.json

# Blind test 2 shot
uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_1_pro_preview_blind_test_2shot \
  --pred-jsons \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_142133_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_152813_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_145519_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_125013_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatio_temporal/20260503_132803_uca_2shot.json

# Ablation 1 shot
uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_1_pro_preview_1shot \
  --pred-jsons \
  data/eval/gemini_3_1_pro_preview_1shot/spatio_temporal/20260504_151907_american_football_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatio_temporal/20260504_120209_egosurgery_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatio_temporal/20260504_061712_enigma_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatio_temporal/20260504_043238_mouse_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatio_temporal/20260504_090126_uca_1shot.json

# Ablation 4 shot
uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_1_pro_preview_4shot \
  --pred-jsons \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260505_014317_american_football_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260504_202919_egosurgery_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260504_100530_enigma_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260504_070326_mouse_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260504_151000_uca_4shot.json


# Ablation retrieval (visual features)
# simrank a = 0
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatio_temporal --model gemini_3_1_pro_preview_a_0 --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_0/spatio_temporal/20260505_005434_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatio_temporal/20260505_005455_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatio_temporal/20260505_005418_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatio_temporal/20260504_224745_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatio_temporal/20260505_021154_uca_2shot.json

# Ablation retrieval (text features)
# simrank a = 1
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatio_temporal --model gemini_3_1_pro_preview_a_1 --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_1/spatio_temporal/20260505_085936_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatio_temporal/20260505_071512_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatio_temporal/20260505_065437_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatio_temporal/20260505_045613_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatio_temporal/20260505_061215_uca_2shot.json
# uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_a_1 --pred-jsons \


# Ablation retrieval (random)
# simrank a: random
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatio_temporal --model gemini_3_1_pro_preview_a_random --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_random/spatio_temporal/20260505_145721_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatio_temporal/20260505_162356_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatio_temporal/20260505_150507_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatio_temporal/20260505_105135_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatio_temporal/20260505_151831_uca_2shot.json



# OSS 0 shot
# dataset            theoretical  q3-vl-4B  q3-vl-8B  q3.5-4B  q3.5-9B  eagle  iv3-8B  iv3.5-8B  iv3-14B
# animal_kingdom     106          106       106       106       106       106    106     106        106
# american_football  37           37        37        37        37        37     37      37         37
# cholectrack20      37           37        37        37        37        37     37      37         37
# dota               231          231       231       231       231       231    231     231        231
# egosurgery         179          179       179       179       179       179    179     179        179
# enigma             58           58        58        58        58        58     58      58         58
# meccano            111          111       111       111       111       111    111     111        111
# mouse              51           51        51        51        51        51     51      51         51
# multisports        126          126       126       126       126       126    126     126        126
# uca                19           19        19        19        19        19     19      19         19

cp results/spatio_temporal/20260504_194239.json data/eval/internvl3_14B/spatio_temporal/20260504_194239_cholectrack20.json
cp results/spatio_temporal/20260504_203703.json data/eval/internvl3_14B/spatio_temporal/20260504_203703_dota.json
cp results/spatio_temporal/20260504_213117.json data/eval/internvl3_14B/spatio_temporal/20260504_213117_animal_kingdom.json
cp results/spatio_temporal/20260505_015245.json data/eval/internvl3_14B/spatio_temporal/20260505_015245_egosurgery.json
cp results/spatio_temporal/20260505_001800.json data/eval/internvl3_14B/spatio_temporal/20260505_001800_uca.json
cp results/spatio_temporal/20260505_003220.json data/eval/internvl3_14B/spatio_temporal/20260505_003220_mouse.json
cp results/spatio_temporal/20260505_012618.json data/eval/internvl3_14B/spatio_temporal/20260505_012618_american_football.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_vl_4B_instruct \
  --pred-jsons \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260419_090839_animal_kingdom.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260427_005212_american_football.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260421_102642_cholectrack20.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260420_181800_dota.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260419_175016_egosurgery.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260424_023643_enigma.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260420_224139_meccano.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260424_014459_mouse.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260421_022325_multisports.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260421_044045_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_vl_8B_instruct \
  --pred-jsons \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260418_013233_animal_kingdom.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260501_010829_american_football.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260423_030949_cholectrack20.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260421_022221_dota.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260418_040555_egosurgery.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260424_041955_enigma.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260421_025651_meccano.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260424_033616_mouse.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260418_074101_multisports.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260418_085636_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_5_4B \
  --pred-jsons \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_135135_animal_kingdom.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_145446_american_football.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_141823_cholectrack20.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_132217_dota.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_151023_egosurgery.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_103958_enigma.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_123555_meccano.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_113616_mouse.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_121435_multisports.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_112448_uca.json


uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_5_9B \
  --pred-jsons \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_123608_animal_kingdom.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_141609_american_football.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260504_124152_cholectrack20.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_114455_dota.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_065618_egosurgery.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_043354_enigma.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_103931_meccano.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_055927_mouse.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_100643_multisports.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260501_054158_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model internvl3_8B \
  --pred-jsons \
  data/eval/internvl3_8B/spatio_temporal/20260419_123413_animal_kingdom.json \
  data/eval/internvl3_8B/spatio_temporal/20260427_025852_american_football.json \
  data/eval/internvl3_8B/spatio_temporal/20260423_121429_cholectrack20.json \
  data/eval/internvl3_8B/spatio_temporal/20260423_122523_dota.json \
  data/eval/internvl3_8B/spatio_temporal/20260419_135531_egosurgery.json \
  data/eval/internvl3_8B/spatio_temporal/20260424_095642_enigma.json \
  data/eval/internvl3_8B/spatio_temporal/20260423_125145_meccano.json \
  data/eval/internvl3_8B/spatio_temporal/20260424_094636_mouse.json \
  data/eval/internvl3_8B/spatio_temporal/20260419_151922_multisports.json \
  data/eval/internvl3_8B/spatio_temporal/20260419_154653_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model internvl3_5_8B \
  --pred-jsons \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_041344_animal_kingdom.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_034519_american_football.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_005310_cholectrack20.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_013040_dota.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_020357_egosurgery.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260424_164949_enigma.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_030411_meccano.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260424_105909_mouse.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_034000_multisports.json \
  data/eval/internvl3_5_8B/spatio_temporal/20260427_040348_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model internvl3_14B \
  --pred-jsons \
  data/eval/internvl3_14B/spatio_temporal/20260504_213117_animal_kingdom.json \
  data/eval/internvl3_14B/spatio_temporal/20260505_012618_american_football.json \
  data/eval/internvl3_14B/spatio_temporal/20260504_194239_cholectrack20.json \
  data/eval/internvl3_14B/spatio_temporal/20260504_203703_dota.json \
  data/eval/internvl3_14B/spatio_temporal/20260505_015245_egosurgery.json \
  data/eval/internvl3_14B/spatio_temporal/20260424_104542_enigma.json \
  data/eval/internvl3_14B/spatio_temporal/20260501_223807_meccano.json \
  data/eval/internvl3_14B/spatio_temporal/20260505_003220_mouse.json \
  data/eval/internvl3_14B/spatio_temporal/20260502_000644_multisports.json \
  data/eval/internvl3_14B/spatio_temporal/20260505_001800_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model eagle \
  --pred-jsons \
  data/eval/eagle/spatio_temporal/20260504_121222_animal_kingdom.json \
  data/eval/eagle/spatio_temporal/20260504_093042_american_football.json \
  data/eval/eagle/spatio_temporal/20260504_094413_cholectrack20.json \
  data/eval/eagle/spatio_temporal/20260504_093740_dota.json \
  data/eval/eagle/spatio_temporal/20260504_122217_egosurgery.json \
  data/eval/eagle/spatio_temporal/20260504_091341_enigma.json \
  data/eval/eagle/spatio_temporal/20260504_092448_meccano.json \
  data/eval/eagle/spatio_temporal/20260504_093238_mouse.json \
  data/eval/eagle/spatio_temporal/20260504_121759_multisports.json \
  data/eval/eagle/spatio_temporal/20260504_121019_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model LLaVA-ST \
  --pred-jsons \
  data/eval/LLaVA-ST/spatio_temporal/20260423_184620_animal_kingdom.json \
  data/eval/LLaVA-ST/spatio_temporal/20260423_190645_cholectrack20.json \
  data/eval/LLaVA-ST/spatio_temporal/20260423_191249_dota.json \
  data/eval/LLaVA-ST/spatio_temporal/20260423_202601_egosurgery.json \
  data/eval/LLaVA-ST/spatio_temporal/20260423_205810_meccano.json \
  data/eval/LLaVA-ST/spatio_temporal/20260423_210906_multisports.json \
  data/eval/LLaVA-ST/spatio_temporal/20260423_212800_uca.json \
  data/eval/LLaVA-ST/spatio_temporal/20260424_180142_enigma.json \
  data/eval/LLaVA-ST/spatio_temporal/20260424_183242_mouse.json


# Proprietary 0 shot
# dataset            theoretical  gpt-4o  gpt5.1  gemini2.5-flash  gemini2.5-pro  gemini3  gemini3.1-pro
# animal_kingdom     106          106     106     106              106            106      106
# american_football  37           37      37      37               37             37       37
# cholectrack20      37           37      37      37               37             37       37
# dota               231          231     231     231              231            231      231
# egosurgery         179          179     179     179              179            179      179
# enigma             58           58      58      58               58             58       58
# meccano            111          111     111     111              111            111      111
# mouse              51           51      51      51               51             51       51
# multisports        126          126     126     126              126            126      126
# uca                19           19      19      19               19             19       19


uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_2_5_flash \
  --pred-jsons \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_063040_animal_kingdom.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_024420_american_football.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_091539_cholectrack20.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_074818_dota.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_041616_egosurgery.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_032207_enigma.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_070733_meccano.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_025751_mouse.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_054418_multisports.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260501_040632_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_2_5_pro \
  --pred-jsons \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_072120_animal_kingdom.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_024456_american_football.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_104533_cholectrack20.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_090000_dota.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_043001_egosurgery.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_033320_enigma.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_080926_meccano.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_030055_mouse.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_062013_multisports.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260501_041814_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_flash_preview \
  --pred-jsons \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_062337_animal_kingdom.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_024526_american_football.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_092336_cholectrack20.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_074328_dota.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_041441_egosurgery.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_032100_enigma.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_070053_meccano.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_025931_mouse.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_053204_multisports.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260501_040523_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_1_pro_preview \
  --pred-jsons \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_065051_animal_kingdom.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_024538_american_football.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_100919_cholectrack20.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_082509_dota.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_041751_egosurgery.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_032434_enigma.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_073254_meccano.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_030009_mouse.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_055635_multisports.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260501_040821_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gpt-4o \
  --pred-jsons \
  data/eval/gpt-4o/spatio_temporal/20260421_210152_animal_kingdom.json \
  data/eval/gpt-4o/spatio_temporal/20260426_193730_american_football.json \
  data/eval/gpt-4o/spatio_temporal/20260421_214538_cholectrack20.json \
  data/eval/gpt-4o/spatio_temporal/20260421_225726_dota.json \
  data/eval/gpt-4o/spatio_temporal/20260421_231513_egosurgery.json \
  data/eval/gpt-4o/spatio_temporal/20260424_201036_enigma.json \
  data/eval/gpt-4o/spatio_temporal/20260422_000839_meccano.json \
  data/eval/gpt-4o/spatio_temporal/20260424_191528_mouse.json \
  data/eval/gpt-4o/spatio_temporal/20260422_005857_multisports.json \
  data/eval/gpt-4o/spatio_temporal/20260422_011826_uca.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gpt5_1 \
  --pred-jsons \
  data/eval/gpt5_1/spatio_temporal/20260422_083930_animal_kingdom.json \
  data/eval/gpt5_1/spatio_temporal/20260426_213638_american_football.json \
  data/eval/gpt5_1/spatio_temporal/20260422_085433_cholectrack20.json \
  data/eval/gpt5_1/spatio_temporal/20260422_090933_dota.json \
  data/eval/gpt5_1/spatio_temporal/20260422_092516_egosurgery.json \
  data/eval/gpt5_1/spatio_temporal/20260426_214058_enigma.json \
  data/eval/gpt5_1/spatio_temporal/20260422_094718_meccano.json \
  data/eval/gpt5_1/spatio_temporal/20260426_220600_mouse.json \
  data/eval/gpt5_1/spatio_temporal/20260422_100642_multisports.json \
  data/eval/gpt5_1/spatio_temporal/20260422_101652_uca.json


# OSS 2 shot

# domain    theoretical  q3-vl-4B  q3-vl-8B  q3.5-4B  q3.5-9B  eagle  iv3-8B  iv3.5-8B  iv3-14B
# sports    163          163       163       163       146       163    163     163        163
# surgery   216          216       216       216       190       216    216     216        216
# industry  169          169       169       169       169       169    169     169        169
# animal    157          157       157       157       157       157    157     157        157
# safety    250          250       250       250       250       250    250     250        250

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_vl_4B_instruct_2shot \
  --pred-jsons \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260504_031705_american_football_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260504_033150_enigma_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260504_051501_egosurgery_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260504_043133_mouse_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatio_temporal/20260504_041222_uca_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_vl_8B_instruct_2shot \
  --pred-jsons \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260504_064050_american_football_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260504_070027_enigma_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260504_085806_egosurgery_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260504_080853_mouse_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatio_temporal/20260504_074523_uca_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_5_4B_2shot \
  --pred-jsons \
  data/eval/qwen3_5_4B/spatio_temporal/20260504_022638_american_football_2shot.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260503_194529_enigma_2shot.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260503_233546_egosurgery_2shot.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260503_220554_mouse_2shot.json \
  data/eval/qwen3_5_4B/spatio_temporal/20260503_213523_uca_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model qwen3_5_9B_2shot \
  --pred-jsons \
  data/eval/qwen3_5_9B/spatio_temporal/20260502_211622_american_football_2shot.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260502_234542_enigma_2shot.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260502_211841_mouse_2shot.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260503_021401_uca_2shot.json \
  data/eval/qwen3_5_9B/spatio_temporal/20260502_223146_egosurgery_2shot.json


uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model eagle_2shot \
  --pred-jsons \
  data/eval/eagle_2shot/spatio_temporal/20260504_070757_american_football_2shot.json \
  data/eval/eagle_2shot/spatio_temporal/20260504_082134_egosurgery_2shot.json \
  data/eval/eagle_2shot/spatio_temporal/20260504_061851_enigma_2shot.json \
  data/eval/eagle_2shot/spatio_temporal/20260504_072253_mouse_2shot.json \
  data/eval/eagle_2shot/spatio_temporal/20260504_075954_uca_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model internvl3_8B_2shot \
  --pred-jsons \
  data/eval/internvl3_8B_2shot/spatio_temporal/20260503_150320_american_football_2shot.json \
  data/eval/internvl3_8B_2shot/spatio_temporal/20260503_141631_enigma_2shot.json \
  data/eval/internvl3_8B_2shot/spatio_temporal/20260503_161145_egosurgery_2shot.json \
  data/eval/internvl3_8B_2shot/spatio_temporal/20260503_151709_mouse_2shot.json \
  data/eval/internvl3_8B_2shot/spatio_temporal/20260503_155422_uca_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model internvl3_5_8B_2shot \
  --pred-jsons \
  data/eval/internvl3_5_8B_2shot/spatio_temporal/20260503_214539_american_football_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatio_temporal/20260503_182733_enigma_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatio_temporal/20260504_033215_egosurgery_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatio_temporal/20260503_233133_mouse_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatio_temporal/20260504_011748_uca_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model internvl3_14B_2shot \
  --pred-jsons \
  data/eval/internvl3_14B_2shot/spatio_temporal/20260503_064514_american_football_2shot.json \
  data/eval/internvl3_14B_2shot/spatio_temporal/20260503_045103_enigma_2shot.json \
  data/eval/internvl3_14B_2shot/spatio_temporal/20260503_100802_egosurgery_2shot.json \
  data/eval/internvl3_14B_2shot/spatio_temporal/20260503_072209_mouse_2shot.json \
  data/eval/internvl3_14B_2shot/spatio_temporal/20260503_091634_uca_2shot.json


# Proprietary 2 shot
# domain    theoretical  gpt-4o  gpt5.1  gemini2.5-flash  gemini2.5-pro  gemini3-flash  gemini3.1-pro
# sports    163          163     163     163              163            163            163
# surgery   216          180     189     216              216            216            216
# industry  169          115     132     169              169            169            169
# animal    157          115     152     157              157            157            157
# safety    250          236     249     250              250            250            250

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_2_5_flash_2shot \
  --pred-jsons \
  data/eval/gemini_2_5_flash/spatio_temporal/20260503_034705_american_football_2shot.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260503_055115_mouse_2shot.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260503_080802_enigma_2shot.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260503_111529_uca_2shot.json \
  data/eval/gemini_2_5_flash/spatio_temporal/20260503_141025_egosurgery_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_2_5_pro_2shot \
  --pred-jsons \
  data/eval/gemini_2_5_pro/spatio_temporal/20260503_034730_american_football_2shot.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260503_055501_mouse_2shot.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260503_082105_enigma_2shot.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260503_114022_uca_2shot.json \
  data/eval/gemini_2_5_pro/spatio_temporal/20260503_144000_egosurgery_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_flash_preview_2shot \
  --pred-jsons \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260503_034802_american_football_2shot.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260503_054956_mouse_2shot.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260503_075105_enigma_2shot.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260503_110751_uca_2shot.json \
  data/eval/gemini_3_flash_preview/spatio_temporal/20260503_140503_egosurgery_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gemini_3_1_pro_preview_2shot \
  --pred-jsons \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260502_231455_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260503_013306_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260503_090213_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260502_205600_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatio_temporal/20260503_052913_uca_2shot.json 

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gpt-4o_2shot \
  --pred-jsons \
  data/eval/gpt-4o/spatio_temporal/20260503_114122_american_football_2shot.json \
  data/eval/gpt-4o/spatio_temporal/20260503_134207_mouse_2shot.json \
  data/eval/gpt-4o/spatio_temporal/20260503_164708_uca_2shot.json \
  data/eval/gpt-4o/spatio_temporal/20260503_180153_enigma_2shot.json \
  data/eval/gpt-4o/spatio_temporal/20260503_220233_egosurgery_2shot.json

uv --project envs/download run python -m scripts.eval.run_spatio_temporal_grounding_bundle \
  --model gpt5_1_2shot \
  --pred-jsons \
  data/eval/gpt5_1/spatio_temporal/20260502_213043_american_football_2shot.json \
  data/eval/gpt5_1/spatio_temporal/20260502_215911_mouse_2shot.json \
  data/eval/gpt5_1/spatio_temporal/20260502_225158_uca_2shot.json \
  data/eval/gpt5_1/spatio_temporal/20260503_114456_enigma_2shot.json \
  data/eval/gpt5_1/spatio_temporal/20260503_131150_egosurgery_2shot.json




  
