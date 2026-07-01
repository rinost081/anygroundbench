#!/usr/bin/env bash
set -euo pipefail

# Blind test 0 shot
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_blind_test --pred-jsons \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260503_230311_animal_kingdom.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260504_010643_american_football.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260504_005859_cholectrack20.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260503_221041_dota.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260503_233559_egosurgery.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260503_232108_enigma.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260504_003915_meccano.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260503_220026_mouse.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260504_001831_multisports.json \
  data/eval/gemini_3_1_pro_preview_blind_test/spatial/20260503_233140_uca.json

# Blind test 2 shot
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_blind_test_2shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/spatial/20260504_044503_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/spatial/20260504_055749_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/spatial/20260504_051845_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/spatial/20260504_020946_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/spatial/20260504_034229_uca_2shot.json


# Ablation 1 shot
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_1shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview_1shot/spatial/20260505_015136_american_football_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatial/20260504_232443_egosurgery_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatial/20260504_184715_enigma_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatial/20260504_170556_mouse_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/spatial/20260504_202809_uca_1shot.json


# Ablation 4 shot
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_4shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview/spatial/20260504_164018_american_football_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260504_164010_egosurgery_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260504_163949_enigma_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260504_163937_mouse_4shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260504_163959_uca_4shot.json


# Ablation retrieval (visual features)
# simrank a = 0
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_a_0 --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_0/spatial/20260505_030556_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatial/20260505_041302_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatial/20260505_034105_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatial/20260505_005332_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/spatial/20260505_054627_uca_2shot.json

# Ablation retrieval (text features)
# simrank a = 1
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_a_1 --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_1/spatial/20260505_110722_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatial/20260505_105006_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatial/20260505_101157_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatial/20260505_070836_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/spatial/20260505_093215_uca_2shot.json


# Ablation retrieval (random)
# simrank a: random
uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_a_random --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_random/spatial/20260505_171342_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatial/20260505_195812_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatial/20260505_184036_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatial/20260505_131237_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/spatial/20260505_185435_uca_2shot.json




# 0 shot, proprietary & specialist
# dataset            theoretical  gemini-2.5-flash  gemini-2.5-pro  gemini-3-flash  gemini-3.1-pro  gpt-4o  gpt-5.1  LLaVA-ST
# animal_kingdom     106          106               106             106              106               106     106      106
# american_football  37           37                37              37               37                37      37       37
# cholectrack20      37           37                37              37               37                37      37       37
# dota               231          231               231             231              231               231     231      231
# egosurgery         179          155               155             155              155               155     155      172
# enigma             58           58                58              58               58                58      58       58
# meccano            111          111               111             111              111               111     111      111
# mouse              51           51                51              51               51                51      51       51
# multisports        126          126               126             126              126               126     126      126
# uca                19           19                19              19               19                19      19       19

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_2_5_flash --pred-jsons \
  data/eval/gemini_2_5_flash/spatial/20260502_070748_animal_kingdom.json \
  data/eval/gemini_2_5_flash/spatial/20260502_033030_american_football.json \
  data/eval/gemini_2_5_flash/spatial/20260502_104026_cholectrack20.json \
  data/eval/gemini_2_5_flash/spatial/20260502_090355_dota.json \
  data/eval/gemini_2_5_flash/spatial/20260502_044831_egosurgery.json \
  data/eval/gemini_2_5_flash/spatial/20260502_040615_enigma.json \
  data/eval/gemini_2_5_flash/spatial/20260502_075148_meccano.json \
  data/eval/gemini_2_5_flash/spatial/20260502_034500_mouse.json \
  data/eval/gemini_2_5_flash/spatial/20260502_060257_multisports.json \
  data/eval/gemini_2_5_flash/spatial/20260502_044143_uca.json 

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_2_5_pro --pred-jsons \
  data/eval/gemini_2_5_pro/spatial/20260502_072230_animal_kingdom.json \
  data/eval/gemini_2_5_pro/spatial/20260502_033114_american_football.json \
  data/eval/gemini_2_5_pro/spatial/20260502_104029_cholectrack20.json \
  data/eval/gemini_2_5_pro/spatial/20260502_085733_dota.json \
  data/eval/gemini_2_5_pro/spatial/20260502_044900_egosurgery.json \
  data/eval/gemini_2_5_pro/spatial/20260502_041252_enigma.json \
  data/eval/gemini_2_5_pro/spatial/20260502_080425_meccano.json \
  data/eval/gemini_2_5_pro/spatial/20260502_034954_mouse.json \
  data/eval/gemini_2_5_pro/spatial/20260502_062506_multisports.json \
  data/eval/gemini_2_5_pro/spatial/20260502_043825_uca.json 

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_flash_preview --pred-jsons \
  data/eval/gemini_3_flash_preview/spatial/20260502_060344_animal_kingdom.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_033118_american_football.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_082723_cholectrack20.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_071348_dota.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_042418_egosurgery.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_035832_enigma.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_063721_meccano.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_034219_mouse.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_052014_multisports.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_041740_uca.json 

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview --pred-jsons \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_062459_animal_kingdom.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_033121_american_football.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_090151_cholectrack20.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_074042_dota.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_043151_egosurgery.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_040406_enigma.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_065848_meccano.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_034616_mouse.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_054838_multisports.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_042417_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gpt-4o --pred-jsons \
  data/eval/gpt-4o/spatial/20260502_015136_animal_kingdom.json \
  data/eval/gpt-4o/spatial/20260502_011402_american_football.json \
  data/eval/gpt-4o/spatial/20260502_021129_cholectrack20.json \
  data/eval/gpt-4o/spatial/20260502_020034_dota.json \
  data/eval/gpt-4o/spatial/20260502_012914_egosurgery.json \
  data/eval/gpt-4o/spatial/20260502_012459_enigma.json \
  data/eval/gpt-4o/spatial/20260502_022748_meccano.json \
  data/eval/gpt-4o/spatial/20260502_011601_mouse.json \
  data/eval/gpt-4o/spatial/20260502_014551_multisports.json \
  data/eval/gpt-4o/spatial/20260502_012212_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gpt5_1 --pred-jsons \
  data/eval/gpt5_1/spatial/20260502_040956_animal_kingdom.json \
  data/eval/gpt5_1/spatial/20260502_034343_american_football.json \
  data/eval/gpt5_1/spatial/20260502_042736_cholectrack20.json \
  data/eval/gpt5_1/spatial/20260502_041610_dota.json \
  data/eval/gpt5_1/spatial/20260502_035443_egosurgery.json \
  data/eval/gpt5_1/spatial/20260502_035121_enigma.json \
  data/eval/gpt5_1/spatial/20260502_043214_meccano.json \
  data/eval/gpt5_1/spatial/20260502_034546_mouse.json \
  data/eval/gpt5_1/spatial/20260502_040553_multisports.json \
  data/eval/gpt5_1/spatial/20260502_034939_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model LLaVA-ST --pred-jsons \
  data/eval/LLaVA-ST/spatial/20260423_213114_animal_kingdom.json \
  data/eval/LLaVA-ST/spatial/20260426_163045_american_football.json \
  data/eval/LLaVA-ST/spatial/20260423_221747_cholectrack20.json \
  data/eval/LLaVA-ST/spatial/20260423_223444_dota.json \
  data/eval/LLaVA-ST/spatial/20260424_001538_egosurgery.json \
  data/eval/LLaVA-ST/spatial/20260426_144100_enigma.json \
  data/eval/LLaVA-ST/spatial/20260424_013416_meccano.json \
  data/eval/LLaVA-ST/spatial/20260424_190539_mouse.json \
  data/eval/LLaVA-ST/spatial/20260424_022440_multisports.json \
  data/eval/LLaVA-ST/spatial/20260424_032039_uca.json


# 0shot, OSS
# dataset            theoretical  q3-vl-4B  q3-vl-8B  q3.5-4B  q3.5-9B  eagle  iv3-8B  iv3.5-8B  iv3-14B
# animal_kingdom     106          -         106       106      106      106    106     106        106
# american_football  37           37        37        37       37       37     37      37         37
# cholectrack20      37           37        37        37       37       37     37      37         37
# dota               231          231       231       231      231      231    231     231        231
# egosurgery         179          172       172       155      155      155    155     155        155
# enigma             58           58        58        58       58       58     58      58         58
# meccano            111          111       111       111      111      111    111     111        111
# mouse              51           51        51        51       51       51     51      51         51
# multisports        126          126       126       126      126      126    126     126        126
# uca                19           19        19        19       19       19     19      19         19

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_vl_4B_instruct --pred-jsons \
  data/eval/qwen3_vl_4B_instruct/spatial/20260421_110332_animal_kingdom.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260504_134446_american_football.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260421_114159_cholectrack20.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260421_124037_dota.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260421_160940_egosurgery.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260426_192822_enigma.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260421_135130_meccano.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260424_112347_mouse.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260421_135653_multisports.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260421_141232_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_vl_8B_instruct --pred-jsons \
  data/eval/qwen3_vl_8B_instruct/spatial/20260421_035005_animal_kingdom.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260426_195109_american_football.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260423_035017_cholectrack20.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260421_040026_dota.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260423_040707_egosurgery.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260426_194535_enigma.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260421_050104_meccano.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260424_113444_mouse.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260421_050629_multisports.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260421_050932_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_5_4B --pred-jsons \
  data/eval/qwen3_5_4B/spatial/20260502_181345_animal_kingdom.json \
  data/eval/qwen3_5_4B/spatial/20260502_184902_american_football.json \
  data/eval/qwen3_5_4B/spatial/20260502_183840_cholectrack20.json \
  data/eval/qwen3_5_4B/spatial/20260502_160541_dota.json \
  data/eval/qwen3_5_4B/spatial/20260502_143256_egosurgery.json \
  data/eval/qwen3_5_4B/spatial/20260502_140449_enigma.json \
  data/eval/qwen3_5_4B/spatial/20260502_160118_meccano.json \
  data/eval/qwen3_5_4B/spatial/20260502_142142_mouse.json \
  data/eval/qwen3_5_4B/spatial/20260502_152337_multisports.json \
  data/eval/qwen3_5_4B/spatial/20260502_141614_uca.json 

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_5_9B --pred-jsons \
  data/eval/qwen3_5_9B/spatial/20260502_085049_animal_kingdom.json \
  data/eval/qwen3_5_9B/spatial/20260502_092140_american_football.json \
  data/eval/qwen3_5_9B/spatial/20260502_090314_cholectrack20.json \
  data/eval/qwen3_5_9B/spatial/20260502_081218_dota.json \
  data/eval/qwen3_5_9B/spatial/20260502_070933_egosurgery.json \
  data/eval/qwen3_5_9B/spatial/20260502_062812_enigma.json \
  data/eval/qwen3_5_9B/spatial/20260502_080742_meccano.json \
  data/eval/qwen3_5_9B/spatial/20260502_065827_mouse.json \
  data/eval/qwen3_5_9B/spatial/20260502_080053_multisports.json \
  data/eval/qwen3_5_9B/spatial/20260502_065417_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model internvl3_8B --pred-jsons \
  data/eval/internvl3_8B/spatial/20260502_105459_animal_kingdom.json \
  data/eval/internvl3_8B/spatial/20260502_091657_american_football.json \
  data/eval/internvl3_8B/spatial/20260502_115146_cholectrack20.json \
  data/eval/internvl3_8B/spatial/20260502_112437_dota.json \
  data/eval/internvl3_8B/spatial/20260502_093604_egosurgery.json \
  data/eval/internvl3_8B/spatial/20260502_090345_enigma.json \
  data/eval/internvl3_8B/spatial/20260502_100019_meccano.json \
  data/eval/internvl3_8B/spatial/20260502_092435_mouse.json \
  data/eval/internvl3_8B/spatial/20260502_103218_multisports.json \
  data/eval/internvl3_8B/spatial/20260502_093352_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model internvl3_5_8B --pred-jsons \
  data/eval/internvl3_5_8B/spatial/20260502_134209_animal_kingdom.json \
  data/eval/internvl3_5_8B/spatial/20260502_120254_american_football.json \
  data/eval/internvl3_5_8B/spatial/20260502_141249_cholectrack20.json \
  data/eval/internvl3_5_8B/spatial/20260502_140106_dota.json \
  data/eval/internvl3_5_8B/spatial/20260502_122041_egosurgery.json \
  data/eval/internvl3_5_8B/spatial/20260502_115837_enigma.json \
  data/eval/internvl3_5_8B/spatial/20260502_124241_meccano.json \
  data/eval/internvl3_5_8B/spatial/20260502_120615_mouse.json \
  data/eval/internvl3_5_8B/spatial/20260502_133749_multisports.json \
  data/eval/internvl3_5_8B/spatial/20260502_121731_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model internvl3_14B --pred-jsons \
  data/eval/internvl3_14B/spatial/20260502_072701_animal_kingdom.json \
  data/eval/internvl3_14B/spatial/20260502_052604_american_football.json \
  data/eval/internvl3_14B/spatial/20260502_084725_cholectrack20.json \
  data/eval/internvl3_14B/spatial/20260502_075604_dota.json \
  data/eval/internvl3_14B/spatial/20260502_055027_egosurgery.json \
  data/eval/internvl3_14B/spatial/20260502_051246_enigma.json \
  data/eval/internvl3_14B/spatial/20260502_062738_meccano.json \
  data/eval/internvl3_14B/spatial/20260502_053425_mouse.json \
  data/eval/internvl3_14B/spatial/20260502_065542_multisports.json \
  data/eval/internvl3_14B/spatial/20260502_054433_uca.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model eagle --pred-jsons \
  data/eval/eagle/spatial/20260502_145250_animal_kingdom.json \
  data/eval/eagle/spatial/20260502_143639_american_football.json \
  data/eval/eagle/spatial/20260502_150157_cholectrack20.json \
  data/eval/eagle/spatial/20260502_145605_dota.json \
  data/eval/eagle/spatial/20260502_144026_egosurgery.json \
  data/eval/eagle/spatial/20260502_143437_enigma.json \
  data/eval/eagle/spatial/20260502_144738_meccano.json \
  data/eval/eagle/spatial/20260502_143759_mouse.json \
  data/eval/eagle/spatial/20260502_144920_multisports.json \
  data/eval/eagle/spatial/20260502_143937_uca.json 


# OSS In-Context Learning

# domain    theoretical  iv3.5-8B  iv3-14B  iv3-8B  q3-vl-4B  q3-vl-8B  q3.5-4B  q3.5-9B  eagle
# sports    163          163       163      163     163       163       163       163       163
# surgery   216          149       147      149     150       150       150       150       149
# industry  169          169       169      169     169       169       169       169       169
# animal    157          157       157      157     157       157       157       157       157
# safety    250          250       250      250     250       250       250       250       250

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model internvl3_5_8B_2shot --pred-jsons \
  data/eval/internvl3_5_8B_2shot/spatial/20260503_191247_american_football_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatial/20260504_024822_egosurgery_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatial/20260503_172327_enigma_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatial/20260503_201159_mouse_2shot.json \
  data/eval/internvl3_5_8B_2shot/spatial/20260503_223843_uca_2shot.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model internvl3_14B_2shot --pred-jsons \
  data/eval/internvl3_14B/spatial/20260502_172845_american_football_2shot.json \
  data/eval/internvl3_14B/spatial/20260503_130017_egosurgery_2shot.json \
  data/eval/internvl3_14B/spatial/20260502_150409_enigma_2shot.json \
  data/eval/internvl3_14B/spatial/20260503_033132_mouse_2shot.json \
  data/eval/internvl3_14B/spatial/20260503_064909_uca_2shot.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model internvl3_8B_2shot --pred-jsons \
  data/eval/internvl3_8B/spatial/20260503_152110_american_football_2shot.json \
  data/eval/internvl3_8B/spatial/20260503_164005_egosurgery_2shot.json \
  data/eval/internvl3_8B/spatial/20260503_150420_enigma_2shot.json \
  data/eval/internvl3_8B/spatial/20260503_153556_mouse_2shot.json \
  data/eval/internvl3_8B/spatial/20260503_161634_uca_2shot.json \

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_vl_4B_instruct_2shot --pred-jsons \
  data/eval/qwen3_vl_4B_instruct/spatial/20260503_034516_american_football_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260503_150258_egosurgery_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260503_041340_enigma_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260503_140548_mouse_2shot.json \
  data/eval/qwen3_vl_4B_instruct/spatial/20260503_122631_uca_2shot.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_vl_8B_instruct_2shot --pred-jsons \
  data/eval/qwen3_vl_8B_instruct/spatial/20260502_235038_american_football_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260503_024058_egosurgery_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260503_004943_enigma_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260503_015420_mouse_2shot.json \
  data/eval/qwen3_vl_8B_instruct/spatial/20260503_011250_uca_2shot.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_5_4B_2shot --pred-jsons \
  data/eval/qwen3_5_4B/spatial/20260503_184736_american_football_2shot.json \
  data/eval/qwen3_5_4B/spatial/20260503_180138_egosurgery_2shot.json \
  data/eval/qwen3_5_4B/spatial/20260503_155623_enigma_2shot.json \
  data/eval/qwen3_5_4B/spatial/20260503_172051_mouse_2shot.json \
  data/eval/qwen3_5_4B/spatial/20260503_164130_uca_2shot.json \

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model qwen3_5_9B_2shot --pred-jsons \
  data/eval/qwen3_5_9B/spatial/20260502_191220_enigma_2shot.json \
  data/eval/qwen3_5_9B/spatial/20260502_194919_uca_2shot.json \
  data/eval/qwen3_5_9B/spatial/20260502_205141_mouse_2shot.json \
  data/eval/qwen3_5_9B/spatial/20260502_215658_egosurgery_2shot.json \
  data/eval/qwen3_5_9B/spatial/20260502_225215_multisports_2shot.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model eagle_2shot --pred-jsons \
  data/eval/eagle_2shot/spatial/20260504_052100_american_football_2shot.json \
  data/eval/eagle_2shot/spatial/20260504_051054_enigma_2shot.json \
  data/eval/eagle_2shot/spatial/20260504_055251_egosurgery_2shot.json \
  data/eval/eagle_2shot/spatial/20260504_052841_mouse_2shot.json \
  data/eval/eagle_2shot/spatial/20260504_054134_uca_2shot.json


# 2shot, proprietary & specialist
# domain    theoretical  gemini-2.5-flash  gemini-2.5-pro  gemini-3-flash  gemini-3.1-pro  gpt-4o  gpt-5.1  LLaVA-ST
# sports    163          163               163             163              163               163     163      -
# surgery   216          148               149             149              149               146     149      -
# industry  169          169               169             169              169               169     169      -
# animal    157          157               157             157              157               157     157      -
# safety    250          250               250             249              250               250     250      -


uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_2_5_flash_2shot --pred-jsons \
  data/eval/gemini_2_5_flash/spatial/20260502_110125_american_football_2shot.json \
  data/eval/gemini_2_5_flash/spatial/20260502_181709_egosurgery_2shot.json \
  data/eval/gemini_2_5_flash/spatial/20260502_141114_enigma_2shot.json \
  data/eval/gemini_2_5_flash/spatial/20260502_123652_mouse_2shot.json \
  data/eval/gemini_2_5_flash/spatial/20260502_154752_uca_2shot.json 

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_2_5_pro_2shot --pred-jsons \
  data/eval/gemini_2_5_pro/spatial/20260502_110159_american_football_2shot.json \
  data/eval/gemini_2_5_pro/spatial/20260502_190618_egosurgery_2shot.json \
  data/eval/gemini_2_5_pro/spatial/20260502_142901_enigma_2shot.json \
  data/eval/gemini_2_5_pro/spatial/20260502_124615_mouse_2shot.json \
  data/eval/gemini_2_5_pro/spatial/20260502_162049_uca_2shot.json 

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_flash_preview_2shot --pred-jsons \
  data/eval/gemini_3_flash_preview/spatial/20260502_084400_american_football_2shot.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_165511_egosurgery_2shot.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_121551_enigma_2shot.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_103554_mouse_2shot.json \
  data/eval/gemini_3_flash_preview/spatial/20260502_140501_uca_2shot.json 

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gemini_3_1_pro_preview_2shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_091955_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_183629_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_130559_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_111424_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview/spatial/20260502_150938_uca_2shot.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gpt-4o_2shot --pred-jsons \
  data/eval/gpt-4o/spatial/20260501_221218_american_football_2shot.json \
  data/eval/gpt-4o/spatial/20260501_234922_egosurgery_2shot.json \
  data/eval/gpt-4o/spatial/20260501_233249_enigma_2shot.json \
  data/eval/gpt-4o/spatial/20260501_222350_mouse_2shot.json \
  data/eval/gpt-4o/spatial/20260501_230148_uca_2shot.json

uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model gpt5_1_2shot --pred-jsons \
  data/eval/gpt5_1/spatial/20260501_072551_american_football_2shot.json \
  data/eval/gpt5_1/spatial/20260501_081105_egosurgery_2shot.json \
  data/eval/gpt5_1/spatial/20260501_075947_enigma_2shot.json \
  data/eval/gpt5_1/spatial/20260501_221040_mouse_2shot.json \
  data/eval/gpt5_1/spatial/20260501_074228_uca_2shot.json

# uv run --project envs/download scripts/eval/run_spatio_temporal_grounding_bundle.py --task spatial --model LLaVA-ST_2shot --pred-jsons \
