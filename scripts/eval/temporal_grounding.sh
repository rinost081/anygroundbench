#!/usr/bin/env bash
set -euo pipefail

# blind test 0shot
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_blind_test --pred-jsons \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_042348_american_football.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_012328_animal_kingdom.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_040502_cholectrack20.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_014329_dota.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_023127_egosurgery.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_021757_enigma.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_034537_meccano.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_011411_mouse.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_033312_multisports.json \
  data/eval/gemini_3_1_pro_preview_blind_test/temporal/20260504_022805_uca.json

# blind test 2shot
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_blind_test_2shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/temporal/20260503_231002_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/temporal/20260504_011751_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/temporal/20260503_235516_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/temporal/20260504_033312_multisports_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/temporal/20260504_030045_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_blind_test_2shot/temporal/20260503_222908_uca_2shot.json


# Ablation 1 shot
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_1shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview_1shot/temporal/20260504_234737_american_football_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/temporal/20260505_012431_egosurgery_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/temporal/20260505_050021_enigma_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/temporal/20260505_032816_mouse_1shot.json \
  data/eval/gemini_3_1_pro_preview_1shot/temporal/20260505_035923_uca_1shot.json


# Ablation 4 shot
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_4shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview/temporal/20260504_195136_american_football_4shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260504_195153_egosurgery_4shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260504_195121_enigma_4shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260504_195102_mouse_4shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260504_213138_uca_4shot.json


# Ablation retrieval (visual features)
# simrank a = 0
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_a_0 --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_0/temporal/20260505_045425_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/temporal/20260505_062211_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/temporal/20260505_053651_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/temporal/20260505_024829_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_0/temporal/20260505_043217_uca_2shot.json

# Ablation retrieval (text features)
# simrank a = 1
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_a_1 --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_1/temporal/20260505_125947_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/temporal/20260505_132036_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/temporal/20260505_120447_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/temporal/20260505_090327_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_1/temporal/20260505_123135_uca_2shot.json


# Ablation retrieval (random)
# simrank a: random
uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_a_random --pred-jsons \
  data/eval/gemini_3_1_pro_preview_a_random/temporal/20260505_190703_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/temporal/20260505_221945_egosurgery_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/temporal/20260505_203314_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/temporal/20260505_145810_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview_a_random/temporal/20260505_221745_uca_2shot.json


# 0 shot, proprietary & specialist
# dataset            theoretical  gemini-2.5-flash  gemini-2.5-pro  gemini-3  gemini-3.1-pro  gpt-4o  gpt-5.1  LLaVA-ST
# animal_kingdom     106          106               106             106       106             106     106      106
# american_football  37           37                37              37        37              37      37       37
# cholectrack20      37           37                37              37        37              37      37       37
# dota               231          231               230             231       230             231     231      231
# egosurgery         179          179               179             179       179             179     179      179
# enigma             58           58                58              58        58              58      58       58
# meccano            111          111               111             111       111             111     111      111
# mouse              51           51                51              51        51              51      51       51
# multisports        126          126               126             126       126             126     126      126
# uca                19           19                19              19        19              19      19       19

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_2_5_flash --pred-jsons \
  data/eval/gemini_2_5_flash/temporal/20260430_110749_animal_kingdom.json \
  data/eval/gemini_2_5_flash/temporal/20260430_081544_mouse.json \
  data/eval/gemini_2_5_flash/temporal/20260430_102751_multisports.json \
  data/eval/gemini_2_5_flash/temporal/20260430_080240_american_football.json \
  data/eval/gemini_2_5_flash/temporal/20260430_091527_uca.json \
  data/eval/gemini_2_5_flash/temporal/20260430_121901_dota.json \
  data/eval/gemini_2_5_flash/temporal/20260430_092440_egosurgery.json \
  data/eval/gemini_2_5_flash/temporal/20260430_163101_cholectrack20.json \
  data/eval/gemini_2_5_flash/temporal/20260430_114149_meccano.json \
  data/eval/gemini_2_5_flash/temporal/20260430_083610_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_2_5_pro --pred-jsons \
  data/eval/gemini_2_5_pro/temporal/20260430_132104_animal_kingdom.json \
  data/eval/gemini_2_5_pro/temporal/20260430_095819_mouse.json \
  data/eval/gemini_2_5_pro/temporal/20260430_123017_multisports.json \
  data/eval/gemini_2_5_pro/temporal/20260430_094251_american_football.json \
  data/eval/gemini_2_5_pro/temporal/20260430_110211_uca.json \
  data/eval/gemini_2_5_pro/temporal/20260430_144956_dota.json \
  data/eval/gemini_2_5_pro/temporal/20260430_111326_egosurgery.json \
  data/eval/gemini_2_5_pro/temporal/20260430_163925_cholectrack20.json \
  data/eval/gemini_2_5_pro/temporal/20260430_140327_meccano.json \
  data/eval/gemini_2_5_pro/temporal/20260430_101945_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_flash_preview --pred-jsons \
  data/eval/gemini_3_flash_preview/temporal/20260430_071458_animal_kingdom.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_043511_mouse.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_063212_multisports.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_042206_american_football.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_053309_uca.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_082523_dota.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_054042_egosurgery.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_123036_cholectrack20.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_074736_meccano.json \
  data/eval/gemini_3_flash_preview/temporal/20260430_045331_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview --pred-jsons \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_093711_animal_kingdom.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_064956_mouse.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_085224_multisports.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_063607_american_football.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_074407_uca.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_104857_dota.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_075152_egosurgery.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_122957_cholectrack20.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_100954_meccano.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260430_070813_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gpt-4o --pred-jsons \
  data/eval/gpt-4o/temporal/20260421_163607_animal_kingdom.json \
  data/eval/gpt-4o/temporal/20260424_164530_mouse.json \
  data/eval/gpt-4o/temporal/20260421_202320_multisports.json \
  data/eval/gpt-4o/temporal/20260424_163519_american_football.json \
  data/eval/gpt-4o/temporal/20260421_204310_uca.json \
  data/eval/gpt-4o/temporal/20260421_182134_dota.json \
  data/eval/gpt-4o/temporal/20260421_184021_egosurgery.json \
  data/eval/gpt-4o/temporal/20260421_171542_cholectrack20.json \
  data/eval/gpt-4o/temporal/20260421_193346_meccano.json \
  data/eval/gpt-4o/temporal/20260424_174030_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gpt5_1 --pred-jsons \
  data/eval/gpt5_1/temporal/20260426_224711_animal_kingdom.json \
  data/eval/gpt5_1/temporal/20260426_223909_mouse.json \
  data/eval/gpt5_1/temporal/20260422_082624_multisports.json \
  data/eval/gpt5_1/temporal/20260426_212802_american_football.json \
  data/eval/gpt5_1/temporal/20260422_083556_uca.json \
  data/eval/gpt5_1/temporal/20260422_074246_dota.json \
  data/eval/gpt5_1/temporal/20260422_075447_egosurgery.json \
  data/eval/gpt5_1/temporal/20260422_073019_cholectrack20.json \
  data/eval/gpt5_1/temporal/20260422_081123_meccano.json \
  data/eval/gpt5_1/temporal/20260426_221703_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model LLaVA-ST --pred-jsons \
  data/eval/LLaVA-ST/temporal/20260423_174336_animal_kingdom.json \
  data/eval/LLaVA-ST/temporal/20260424_175509_mouse.json \
  data/eval/LLaVA-ST/temporal/20260423_183510_multisports.json \
  data/eval/LLaVA-ST/temporal/20260424_172915_american_football.json \
  data/eval/LLaVA-ST/temporal/20260424_174020_uca.json \
  data/eval/LLaVA-ST/temporal/20260423_175412_dota.json \
  data/eval/LLaVA-ST/temporal/20260423_181009_egosurgery.json \
  data/eval/LLaVA-ST/temporal/20260423_175054_cholectrack20.json \
  data/eval/LLaVA-ST/temporal/20260423_182517_meccano.json \
  data/eval/LLaVA-ST/temporal/20260424_174233_enigma.json


# 0shot, OSS
# dataset            theoretical  q3-vl-4B  q3-vl-8B  q3.5-4B  q3.5-9B  eagle  iv3-8B   iv3.5-8B  iv3-14B
# animal_kingdom     106          106       106       106       106       106  106       106       106
# american_football  37           37        37        37        37        37    37        37        37
# cholectrack20      37           37        37        37        37        37    37        37        37
# dota               231          231       231       231       231       231  231       231       231
# egosurgery         179          179       179       179       179       179  179       179       179
# enigma             58           58        58        58        58        58    58        58        58
# meccano            111          111       111       111       111       111  111       111       111
# mouse              51           51        51        51        51        51    51        51        51
# multisports        126          126       126       126       126       126  125       126       126
# uca                19           19        19        19        19        19    19        19        19

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_vl_4B_instruct --pred-jsons \
  data/eval/qwen3_vl_4B_instruct/temporal/20260419_085353_animal_kingdom.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260423_215335_mouse.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260429_203407_multisports.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260423_215115_american_football.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260419_090756_uca.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260420_181113_dota.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260419_090024_egosurgery.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260421_083903_cholectrack20.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260420_181518_meccano.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260423_215509_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_vl_8B_instruct --pred-jsons \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_203531_animal_kingdom.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_205616_american_football.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260418_013151_uca.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_203723_cholectrack20.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260423_220240_mouse.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_203931_dota.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_204116_egosurgery.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_204432_meccano.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_204718_multisports.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260429_204900_enigma.json


uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_5_4B --pred-jsons \
  data/eval/qwen3_5_4B/temporal/20260429_221759_animal_kingdom.json \
  data/eval/qwen3_5_4B/temporal/20260429_224311_american_football.json \
  data/eval/qwen3_5_4B/temporal/20260429_221955_cholectrack20.json \
  data/eval/qwen3_5_4B/temporal/20260429_222212_dota.json \
  data/eval/qwen3_5_4B/temporal/20260429_222421_egosurgery.json \
  data/eval/qwen3_5_4B/temporal/20260429_223530_enigma.json \
  data/eval/qwen3_5_4B/temporal/20260429_222757_meccano.json \
  data/eval/qwen3_5_4B/temporal/20260429_223333_mouse.json \
  data/eval/qwen3_5_4B/temporal/20260429_223101_multisports.json \
  data/eval/qwen3_5_4B/temporal/20260429_223251_uca.json 


uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_5_9B --pred-jsons \
  data/eval/qwen3_5_9B/temporal/20260429_224424_animal_kingdom.json \
  data/eval/qwen3_5_9B/temporal/20260429_231444_american_football.json \
  data/eval/qwen3_5_9B/temporal/20260429_224843_cholectrack20.json \
  data/eval/qwen3_5_9B/temporal/20260429_225127_dota.json \
  data/eval/qwen3_5_9B/temporal/20260429_225359_egosurgery.json \
  data/eval/qwen3_5_9B/temporal/20260429_230727_enigma.json \
  data/eval/qwen3_5_9B/temporal/20260429_225807_meccano.json \
  data/eval/qwen3_5_9B/temporal/20260429_230508_mouse.json \
  data/eval/qwen3_5_9B/temporal/20260429_230136_multisports.json \
  data/eval/qwen3_5_9B/temporal/20260429_230352_uca.json 

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model internvl3_8B --pred-jsons \
  data/eval/internvl3_8B/temporal/20260419_120921_animal_kingdom.json \
  data/eval/internvl3_8B/temporal/20260429_205747_egosurgery.json \
  data/eval/internvl3_8B/temporal/20260429_210412_multisports.json \
  data/eval/internvl3_8B/temporal/20260429_211338_uca.json \
  data/eval/internvl3_8B/temporal/20260419_122518_meccano.json \
  data/eval/internvl3_8B/temporal/20260423_120420_cholectrack20.json \
  data/eval/internvl3_8B/temporal/20260423_120812_dota.json \
  data/eval/internvl3_8B/temporal/20260423_224839_american_football.json \
  data/eval/internvl3_8B/temporal/20260423_225019_mouse.json \
  data/eval/internvl3_8B/temporal/20260423_225356_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model internvl3_5_8B --pred-jsons \
  data/eval/internvl3_5_8B/temporal/20260423_210109_animal_kingdom.json \
  data/eval/internvl3_5_8B/temporal/20260423_214307_mouse.json \
  data/eval/internvl3_5_8B/temporal/20260429_215537_multisports.json \
  data/eval/internvl3_5_8B/temporal/20260423_212639_american_football.json \
  data/eval/internvl3_5_8B/temporal/20260423_205451_uca.json \
  data/eval/internvl3_5_8B/temporal/20260423_210954_dota.json \
  data/eval/internvl3_5_8B/temporal/20260429_211832_egosurgery.json \
  data/eval/internvl3_5_8B/temporal/20260423_210454_cholectrack20.json \
  data/eval/internvl3_5_8B/temporal/20260423_211854_meccano.json \
  data/eval/internvl3_5_8B/temporal/20260423_213242_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model internvl3_14B --pred-jsons \
  data/eval/internvl3_14B/temporal/20260419_151356_animal_kingdom.json \
  data/eval/internvl3_14B/temporal/20260423_231944_mouse.json \
  data/eval/internvl3_14B/temporal/20260419_155638_multisports.json \
  data/eval/internvl3_14B/temporal/20260423_231733_american_football.json \
  data/eval/internvl3_14B/temporal/20260419_160219_uca.json \
  data/eval/internvl3_14B/temporal/20260420_183041_dota.json \
  data/eval/internvl3_14B/temporal/20260419_153506_egosurgery.json \
  data/eval/internvl3_14B/temporal/20260421_074856_cholectrack20.json \
  data/eval/internvl3_14B/temporal/20260420_183749_meccano.json \
  data/eval/internvl3_14B/temporal/20260423_232553_enigma.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model eagle --pred-jsons \
  data/eval/eagle/temporal/20260501_161916_animal_kingdom.json \
  data/eval/eagle/temporal/20260501_151905_mouse.json \
  data/eval/eagle/temporal/20260501_161554_multisports.json \
  data/eval/eagle/temporal/20260501_151309_american_football.json \
  data/eval/eagle/temporal/20260501_153430_uca.json \
  data/eval/eagle/temporal/20260501_163324_dota.json \
  data/eval/eagle/temporal/20260501_153850_egosurgery.json \
  data/eval/eagle/temporal/20260501_163558_cholectrack20.json \
  data/eval/eagle/temporal/20260501_155724_meccano.json \
  data/eval/eagle/temporal/20260501_143201_enigma.json


# In-Context Learning

# domain    theoretical  iv3.5-8B  iv3-14B  iv3-8B  q3-vl-4B  q3-vl-8B  q3.5-4B  q3.5-9B  eagle
# sports    163          163       162      163     163       163       163       163       163
# surgery   216          216       179      216     216       216       216       216       216
# industry  169          128       128      169     169       169       169       169       169
# animal    157          157       118      157     157       157       157       157       157
# safety    250          250       242      250     250       250       250       250       250

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model internvl3_5_8B_2shot --pred-jsons \
  data/eval/internvl3_5_8B/temporal/20260426_021805_mouse_2shot.json \
  data/eval/internvl3_5_8B/temporal/20260426_024528_uca_2shot.json \
  data/eval/internvl3_5_8B/temporal/20260426_025657_egosurgery_2shot.json \
  data/eval/internvl3_5_8B/temporal/20260429_063710_american_football_2shot.json \
  data/eval/internvl3_5_8B/temporal/20260426_204855_enigma_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model internvl3_14B_2shot --pred-jsons \
  data/eval/internvl3_14B/temporal/20260426_001322_mouse_2shot.json \
  data/eval/internvl3_14B/temporal/20260426_004159_enigma_2shot.json \
  data/eval/internvl3_14B/temporal/20260426_012909_uca_2shot.json \
  data/eval/internvl3_14B/temporal/20260426_014319_egosurgery_2shot.json \
  data/eval/internvl3_14B/temporal/20260429_035652_american_football_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model internvl3_8B_2shot --pred-jsons \
  data/eval/internvl3_8B/temporal/20260425_222422_mouse_2shot.json \
  data/eval/internvl3_8B/temporal/20260425_224956_enigma_2shot.json \
  data/eval/internvl3_8B/temporal/20260425_233415_uca_2shot.json \
  data/eval/internvl3_8B/temporal/20260425_234337_egosurgery_2shot.json \
  data/eval/internvl3_8B/temporal/20260429_061022_american_football_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_vl_4B_instruct_2shot --pred-jsons \
  data/eval/qwen3_vl_4B_instruct/temporal/20260424_014050_american_football_2shot.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260424_092308_mouse_2shot.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260424_164551_enigma_2shot.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260424_093011_uca_2shot.json \
  data/eval/qwen3_vl_4B_instruct/temporal/20260424_093402_egosurgery_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_vl_8B_instruct_2shot --pred-jsons \
  data/eval/qwen3_vl_8B_instruct/temporal/20260424_014626_american_football_2shot.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260424_094427_mouse_2shot.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260424_095312_uca_2shot.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260424_095753_egosurgery_2shot.json \
  data/eval/qwen3_vl_8B_instruct/temporal/20260424_171404_enigma_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_5_4B_2shot --pred-jsons \
  data/eval/qwen3_5_4B/temporal/20260428_231628_mouse_2shot.json \
  data/eval/qwen3_5_4B/temporal/20260428_232822_uca_2shot.json \
  data/eval/qwen3_5_4B/temporal/20260428_233419_enigma_2shot.json \
  data/eval/qwen3_5_4B/temporal/20260429_000412_american_football_2shot.json \
  data/eval/qwen3_5_4B/temporal/20260429_001054_egosurgery_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model qwen3_5_9B_2shot --pred-jsons \
  data/eval/qwen3_5_9B/temporal/20260429_002435_mouse_2shot.json \
  data/eval/qwen3_5_9B/temporal/20260429_003550_uca_2shot.json \
  data/eval/qwen3_5_9B/temporal/20260429_004211_enigma_2shot.json \
  data/eval/qwen3_5_9B/temporal/20260429_011334_american_football_2shot.json \
  data/eval/qwen3_5_9B/temporal/20260429_012051_egosurgery_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model eagle_2shot --pred-jsons \
  data/eval/eagle/temporal/20260429_032205_mouse_2shot.json \
  data/eval/eagle/temporal/20260429_014315_uca_2shot.json \
  data/eval/eagle/temporal/20260429_020149_enigma_2shot.json \
  data/eval/eagle/temporal/20260429_024416_american_football_2shot.json \
  data/eval/eagle/temporal/20260429_025351_egosurgery_2shot.json


# 2shot, proprietary & specialist
# domain    theoretical  gemini-2.5-flash  gemini-2.5-pro  gemini-3-flash  gemini-3.1-pro  gpt-4o  gpt-5.1  LLaVA-ST
# sports    163          163               163             163              163              163     163      163
# surgery   216          216               216             216              216              180     189      -
# industry  169          169               169             169              169              115     132      169
# animal    157          157               157             157              157              115     152      157
# safety    250          250               250             250              250              236     250      250


uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_2_5_flash_2shot --pred-jsons \
  data/eval/gemini_2_5_flash/temporal/20260428_211943_mouse_2shot.json \
  data/eval/gemini_2_5_flash/temporal/20260428_231706_enigma_2shot.json \
  data/eval/gemini_2_5_flash/temporal/20260427_010249_american_football_2shot.json \
  data/eval/gemini_2_5_flash/temporal/20260429_022051_egosurgery_2shot.json \
  data/eval/gemini_2_5_flash/temporal/20260430_051524_uca_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_2_5_pro_2shot --pred-jsons \
  data/eval/gemini_2_5_pro/temporal/20260427_030557_mouse_2shot.json \
  data/eval/gemini_2_5_pro/temporal/20260427_084204_uca_2shot.json \
  data/eval/gemini_2_5_pro/temporal/20260427_052038_enigma_2shot.json \
  data/eval/gemini_2_5_pro/temporal/20260427_010300_american_football_2shot.json \
  data/eval/gemini_2_5_pro/temporal/20260430_054802_egosurgery_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_flash_preview_2shot --pred-jsons \
  data/eval/gemini_3_flash_preview/temporal/20260428_212813_mouse_2shot.json \
  data/eval/gemini_3_flash_preview/temporal/20260427_081531_uca_2shot.json \
  data/eval/gemini_3_flash_preview/temporal/20260428_231545_enigma_2shot.json \
  data/eval/gemini_3_flash_preview/temporal/20260429_050836_american_football_2shot.json \
  data/eval/gemini_3_flash_preview/temporal/20260429_022035_egosurgery_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gemini_3_1_pro_preview_2shot --pred-jsons \
  data/eval/gemini_3_1_pro_preview/temporal/20260428_212929_mouse_2shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260428_082506_uca_2shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260428_235535_enigma_2shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260429_061228_american_football_2shot.json \
  data/eval/gemini_3_1_pro_preview/temporal/20260429_031516_egosurgery_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gpt-4o_2shot --pred-jsons \
  data/eval/gpt-4o/temporal/20260430_144117_mouse_2shot.json \
  data/eval/gpt-4o/temporal/20260430_182549_uca_2shot.json \
  data/eval/gpt-4o/temporal/20260430_194801_enigma_2shot.json \
  data/eval/gpt-4o/temporal/20260501_003123_egosurgery_2shot.json \
  data/eval/gpt-4o/temporal/20260430_122838_american_football_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model gpt5_1_2shot --pred-jsons \
  data/eval/gpt5_1/temporal/20260430_030306_mouse_2shot.json \
  data/eval/gpt5_1/temporal/20260430_035549_uca_2shot.json \
  data/eval/gpt5_1/temporal/20260430_042708_enigma_2shot.json \
  data/eval/gpt5_1/temporal/20260430_023336_american_football_2shot.json \
  data/eval/gpt5_1/temporal/20260430_055149_egosurgery_2shot.json

uv run --project envs/download scripts/eval/run_temporal_grounding_bundle.py --model LLaVA-ST_2shot --pred-jsons \
  data/eval/LLaVA-ST/temporal/20260427_022629_mouse_2shot.json \
  data/eval/LLaVA-ST/temporal/20260427_023505_uca_2shot.json \
  data/eval/LLaVA-ST/temporal/20260427_024729_enigma_2shot.json \
  data/eval/LLaVA-ST/temporal/20260427_020359_american_football_2shot.json
