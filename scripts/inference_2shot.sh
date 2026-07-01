#!/usr/bin/env bash
set -euo pipefail

# gemini_2_5_flash
uv run --project envs/gemini python test.py --task temporal --domain_name animal --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name industry --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name safety --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name sports --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name surgery --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name animal --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name industry --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name safety --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name sports --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name surgery --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name animal --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name industry --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name safety --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name sports --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name surgery --model_name gemini --model_id gemini-2.5-flash --n_shot 2 --alpha 0.5 --device cuda:0

# gemini_2_5_pro
uv run --project envs/gemini python test.py --task temporal --domain_name animal --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name industry --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name safety --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name sports --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name surgery --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name animal --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name industry --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name safety --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name sports --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name surgery --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name animal --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name industry --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name safety --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name sports --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name surgery --model_name gemini --model_id gemini-2.5-pro --n_shot 2 --alpha 0.5 --device cuda:0

# gemini_3_flash_preview
uv run --project envs/gemini python test.py --task temporal --domain_name animal --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name industry --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name safety --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name sports --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name surgery --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name animal --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name industry --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name safety --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name sports --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name surgery --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name animal --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name industry --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name safety --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name sports --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name surgery --model_name gemini --model_id gemini-3-flash-preview --n_shot 2 --alpha 0.5 --device cuda:0

# gemini_3_1_pro_preview
uv run --project envs/gemini python test.py --task temporal --domain_name animal --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name industry --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name safety --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name sports --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task temporal --domain_name surgery --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name animal --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name industry --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name safety --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name sports --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatio_temporal --domain_name surgery --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name animal --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name industry --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name safety --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name sports --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gemini python test.py --task spatial --domain_name surgery --model_name gemini --model_id gemini-3.1-pro-preview --n_shot 2 --alpha 0.5 --device cuda:0

# gpt_4o
uv run --project envs/gpt python test.py --task temporal --domain_name animal --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name industry --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name safety --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name sports --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name surgery --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name animal --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name industry --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name safety --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name sports --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name surgery --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name animal --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name industry --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name safety --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name sports --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name surgery --model_name gpt --model_id gpt-4o --n_shot 2 --alpha 0.5 --device cuda:0

# gpt_5_1
uv run --project envs/gpt python test.py --task temporal --domain_name animal --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name industry --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name safety --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name sports --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task temporal --domain_name surgery --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name animal --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name industry --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name safety --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name sports --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatio_temporal --domain_name surgery --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name animal --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name industry --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name safety --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name sports --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/gpt python test.py --task spatial --domain_name surgery --model_name gpt --model_id gpt-5.1 --n_shot 2 --alpha 0.5 --device cuda:0

# llava_st
uv run --project envs/llava_st python test.py --task temporal --domain_name animal --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task temporal --domain_name industry --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task temporal --domain_name safety --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task temporal --domain_name sports --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task temporal --domain_name surgery --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatio_temporal --domain_name animal --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatio_temporal --domain_name industry --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatio_temporal --domain_name safety --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatio_temporal --domain_name sports --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatio_temporal --domain_name surgery --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatial --domain_name animal --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatial --domain_name industry --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatial --domain_name safety --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatial --domain_name sports --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/llava_st python test.py --task spatial --domain_name surgery --model_name llava_st --model_id appletea2333/LLaVA-ST-Qwen2-7B --n_shot 2 --alpha 0.5 --device cuda:0

# qwen3_vl_4B_instruct
uv run --project envs/qwen python test.py --task temporal --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name industry --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name safety --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name sports --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name surgery --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name industry --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name safety --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name sports --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name surgery --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name industry --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name safety --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name sports --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name surgery --model_name qwen3 --model_id Qwen/Qwen3-VL-4B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0

# qwen3_vl_8B_instruct
uv run --project envs/qwen python test.py --task temporal --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name industry --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name safety --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name sports --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task temporal --domain_name surgery --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name industry --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name safety --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name sports --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatio_temporal --domain_name surgery --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name animal --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name industry --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name safety --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name sports --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen python test.py --task spatial --domain_name surgery --model_name qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct --n_shot 2 --alpha 0.5 --device cuda:0

# qwen3_5_4B
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name animal --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name industry --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name safety --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name sports --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name surgery --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name animal --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name industry --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name safety --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name sports --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name surgery --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name animal --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name industry --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name safety --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name sports --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name surgery --model_name qwen3_5 --model_id Qwen/Qwen3.5-4B --n_shot 2 --alpha 0.5 --device cuda:0

# qwen3_5_9B
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name animal --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name industry --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name safety --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name sports --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task temporal --domain_name surgery --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name animal --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name industry --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name safety --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name sports --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatio_temporal --domain_name surgery --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name animal --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name industry --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name safety --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name sports --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/qwen3_5 python test.py --task spatial --domain_name surgery --model_name qwen3_5 --model_id Qwen/Qwen3.5-9B --n_shot 2 --alpha 0.5 --device cuda:0

# internvl3_8B
uv run --project envs/internvl python test.py --task temporal --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0

# internvl3_5_8B
uv run --project envs/internvl python test.py --task temporal --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task temporal --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatio_temporal --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl python test.py --task spatial --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3_5-8B-hf --n_shot 2 --alpha 0.5 --device cuda:0

# internvl3_14B
uv run --project envs/internvl14B python test.py --task temporal --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task temporal --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task temporal --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task temporal --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task temporal --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatio_temporal --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatio_temporal --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatio_temporal --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatio_temporal --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatio_temporal --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatial --domain_name animal --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatial --domain_name industry --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatial --domain_name safety --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatial --domain_name sports --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/internvl14B python test.py --task spatial --domain_name surgery --model_name internvl --model_id OpenGVLab/InternVL3-14B-hf --n_shot 2 --alpha 0.5 --device cuda:0

# eagle
uv run --project envs/eagle python test.py --task temporal --domain_name animal --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task temporal --domain_name industry --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task temporal --domain_name safety --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task temporal --domain_name sports --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task temporal --domain_name surgery --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatio_temporal --domain_name animal --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatio_temporal --domain_name industry --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatio_temporal --domain_name safety --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatio_temporal --domain_name sports --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatio_temporal --domain_name surgery --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatial --domain_name animal --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatial --domain_name industry --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatial --domain_name safety --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatial --domain_name sports --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
uv run --project envs/eagle python test.py --task spatial --domain_name surgery --model_name eagle --model_id nvidia/Eagle2.5-8B --n_shot 2 --alpha 0.5 --device cuda:0
