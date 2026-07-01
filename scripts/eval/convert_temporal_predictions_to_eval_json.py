import argparse
import json
import re
from pathlib import Path

DATASET_ALIASES = {
    "animal_kingdom": "animal_kingdom",
    "multisports": "MultiSports",
    "uca": "uca",
    "dota": "DoTA",
    "meccano": "MECCANO",
    "cholectrack20": "CholecTrack20",
    "american_football": "american_football",
    "mouse": "mouse",
    "enigma": "ENIGMA",
    "egosurgery": "egosurgery",
}

TWO_SHOT_DOMAIN_ALIASES = {
    "american_football": "sports",
    "mouse": "animal",
    "uca": "safety",
    "egosurgery": "surgery",
    "enigma": "industry",
}

FRAME_RANGE_PATTERNS = [
    re.compile(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$"),
    re.compile(r"(-?\d+)\s*-\s*(-?\d+)"),
    re.compile(r"(-?\d+)\s+to\s+(-?\d+)", re.IGNORECASE),
]
LLAVA_ST_TEMP_PATTERN = re.compile(r"<TEMP-(\d{3})>")

def build_metadata_index(metadata: dict) -> dict:
    """query_id や動画名・正解値の組から metadata を引ける索引を作る。"""
    index = {}
    for query_id, sample in metadata.items():
        video_name = sample["video_name"]
        text = sample["text"]
        temporal_range = sample["temporal_range"]
        index[str(query_id)] = sample
        index[(video_name, temporal_range)] = sample
        index[(f"{video_name}.mp4", temporal_range)] = sample
        index[(video_name, text)] = sample
        index[(f"{video_name}.mp4", text)] = sample
    return index


def find_sample(item: dict, metadata_index: dict) -> dict:
    """予測 item に対応する metadata 上のサンプルを索引から特定する。"""
    query_id = str(item["query_id"])
    if query_id in metadata_index:
        return metadata_index[query_id]

    gt = item.get("gt") or item.get("ground_truth")
    query_video_path = item.get("query_video_path") or item.get("query")
    video_stem = Path(query_video_path).stem
    for key in [(query_video_path, gt), (video_stem, gt)]:
        if key in metadata_index:
            return metadata_index[key]

    raise KeyError(f"Cannot find metadata sample for query_id={query_id}")

def parse_gpt_frame_range(answer: str) -> tuple[int, int] | None:
    """GPT の出力文字列からフレーム範囲を抽出する。"""
    answer = answer.strip()
    for pattern in FRAME_RANGE_PATTERNS:
        match = pattern.search(answer)
        if match is not None:
            start_idx = int(match.group(1))
            end_idx = int(match.group(2))
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            return start_idx, end_idx
    return None


def parse_llava_st_temp_range(answer: str) -> tuple[int, int] | None:
    """LLaVA-ST の出力文字列から TEMP トークンの範囲を抽出する。"""
    temp_tags = LLAVA_ST_TEMP_PATTERN.findall(answer)
    if len(temp_tags) < 2:
        return None

    start_idx = int(temp_tags[0])
    end_idx = int(temp_tags[1])
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    return start_idx, end_idx


def resolve_query_video_path(item: dict, dataset: str) -> Path:
    """予測 item からクエリ動画ファイルの実パスを組み立てる。

    legacy実装で表記揺れが生じしているため対応している
    """
    query_video_path = item.get("query_video_path") or item.get("query")
    video_dir = str(item["split_num"]) if "split_num" in item else "clips"
    return Path("data") / dataset / "videos" / video_dir / query_video_path


def convert_gpt_frame_range_to_timestamps(
    answer: str,
    item: dict,
    dataset: str,
    fps: float,
    max_frames: int,
) -> list[tuple[float, float]] | None:
    """GPT のフレーム範囲回答を秒単位の時間区間へ変換する。"""
    parsed_range = parse_gpt_frame_range(answer)
    if parsed_range is None:
        return None

    start_idx, end_idx = parsed_range
    video_path = resolve_query_video_path(item, dataset)

    # NOTE: 評価パイプラインを走らせるごとに再計算している.
    # 時間はそこまでかからないので毎回再計算させている。
    # 時間あればメタデータとしてキャッシュにして読むだけの実装にする
    from decord import VideoReader, cpu


    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        return None
    video_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 1.0
    if video_fps <= 0:
        video_fps = 1.0

    sample_every = max(1, int(round(video_fps / max(fps, 1e-6))))
    frame_indices = list(range(0, total_frames, sample_every))
    frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        return None

    start_idx = min(max(start_idx, 0), len(frame_indices) - 1)
    end_idx = min(max(end_idx, 0), len(frame_indices) - 1)
    start_sec = frame_indices[start_idx] / video_fps
    end_sec = frame_indices[end_idx] / video_fps
    return [(start_sec, end_sec)]


def convert_llava_st_temp_range_to_timestamps(
    answer: str,
    item: dict,
    dataset: str,
) -> list[tuple[float, float]] | None:
    """LLaVA-ST の TEMP トークン範囲を秒単位の時間区間へ変換する。"""
    parsed_range = parse_llava_st_temp_range(answer)
    if parsed_range is None:
        return None

    start_idx, end_idx = parsed_range
    video_path = resolve_query_video_path(item, dataset)
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    video_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 0.0
    duration_sec = len(vr) / video_fps if video_fps > 0 else 0.0
    if duration_sec <= 0:
        return None

    start_idx = min(max(start_idx, 0), 99)
    end_idx = min(max(end_idx, 0), 99)
    start_sec = duration_sec * start_idx / 99.0
    end_sec = duration_sec * end_idx / 99.0
    return [(start_sec, end_sec)]


def arg_parse() -> None:
    """コマンドライン引数を定義して返す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--query-max-frames-num", type=int, default=120)
    args = parser.parse_args()
    return args

def main(args):
    """予測 JSON を temporal 評価用 JSON に変換して保存する。"""
    dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
    pred_name = Path(args.pred_json).name
    if "_2shot" in pred_name:
        dataset = TWO_SHOT_DOMAIN_ALIASES.get(dataset, dataset)
    predictions = json.loads(Path(args.pred_json).read_text())
    metadata_path = Path("data") / dataset / "meta-data" / "t_test.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_index = build_metadata_index(metadata)

    output = {}
    empty_answers = 0
    for item in predictions:
        sample = find_sample(item, metadata_index)
        video = item.get("query_video_path") or item.get("query")
        text = sample["text"]
        gt = item.get("gt") or item.get("ground_truth")
        answer = item.get("raw_pred") or item.get("prediction") or item.get("pred") or ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not answer:
            empty_answers += 1
        key = f"{video}>>>{text}>>>{gt}"
        if args.model.startswith("gpt"):
            timestamps = convert_gpt_frame_range_to_timestamps(
                answer=answer,
                item=item,
                dataset=dataset,
                fps=args.fps,
                max_frames=args.query_max_frames_num,
            )
            if timestamps is None:
                output[key] = {"answers": answer}
            else:
                output[key] = {"timestamps": timestamps, "answers": answer}
        elif args.model.startswith("LLaVA-ST"):
            timestamps = convert_llava_st_temp_range_to_timestamps(
                answer=answer,
                item=item,
                dataset=dataset,
            )
            if timestamps is None:
                output[key] = {"answers": answer}
            else:
                output[key] = {"timestamps": timestamps, "answers": answer}
        else:
            output[key] = {"answers": answer}

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"saved={output_path}")
    print(f"items={len(output)}")
    print(f"empty_answers={empty_answers}")


if __name__ == "__main__":
    args = arg_parse()
    main(args)
