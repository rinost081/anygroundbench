import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.temporal_grounding import extract_time, iou

DATASET_TO_DOMAIN = {
    "animal_kingdom": "animal",
    "mouse": "animal",
    "meccano": "industry",
    "enigma": "industry",
    "multisports": "sports",
    "american_football": "sports",
    "egosurgery": "surgery",
    "cholectrack20": "surgery",
    "uca": "safety",
    "dota": "safety",
}

DOMAIN_DATASET_ORDER = [
    ("animal", "animal_kingdom"),
    ("animal", "mouse"),
    ("industry", "meccano"),
    ("industry", "enigma"),
    ("sports", "multisports"),
    ("sports", "american_football"),
    ("surgery", "egosurgery"),
    ("surgery", "cholectrack20"),
    ("safety", "uca"),
    ("safety", "dota"),
]
DOMAINS = tuple(dict.fromkeys(domain for domain, _dataset in DOMAIN_DATASET_ORDER))

METRICS = ["iou_0_3", "iou_0_5", "iou_0_7", "miou"]
DATASET_PATH_ALIASES = {
    "multisports": "MultiSports",
    "dota": "DoTA",
    "meccano": "MECCANO",
    "cholectrack20": "CholecTrack20",
    "enigma": "ENIGMA",
}
GROUP_KEYS = ["group", "category"]
VIDEO_DURATION_BINS = [
    (0.0, 30.0, "<30s"),
    (30.0, 60.0, "30-60s"),
    (60.0, 180.0, "1-3min"),
    (180.0, 600.0, "3-10min"),
]
GT_DURATION_BINS = [
    (0.0, 3.0, "<3s"),
    (3.0, 10.0, "3-10s"),
    (10.0, 60.0, "10-60s"),
]
VIDEO_DURATION_CACHE: dict[str, float] = {}


def get_shot_suffix(stem: str) -> str | None:
    """ファイル名末尾の shot suffix を返す。"""
    for suffix in ("_1shot", "_2shot", "_4shot"):
        if stem.endswith(suffix):
            return suffix
    return None


def infer_prediction_target(path: Path) -> tuple[str, str]:
    """予測 JSON のファイル名末尾から dataset または domain 名を推定する。"""
    stem = path.stem
    shot_suffix = get_shot_suffix(stem)
    if shot_suffix is not None:
        stem = stem[: -len(shot_suffix)]
    for dataset in sorted(DATASET_TO_DOMAIN, key=len, reverse=True):
        if stem.endswith(f"_{dataset}"):
            return "dataset", dataset
    for domain in sorted(DOMAINS, key=len, reverse=True):
        if stem.endswith(f"_{domain}"):
            return "domain", domain
    raise ValueError(f"Cannot infer dataset or domain from prediction path: {path}")


def domain_for_target(target: str) -> str:
    """dataset/domain 名から domain 名を返す。"""
    return target if target in DOMAINS else DATASET_TO_DOMAIN[target]


def is_domain_target(target: str) -> bool:
    """評価対象名が domain かどうかを返す。"""
    return target in DOMAINS


def parse_log(path: Path) -> dict[str, float]:
    """評価ログから IoU 系メトリクスを抽出する。"""
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("IOU 0.3:"):
            values["iou_0_3"] = float(line.split(":", 1)[1])
        elif line.startswith("IOU 0.5:"):
            values["iou_0_5"] = float(line.split(":", 1)[1])
        elif line.startswith("IOU 0.7:"):
            values["iou_0_7"] = float(line.split(":", 1)[1])
        elif line.startswith("mIOU:"):
            values["miou"] = float(line.split(":", 1)[1])

    missing = [metric for metric in METRICS if metric not in values]
    if missing:
        raise ValueError(f"Missing metrics {missing} in {path}")
    return values


def load_metadata(dataset: str) -> dict:
    """指定 dataset の temporal metadata を読む。"""
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    metadata_path = Path("data") / dataset_dir / "meta-data" / "t_test.json"
    return json.loads(metadata_path.read_text())


def resolve_query_video_path(item: dict, dataset: str) -> Path:
    """予測 item からクエリ動画ファイルの実パスを組み立てる。"""
    dataset_dir = DATASET_PATH_ALIASES.get(dataset, dataset)
    query_video_path = item.get("query_video_path") or item.get("query")
    video_dir = str(item["split_num"]) if "split_num" in item else "clips"
    return Path("data") / dataset_dir / "videos" / video_dir / query_video_path


def get_video_duration_seconds(video_path: Path) -> float:
    """動画長を秒単位で返す。"""
    cache_key = str(video_path)
    if cache_key in VIDEO_DURATION_CACHE:
        return VIDEO_DURATION_CACHE[cache_key]

    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    video_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 0.0
    duration_sec = len(vr) / video_fps if video_fps > 0 else 0.0
    VIDEO_DURATION_CACHE[cache_key] = duration_sec
    return duration_sec


def categorize_duration(value: float, bins: list[tuple[float, float, str]]) -> str | None:
    """半開区間 bins に基づいてカテゴリ名を返す。"""
    for lower, upper, label in bins:
        if lower <= value < upper:
            return label
    return None


def build_match_index(metadata: dict) -> dict:
    """query_id に依存しない照合キーから metadata と query_id を引ける index を作る。"""
    index = {}
    for query_id, sample in metadata.items():
        video_name = sample["video_name"]
        text = sample["text"]
        temporal_range = sample["temporal_range"]
        record = {"query_id": str(query_id), "sample": sample}
        index[video_name] = record
        index[f"{video_name}.mp4"] = record
        index[(video_name, temporal_range)] = record
        index[(f"{video_name}.mp4", temporal_range)] = record
        index[(video_name, text)] = record
        index[(f"{video_name}.mp4", text)] = record
    return index


def find_dataset_for_item(
    item: dict,
    candidate_indices: dict[str, dict],
    query_text: str | None = None,
) -> tuple[str, str]:
    """1 prediction item の所属 dataset と dataset 内 query_id を metadata から判定する。"""
    gt = item.get("gt") or item.get("ground_truth")
    query_video_path = item.get("query_video_path") or item.get("query")
    video_stem = Path(query_video_path).stem
    candidate_keys = []
    if query_text is not None:
        candidate_keys.extend([(query_video_path, query_text), (video_stem, query_text)])
    candidate_keys.extend([(query_video_path, gt), (video_stem, gt)])

    matched_records = []
    for dataset, index in candidate_indices.items():
        for key in candidate_keys:
            if key in index:
                matched_records.append((dataset, index[key]["query_id"]))
                break

    if len(matched_records) == 1:
        return matched_records[0]
    if not matched_records:
        raise KeyError(f"Cannot match prediction item to dataset: {item}")
    raise ValueError(f"Ambiguous dataset match {matched_records} for item: {item}")


def split_predictions_by_dataset(pred_json: Path) -> dict[str, list[dict]]:
    """Prediction JSON を domain 内の dataset ごとに分解する。"""
    target_type, target = infer_prediction_target(pred_json)
    predictions = json.loads(pred_json.read_text())

    if target_type == "domain":
        return {target: predictions}

    domain = DATASET_TO_DOMAIN[target]
    if get_shot_suffix(pred_json.stem) is None:
        return {target: predictions}

    candidate_datasets = [
        dataset for row_domain, dataset in DOMAIN_DATASET_ORDER if row_domain == domain
    ]

    if len(candidate_datasets) == 1:
        return {candidate_datasets[0]: predictions}

    candidate_indices = {
        dataset: build_match_index(load_metadata(dataset))
        for dataset in candidate_datasets
    }
    domain_metadata = load_metadata(domain)
    split_predictions: dict[str, list[dict]] = {dataset: [] for dataset in candidate_datasets}
    for item in predictions:
        query_text = domain_metadata[str(item["query_id"])]["text"]
        dataset, dataset_query_id = find_dataset_for_item(
            item,
            candidate_indices,
            query_text=query_text,
        )
        normalized_item = dict(item)
        normalized_item["query_id"] = dataset_query_id
        split_predictions[dataset].append(normalized_item)
    return {
        dataset: items for dataset, items in split_predictions.items() if items
    }


def build_prediction_entries(pred_json: Path) -> list[dict]:
    """元 prediction の各要素に source index / dataset / 解決後 query_id を付ける。"""
    target_type, target = infer_prediction_target(pred_json)
    predictions = json.loads(pred_json.read_text())

    if target_type == "domain" or get_shot_suffix(pred_json.stem) is None:
        return [
            {
                "source_index": index,
                "dataset": target,
                "resolved_query_id": str(item["query_id"]),
                "item": item,
            }
            for index, item in enumerate(predictions)
        ]

    domain = DATASET_TO_DOMAIN[target]
    candidate_datasets = [
        dataset for row_domain, dataset in DOMAIN_DATASET_ORDER if row_domain == domain
    ]

    if len(candidate_datasets) == 1:
        return [
            {
                "source_index": index,
                "dataset": candidate_datasets[0],
                "resolved_query_id": str(item["query_id"]),
                "item": item,
            }
            for index, item in enumerate(predictions)
        ]

    candidate_indices = {
        dataset: build_match_index(load_metadata(dataset))
        for dataset in candidate_datasets
    }
    domain_metadata = load_metadata(domain)
    entries = []
    for index, item in enumerate(predictions):
        query_text = domain_metadata[str(item["query_id"])]["text"]
        dataset, dataset_query_id = find_dataset_for_item(
            item,
            candidate_indices,
            query_text=query_text,
        )
        entries.append(
            {
                "source_index": index,
                "dataset": dataset,
                "resolved_query_id": str(dataset_query_id),
                "item": item,
            }
        )
    return entries


def build_prediction_text(item: dict) -> str:
    """prediction/pred/raw_pred の順で temporal 予測本文を取り出す。"""
    prediction = item.get("prediction") or item.get("pred")
    if isinstance(prediction, list):
        return prediction[0] if prediction else ""
    if isinstance(prediction, str):
        return prediction
    return item.get("raw_pred") or ""


def parse_gt_span(item: dict) -> tuple[float, float]:
    """予測 item が持つ GT 時間区間を float tuple へ変換する。"""
    gt = item.get("gt") or item.get("ground_truth")
    if isinstance(gt, str):
        gt = json.loads(gt)
    return float(gt[0]), float(gt[1])


def compute_per_query_metric(item: dict) -> dict[str, float]:
    """1 件の temporal 予測に対する per-query metric を計算する。"""
    gt_span = parse_gt_span(item)
    timestamps = extract_time(build_prediction_text(item))
    pred_span = timestamps[0] if timestamps else (-100.0, -100.0)
    miou = float(iou(gt_span, pred_span))
    return {
        "iou_0_3": float(miou >= 0.3),
        "iou_0_5": float(miou >= 0.5),
        "iou_0_7": float(miou >= 0.7),
        "miou": miou,
    }


def get_query_text(dataset: str, resolved_query_id: str) -> str:
    """metadata 上の query text を返す。"""
    metadata = load_metadata(dataset)
    return str(metadata[resolved_query_id]["text"])


def write_per_query_augmented_json(
    pred_json: Path,
    output_dir: Path,
    per_index_payload: dict[int, dict],
    prediction_entries: list[dict] | None = None,
) -> list[Path]:
    """元 prediction JSON に per-query metric を付けた派生 JSON を保存する。"""
    original_items = json.loads(pred_json.read_text())
    if "_1shot" not in pred_json.stem and "_2shot" not in pred_json.stem and "_4shot" not in pred_json.stem:
        augmented_items = []
        for index, item in enumerate(original_items):
            augmented_item = dict(item)
            augmented_item.update(per_index_payload[index])
            augmented_items.append(augmented_item)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pred_json.stem}_with_per_query_metrics_temporal.json"
        output_path.write_text(json.dumps(augmented_items, ensure_ascii=False, indent=2) + "\n")
        return [output_path]

    items_by_dataset: dict[str, list[dict]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pred_json.stem.split("_", 2)[:2]
    timestamp_prefix = "_".join(timestamp)
    output_paths = []
    for entry in prediction_entries or []:
        index = entry["source_index"]
        augmented_item = dict(original_items[index])
        augmented_item["dataset"] = entry["dataset"]
        augmented_item["query_id"] = entry["resolved_query_id"]
        augmented_item["resolved_query_id"] = entry["resolved_query_id"]
        augmented_item.update(per_index_payload[index])
        items_by_dataset.setdefault(entry["dataset"], []).append(augmented_item)

    for dataset, augmented_items in items_by_dataset.items():
        shot_suffix = "1shot" if "_1shot" in pred_json.stem else ("2shot" if "_2shot" in pred_json.stem else "4shot")
        output_path = output_dir / f"{timestamp_prefix}_{dataset}_{shot_suffix}_with_per_query_metrics_temporal.json"
        output_path.write_text(
            json.dumps(augmented_items, ensure_ascii=False, indent=2) + "\n"
        )
        output_paths.append(output_path)
    return output_paths


def build_temporal_grouped_query_rows(
    model: str,
    prediction_entries: list[dict],
) -> list[dict]:
    """per-query metric と duration group を持つ query row を作る。"""
    rows = []
    for entry in prediction_entries:
        item = entry["item"]
        dataset = entry["dataset"]
        video_path = resolve_query_video_path(item, dataset)
        video_duration = get_video_duration_seconds(video_path)
        gt_start, gt_end = parse_gt_span(item)
        gt_duration = gt_end - gt_start
        video_category = categorize_duration(video_duration, VIDEO_DURATION_BINS)
        gt_category = categorize_duration(gt_duration, GT_DURATION_BINS)
        metric = compute_per_query_metric(item)
        base = {
            "model": model,
            "domain": domain_for_target(dataset),
            "row_type": "domain" if is_domain_target(dataset) else "query",
            "dataset": "ALL" if is_domain_target(dataset) else dataset,
            "n": 1,
            **metric,
        }
        rows.append(
            {
                **base,
                "group": "overall",
                "category": "overall",
            }
        )
        if video_category is not None:
            rows.append(
                {
                    **base,
                    "group": "video duration",
                    "category": video_category,
                }
            )
        if gt_category is not None:
            rows.append(
                {
                    **base,
                    "group": "gt duration",
                    "category": gt_category,
                }
            )
    return rows


def run_temporal_eval(
    model: str,
    pred_json: Path,
    fps: float,
    query_max_frames_num: int,
    per_query_output_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Prediction JSON を必要に応じて dataset ごとに分解して評価する。"""
    dataset_predictions = split_predictions_by_dataset(pred_json)
    prediction_entries = build_prediction_entries(pred_json)
    rows = []
    bundle_dir = Path("data/eval") / model / "temporal_bundle_inputs"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stem_base = pred_json.stem.replace("_1shot", "").replace("_2shot", "").replace("_4shot", "")

    per_index_payload: dict[int, dict] = {}
    for entry in prediction_entries:
        per_index_payload[entry["source_index"]] = {
            "text": get_query_text(
                dataset=entry["dataset"],
                resolved_query_id=entry["resolved_query_id"],
            ),
            "metric": compute_per_query_metric(entry["item"])
        }

    if per_query_output_dir is not None:
        augmented_paths = write_per_query_augmented_json(
            pred_json,
            per_query_output_dir,
            per_index_payload=per_index_payload,
            prediction_entries=prediction_entries,
        )
        for augmented_path in augmented_paths:
            print(f"saved_per_query={augmented_path}")

    grouped_query_rows = build_temporal_grouped_query_rows(model, prediction_entries)

    for dataset, predictions in dataset_predictions.items():
        split_pred_json = bundle_dir / f"{stem_base}_{dataset}.json"
        split_pred_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n")
        output_json = (
            Path("data/eval")
            / model
            / "temporal_eval"
            / f"{split_pred_json.stem}_eval.json"
        )

        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                "envs/download",
                "python",
                "scripts/eval/convert_temporal_predictions_to_eval_json.py",
                "--pred-json",
                str(split_pred_json),
                "--dataset",
                dataset,
                "--output-json",
                str(output_json),
                "--model",
                model,
                "--fps",
                str(fps),
                "--query-max-frames-num",
                str(query_max_frames_num),
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                "envs/download",
                "python",
                "src/eval/temporal_grounding.py",
                "-f",
                str(output_json),
            ],
            check=True,
        )

        n = len(json.loads(output_json.read_text()))
        metrics = parse_log(output_json.with_suffix(".log"))
        rows.append(
            {
                "model": model,
                "domain": domain_for_target(dataset),
                "row_type": "domain" if is_domain_target(dataset) else "dataset",
                "dataset": "ALL" if is_domain_target(dataset) else dataset,
                "n": n,
                **metrics,
            }
        )
    return rows, grouped_query_rows


def domain_weighted_row(model: str, domain: str, rows: list[dict]) -> dict:
    """同一 domain 内の dataset 行を件数重み付きで集約する。"""
    total_n = sum(row["n"] for row in rows)
    weighted = {
        metric: sum(row[metric] * row["n"] for row in rows) / total_n
        for metric in METRICS
    }
    return {
        "model": model,
        "domain": domain,
        "row_type": "domain_weighted",
        "dataset": "ALL",
        "n": total_n,
        **weighted,
    }


def ordered_summary_rows(model: str, dataset_rows: list[dict]) -> list[dict]:
    """Dataset 行と domain 集約行を所定順で並べる。"""
    rows_by_dataset = {row["dataset"]: row for row in dataset_rows}
    output = []
    for domain in dict.fromkeys(domain for domain, _ in DOMAIN_DATASET_ORDER):
        direct_domain_rows = [
            row for row in dataset_rows
            if row["domain"] == domain and row["row_type"] == "domain"
        ]
        if direct_domain_rows:
            output.extend(direct_domain_rows)
            continue
        domain_rows = []
        for row_domain, dataset in DOMAIN_DATASET_ORDER:
            if row_domain != domain:
                continue
            row = rows_by_dataset.get(dataset)
            if row is None:
                continue
            output.append(row)
            domain_rows.append(row)
        if domain_rows:
            output.append(domain_weighted_row(model, domain, domain_rows))
    return output


def write_summary(path: Path, rows: list[dict]) -> None:
    """Summary CSV を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "domain", "row_type", "dataset", "n", *METRICS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "iou_0_3": f"{row['iou_0_3']:.6f}",
                    "iou_0_5": f"{row['iou_0_5']:.6f}",
                    "iou_0_7": f"{row['iou_0_7']:.6f}",
                    "miou": f"{row['miou']:.6f}",
                }
            )


def summarize_grouped_rows(query_rows: list[dict]) -> list[dict]:
    """query row を dataset/group/category 単位に平均集約する。"""
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in query_rows:
        key = (row["domain"], row["row_type"], row["dataset"], row["group"], row["category"])
        grouped.setdefault(key, []).append(row)

    dataset_rows = []
    for (domain, row_type, dataset, group, category), rows in grouped.items():
        n = len(rows)
        dataset_rows.append(
            {
                "model": rows[0]["model"],
                "domain": domain,
                "row_type": row_type,
                "dataset": dataset,
                "group": group,
                "category": category,
                "n": n,
                **{
                    metric: sum(row[metric] for row in rows) / n
                    for metric in METRICS
                },
            }
        )
    return dataset_rows


def domain_weighted_grouped_rows(dataset_rows: list[dict]) -> list[dict]:
    """同一 domain 内の grouped dataset row を件数重み付きで集約する。"""
    output = []
    dataset_rows = [row for row in dataset_rows if row["row_type"] != "domain"]
    for domain in dict.fromkeys(domain for domain, _ in DOMAIN_DATASET_ORDER):
        domain_rows = [row for row in dataset_rows if row["domain"] == domain]
        if not domain_rows:
            continue
        combos = {
            tuple(row[key] for key in GROUP_KEYS)
            for row in domain_rows
        }
        for combo in combos:
            matched_rows = [
                row for row in domain_rows
                if tuple(row[key] for key in GROUP_KEYS) == combo
            ]
            total_n = sum(row["n"] for row in matched_rows)
            output.append(
                {
                    "model": matched_rows[0]["model"],
                    "domain": domain,
                    "row_type": "domain_weighted",
                    "dataset": "ALL",
                    "group": combo[0],
                    "category": combo[1],
                    "n": total_n,
                    **{
                        metric: sum(row[metric] * row["n"] for row in matched_rows) / total_n
                        for metric in METRICS
                    },
                }
            )
    return output


def ordered_grouped_summary_rows(dataset_rows: list[dict]) -> list[dict]:
    """overall を先頭に置いて grouped row を整列する。"""
    weighted_rows = domain_weighted_grouped_rows(dataset_rows)
    overall_rows = []
    detailed_rows = []
    for domain in dict.fromkeys(domain for domain, _ in DOMAIN_DATASET_ORDER):
        direct_domain_rows = [
            row for row in dataset_rows
            if row["domain"] == domain and row["row_type"] == "domain"
        ]
        if direct_domain_rows:
            overall_rows.extend(
                row for row in direct_domain_rows
                if row["group"] == "overall" and row["category"] == "overall"
            )
            detailed_rows.extend(
                row for row in direct_domain_rows
                if not (row["group"] == "overall" and row["category"] == "overall")
            )
            continue
        matched_weighted_rows = [row for row in weighted_rows if row["domain"] == domain]
        overall_rows.extend(
            row for row in matched_weighted_rows
            if row["group"] == "overall" and row["category"] == "overall"
        )
        for _row_domain, dataset in DOMAIN_DATASET_ORDER:
            if _row_domain != domain:
                continue
            matched_dataset_rows = [
                row for row in dataset_rows
                if row["domain"] == domain and row["dataset"] == dataset
            ]
            overall_rows.extend(
                row for row in matched_dataset_rows
                if row["group"] == "overall" and row["category"] == "overall"
            )
            detailed_rows.extend(
                row for row in matched_dataset_rows
                if not (row["group"] == "overall" and row["category"] == "overall")
            )
        detailed_rows.extend(
            row for row in matched_weighted_rows
            if not (row["group"] == "overall" and row["category"] == "overall")
        )
    return overall_rows + detailed_rows


def write_grouped_summary(path: Path, rows: list[dict]) -> None:
    """temporal grouped summary CSV を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "domain", "row_type", "dataset", "n", "group", "category", *METRICS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "iou_0_3": f"{row['iou_0_3']:.6f}",
                    "iou_0_5": f"{row['iou_0_5']:.6f}",
                    "iou_0_7": f"{row['iou_0_7']:.6f}",
                    "miou": f"{row['miou']:.6f}",
                }
            )


def main() -> None:
    """bundle された temporal prediction 群を dataset ごとに評価し summary を作る。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pred-jsons", nargs="+", required=True)
    parser.add_argument("--summary-csv")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--query-max-frames-num", type=int, default=120)
    parser.add_argument("--per-query-output-dir")
    args = parser.parse_args()

    pred_jsons = [Path(path) for path in args.pred_jsons]
    default_outputs_dir = Path("data/eval") / args.model / "outputs"
    per_query_output_dir = (
        Path(args.per_query_output_dir)
        if args.per_query_output_dir
        else default_outputs_dir
    )
    dataset_rows = []
    grouped_query_rows = []
    for path in pred_jsons:
        output_rows, output_grouped_query_rows = run_temporal_eval(
            args.model,
            path,
            fps=args.fps,
            query_max_frames_num=args.query_max_frames_num,
            per_query_output_dir=per_query_output_dir,
        )
        dataset_rows.extend(output_rows)
        grouped_query_rows.extend(output_grouped_query_rows)
    summary_rows = ordered_summary_rows(args.model, dataset_rows)
    grouped_summary_rows = ordered_grouped_summary_rows(summarize_grouped_rows(grouped_query_rows))

    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else default_outputs_dir / "temporal_summary.csv"
    )
    write_summary(summary_csv, summary_rows)
    print(f"saved={summary_csv}")
    grouped_summary_csv = summary_csv.with_name("temporal_summary_grouped.csv")
    write_grouped_summary(grouped_summary_csv, grouped_summary_rows)
    print(f"saved={grouped_summary_csv}")


if __name__ == "__main__":
    main()
