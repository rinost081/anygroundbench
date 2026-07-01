import json
from pathlib import Path

from run_temporal_grounding_bundle import (
    DATASET_TO_DOMAIN,
    METRICS,
    build_temporal_grouped_query_rows,
    ordered_grouped_summary_rows,
    summarize_grouped_rows,
    write_grouped_summary,
)


def infer_dataset_from_augmented_path(path: Path) -> str:
    stem = path.stem.replace("_with_per_query_metrics_temporal", "")
    if stem.endswith("_2shot"):
        stem = stem[: -len("_2shot")]
    for dataset in sorted(DATASET_TO_DOMAIN, key=len, reverse=True):
        if stem.endswith(f"_{dataset}"):
            return dataset
    raise ValueError(f"Cannot infer dataset from augmented temporal path: {path}")


def load_prediction_entries(model_name: str, output_dir: Path) -> list[dict]:
    entries = []
    for path in sorted(output_dir.glob("*_with_per_query_metrics_temporal.json")):
        dataset = infer_dataset_from_augmented_path(path)
        items = json.loads(path.read_text())
        for item in items:
            metric = item.get("metric", {})
            if not all(metric.get(name) is not None for name in METRICS):
                continue
            entries.append(
                {
                    "dataset": item.get("dataset") or item.get("dataset_name") or dataset,
                    "resolved_query_id": str(item.get("resolved_query_id") or item.get("query_id")),
                    "item": item,
                    "source_index": 0,
                    "model": model_name,
                }
            )
    return entries


def backfill_model_dir(model_dir: Path) -> Path | None:
    output_dir = model_dir / "outputs"
    summary_csv = output_dir / "temporal_summary.csv"
    if not summary_csv.exists():
        return None

    prediction_entries = load_prediction_entries(model_dir.name, output_dir)
    if not prediction_entries:
        return None

    query_rows = build_temporal_grouped_query_rows(model_dir.name, prediction_entries)
    grouped_rows = ordered_grouped_summary_rows(summarize_grouped_rows(query_rows))
    grouped_summary_csv = output_dir / "temporal_summary_grouped.csv"
    write_grouped_summary(grouped_summary_csv, grouped_rows)
    return grouped_summary_csv


def main() -> None:
    eval_root = Path("data/eval")
    for model_dir in sorted(eval_root.iterdir()):
        if not model_dir.is_dir():
            continue
        output_path = backfill_model_dir(model_dir)
        if output_path is not None:
            print(f"saved={output_path}")


if __name__ == "__main__":
    main()
