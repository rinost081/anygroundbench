#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


SOURCE_DATASET = "animal_kingdom"
DOMAIN = "animal"
ANIMAL_KINGDOM_VIDEO_ID = re.compile(r"^[A-Z]{8}$")
FULL_VIDEO_DIR = Path("data/animal_kingdom/dataset")
OUT_ROOT = Path("data")


@dataclass
class RequiredVideo:
    kind: str
    video_name: str
    target_path: Path
    source_candidates: list[Path]
    metadata_files: set[str] = field(default_factory=set)
    query_ids: set[str] = field(default_factory=set)
    resolved_source_path: Path | None = None
    status: str = "pending"
    notes: list[str] = field(default_factory=list)

    def key(self) -> tuple[str, str]:
        return self.kind, self.video_name



def canonical_full_video_name(entry: dict) -> str | None:
    video_name = entry.get("video_name")
    if isinstance(video_name, str):
        video_id = video_name[:-4] if video_name.endswith(".mp4") else video_name
        if ANIMAL_KINGDOM_VIDEO_ID.match(video_id):
            return video_id

    meta_info = entry.get("meta_info")
    if isinstance(meta_info, dict):
        for key in ("source_video_name", "video_name"):
            value = meta_info.get(key)
            if isinstance(value, str):
                video_id = value[:-4] if value.endswith(".mp4") else value
                if ANIMAL_KINGDOM_VIDEO_ID.match(video_id):
                    return video_id

    for label_name in ("spatio_temporal_label", "spatial_label"):
        label = entry.get(label_name)
        source_path = label.get("source_prediction_json_path") if isinstance(label, dict) else None
        if isinstance(source_path, str) and "annotation/animal_kingdom/" in source_path:
            for part in Path(source_path).parts:
                if ANIMAL_KINGDOM_VIDEO_ID.match(part):
                    return part

    return None


def is_animal_kingdom_entry(entry: dict) -> bool:
    if entry.get("source_dataset") == SOURCE_DATASET:
        return True

    for label_name in ("spatio_temporal_label", "spatial_label"):
        label = entry.get(label_name)
        source_path = label.get("source_prediction_json_path") if isinstance(label, dict) else None
        if isinstance(source_path, str) and "annotation/animal_kingdom/" in source_path:
            return True

    return canonical_full_video_name(entry) is not None


def merge_required_video(
    required: dict[tuple[str, str], RequiredVideo],
    item: RequiredVideo,
    metadata_file: Path,
    query_id: str,
    ) -> None:
    key = item.key()
    if key not in required:
        required[key] = item
    else:
        existing = required[key]
        existing.source_candidates.extend(
            candidate for candidate in item.source_candidates if candidate not in existing.source_candidates
        )
    required[key].metadata_files.add(str(metadata_file))
    required[key].query_ids.add(query_id)


def collect_required_videos(
    meta_dir: Path,
    full_video_dir: Path,
    out_root: Path,
    ) -> list[RequiredVideo]:
    required: dict[tuple[str, str], RequiredVideo] = {}
    metadata_paths = sorted(meta_dir.glob("*.json"))

    for metadata_path in metadata_paths:
        with metadata_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            entries = [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
        elif isinstance(payload, list):
            entries = [(str(index), value) for index, value in enumerate(payload) if isinstance(value, dict)]
        else:
            raise TypeError(f"Unsupported metadata container: {type(payload).__name__}")
        is_spatial_file = metadata_path.name.startswith("s_")

        for query_id, entry in entries:
            if not is_animal_kingdom_entry(entry):
                continue

            full_video_id = canonical_full_video_name(entry)
            if full_video_id is not None and not is_spatial_file:
                video_name = full_video_id if full_video_id.endswith(".mp4") else f"{full_video_id}.mp4"
                item = RequiredVideo(
                    kind="full",
                    video_name=full_video_id,
                    target_path=out_root / DOMAIN / "videos" / "clips" / video_name,
                    source_candidates=[full_video_dir / video_name],
                )
                merge_required_video(required, item, metadata_path, query_id)

    return sorted(required.values(), key=lambda item: (item.kind, item.video_name))


def resolve_sources(items: list[RequiredVideo]) -> None:
    for item in items:
        for candidate in item.source_candidates:
            if candidate.exists() and candidate.is_file():
                item.resolved_source_path = candidate
                item.status = "resolved"
                break
        if item.resolved_source_path is None:
            item.status = "missing"


def ffprobe_ok(path: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
    )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def materialize(item: RequiredVideo) -> None:
    if item.resolved_source_path is None:
        return

    target = item.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()

    shutil.copy2(item.resolved_source_path, target)
    item.status = "created"


def validate_videos(items: list[RequiredVideo]) -> None:
    for item in items:
        if not item.target_path.exists():
            continue
        if not ffprobe_ok(item.target_path):
            item.notes.append(f"ffprobe failed: {item.target_path}")
            if item.status not in {"missing"}:
                item.status = "invalid_video"


def summarize(items: list[RequiredVideo]) -> dict:
    by_status: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
        by_kind[item.kind] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "required_videos": len(items),
        "by_kind": dict(sorted(by_kind.items())),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    out_root = OUT_ROOT
    full_video_dir = FULL_VIDEO_DIR
    meta_dir = out_root / DOMAIN / "meta-data"
    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2

    items = collect_required_videos(
        meta_dir=meta_dir,
        full_video_dir=full_video_dir,
        out_root=out_root,
    )
    resolve_sources(items)

    for item in items:
        if item.status == "resolved":
            materialize(item)

    validate_videos(items)
    summary = summarize(items)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing_videos"] or summary["invalid_videos"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
