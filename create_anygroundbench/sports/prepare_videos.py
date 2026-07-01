#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_DATASET = "multisports"
DOMAIN = "sports"
RAW_ROOT = Path("data/MultiSports")
RAW_DATA_DIR = Path("data")
OUT_ROOT = Path("data")
SPORTS = ("aerobic_gymnastics", "basketball", "football", "volleyball")
SPLITS = ("trainval", "test")


@dataclass
class RequiredVideo:
    video_name: str
    target_path: Path
    source_candidates: list[Path]
    metadata_files: set[str] = field(default_factory=set)
    query_ids: set[str] = field(default_factory=set)
    resolved_source_path: Path | None = None
    status: str = "pending"
    notes: list[str] = field(default_factory=list)

    def key(self) -> str:
        return self.video_name


def is_multisports_video_name(value: Any) -> bool:
    return isinstance(value, str) and value.removesuffix(".mp4").startswith("v_")


def canonical_multisports_video_name(entry: dict[str, Any]) -> str | None:
    video_name = entry.get("video_name")
    if is_multisports_video_name(video_name):
        return video_name.removesuffix(".mp4")

    meta_info = entry.get("meta_info")
    if isinstance(meta_info, dict):
        for key in ("source_video_name", "video_name"):
            value = meta_info.get(key)
            if is_multisports_video_name(value):
                return value.removesuffix(".mp4")

    return None


def build_source_index(raw_data_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for split in SPLITS:
        for sport in SPORTS:
            video_dir = raw_data_dir / split / sport
            if not video_dir.exists():
                continue
            for path in sorted(video_dir.glob("*.mp4")):
                index[path.name].append(path)
    return dict(index)


def source_candidates(source_index: dict[str, list[Path]], video_name: str) -> list[Path]:
    video_file = video_name if video_name.endswith(".mp4") else f"{video_name}.mp4"
    return source_index.get(video_file, [])


def merge_required_video(
    required: dict[str, RequiredVideo],
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
    source_index: dict[str, list[Path]],
    out_root: Path,
    ) -> list[RequiredVideo]:
    required: dict[str, RequiredVideo] = {}
    metadata_paths = sorted(meta_dir.glob("*.json"))

    for metadata_path in metadata_paths:
        if metadata_path.name.startswith("s_"):
            continue
        entries_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = [
            (str(key), value)
            for key, value in (
                entries_payload.items() if isinstance(entries_payload, dict) else enumerate(entries_payload)
            )
            if isinstance(value, dict)
        ]

        for query_id, entry in entries:
            video_id = canonical_multisports_video_name(entry)
            if video_id is None:
                continue

            item = RequiredVideo(
                video_name=video_id,
                target_path=out_root / DOMAIN / "videos" / "clips" / (video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"),
                source_candidates=source_candidates(source_index, video_id),
    )
            merge_required_video(required, item, metadata_path, query_id)

    return sorted(required.values(), key=lambda item: item.video_name)


def resolve_sources(items: list[RequiredVideo]) -> None:
    for item in items:
        existing_candidates = [
            candidate for candidate in item.source_candidates if candidate.exists() and candidate.is_file()
        ]
        if len(existing_candidates) == 1:
            item.resolved_source_path = existing_candidates[0]
            item.status = "resolved"
        elif len(existing_candidates) > 1:
            item.status = "ambiguous_source"
            item.notes.append(
                "multiple source files: " + ", ".join(str(candidate) for candidate in existing_candidates)
            )
        else:
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
            if item.status not in {"missing", "ambiguous_source"}:
                item.status = "invalid_video"


def summarize(items: list[RequiredVideo], raw_data_dir: Path) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "raw_data_dir": str(raw_data_dir),
        "required_videos": len(items),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "ambiguous_videos": by_status.get("ambiguous_source", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    raw_data_dir = RAW_ROOT / RAW_DATA_DIR
    out_root = OUT_ROOT
    meta_dir = out_root / DOMAIN / "meta-data"
    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2

    source_index = build_source_index(raw_data_dir)
    items = collect_required_videos(
        meta_dir=meta_dir,
        source_index=source_index,
        out_root=out_root,
    )
    resolve_sources(items)

    for item in items:
            if item.status == "resolved":
                try:
                    materialize(item)
                except subprocess.CalledProcessError as exc:
                    item.status = "materialize_failed"
                    item.notes.append(str(exc))

    validate_videos(items)
    summary = summarize(items, raw_data_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing_videos"] or summary["ambiguous_videos"] or summary["invalid_videos"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
