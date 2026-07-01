#!/usr/bin/env python
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

SOURCE_DATASET = "uca"
DOMAIN = "safety"
RAW_ROOT = Path("data/UCF_Crimes")
SOURCE_VIDEO_DIR = Path(".")
OUT_ROOT = Path("data")

@dataclass
class RequiredVideo:
    video_name: str
    target_path: Path
    source_candidates: list[Path] = field(default_factory=list)
    resolved_source_path: Path | None = None
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


def build_source_index(source_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source_dir.rglob("*.mp4")):
        if path.is_file():
            index[path.name].append(path)
    return index


def collect_required_videos(mapping_path: Path, source_index: dict[str, list[Path]], out_root: Path) -> list[RequiredVideo]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    required: dict[str, RequiredVideo] = {}
    for row in mapping:
        if row.get("dataset") != SOURCE_DATASET:
            continue
        domain_name = str(row["domain_name"])
        video_name = domain_name if domain_name.endswith(".mp4") else f"{domain_name}.mp4"
        required[video_name] = RequiredVideo(
            video_name=video_name,
            target_path=out_root / DOMAIN / "videos" / "clips" / video_name,
            source_candidates=source_index.get(video_name, []),
    )
    return sorted(required.values(), key=lambda item: item.video_name)


def resolve_sources(items: list[RequiredVideo]) -> None:
    for item in items:
        if len(item.source_candidates) == 1:
            item.resolved_source_path = item.source_candidates[0]
            item.status = "resolved"
        elif not item.source_candidates:
            item.status = "missing"
            item.notes.append(f"source video not found: {item.video_name}")
        elif item.video_name == "Normal_Videos_758_x264.mp4":
            preferred = [path for path in item.source_candidates if "Normal_Videos_for_Event_Recognition" in path.parts]
            if len(preferred) == 1:
                item.resolved_source_path = preferred[0]
                item.status = "resolved"
            else:
                item.status = "ambiguous_source"
                item.notes.append("multiple source files: " + ", ".join(str(path) for path in item.source_candidates))
        else:
            item.status = "ambiguous_source"
            item.notes.append("multiple source files: " + ", ".join(str(path) for path in item.source_candidates))


def ffprobe_ok(path: Path) -> bool:
    try:
        subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def summarize(items: list[RequiredVideo], source_dir: Path) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "source_dir": str(source_dir),
        "required_videos": len(items),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "ambiguous_videos": by_status.get("ambiguous_source", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    out_root = OUT_ROOT
    source_dir = RAW_ROOT / SOURCE_VIDEO_DIR
    mapping_path = out_root / DOMAIN / "filename_mapping.json"
    if not mapping_path.exists():
        print(f"filename mapping not found: {mapping_path}", file=sys.stderr)
        return 2
    if not source_dir.exists():
        print(f"source video directory not found: {source_dir}", file=sys.stderr)
        return 2
    source_index = build_source_index(source_dir)
    items = collect_required_videos(mapping_path, source_index, out_root)
    resolve_sources(items)
    for item in items:
            if item.status == "resolved":
                materialize(item)
    validate_videos(items)
    summary = summarize(items, source_dir=source_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["missing_videos"] or summary["ambiguous_videos"] or summary["invalid_videos"]:
        for item in items:
            if item.status in {"missing", "ambiguous_source", "invalid_video"}:
                print(f"{item.video_name}: {item.status}: {'; '.join(item.notes)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
