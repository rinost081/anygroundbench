#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_DATASET = "animal_kingdom"
DOMAIN = "animal"
ANIMAL_KINGDOM_VIDEO_ID = re.compile(r"^[A-Z]{8}$")
OUT_ROOT = Path("data")


@dataclass
class RequiredClip:
    query_id: str
    video_name: str
    source_video_id: str
    start_frame: int
    end_frame: int
    source_path: Path
    target_path: Path
    metadata_file: str
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


def is_animal_kingdom_domain_clip(entry: dict[str, Any]) -> bool:
    video_name = entry.get("video_name")
    return isinstance(video_name, str) and video_name.startswith(f"{SOURCE_DATASET}_")


def frame_window(entry: dict[str, Any], metadata_path: Path, query_id: str) -> tuple[int, int]:
    temporal_range = entry.get("temporal_range")
    if not isinstance(temporal_range, str):
        raise ValueError(f"{metadata_path}:{query_id}: temporal_range is missing")
    parts = temporal_range.split()
    if len(parts) != 2:
        raise ValueError(f"{metadata_path}:{query_id}: invalid temporal_range: {temporal_range!r}")

    meta_info = entry.get("meta_info")
    if not isinstance(meta_info, dict) or meta_info.get("fps") is None:
        raise ValueError(f"{metadata_path}:{query_id}: meta_info.fps is missing")
    fps = float(meta_info["fps"])

    start_frame = int(float(parts[0]) * fps)
    end_frame = int(round(float(parts[1]) * fps))
    if end_frame < start_frame:
        raise ValueError(f"{metadata_path}:{query_id}: invalid temporal_range frame window")
    return start_frame, end_frame


def collect_required_clips(
    meta_dir: Path,
    source_video_dir: Path,
    target_clip_dir: Path,
    ) -> list[RequiredClip]:
    clips: list[RequiredClip] = []

    for split in ("train", "test"):
        metadata_path = meta_dir / f"s_{split}.json"
        temporal_path = meta_dir / f"st_{split}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        if not temporal_path.exists():
            raise FileNotFoundError(temporal_path)

        entries_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = {
            str(key): value
            for key, value in (
                entries_payload.items() if isinstance(entries_payload, dict) else enumerate(entries_payload)
            )
            if isinstance(value, dict)
        }
        temporal_entries_payload = json.loads(temporal_path.read_text(encoding="utf-8"))
        temporal_entries = {
            str(key): value
            for key, value in (
                temporal_entries_payload.items() if isinstance(temporal_entries_payload, dict) else enumerate(temporal_entries_payload)
            )
            if isinstance(value, dict)
        }
        for query_id, entry in entries.items():
            if not is_animal_kingdom_domain_clip(entry):
                continue
            source_entry = temporal_entries.get(query_id)
            if not isinstance(source_entry, dict):
                raise ValueError(f"{temporal_path}:{query_id}: matching st_* entry not found")

            video_name = str(entry["video_name"])
            source_video_name = source_entry.get("video_name")
            if not isinstance(source_video_name, str):
                raise ValueError(f"{temporal_path}:{query_id}: video_name is missing")
            video_file = (video_name if video_name.endswith(".mp4") else f"{video_name}.mp4")
            source_file = (source_video_name if source_video_name.endswith(".mp4") else f"{source_video_name}.mp4")
            start_frame, end_frame = frame_window(source_entry, temporal_path, query_id)
            clips.append(
                RequiredClip(
                    query_id=query_id,
                    video_name=video_name,
                    source_video_id=source_video_name.removesuffix(".mp4"),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    source_path=source_video_dir / source_file,
                    target_path=target_clip_dir / video_file,
                    metadata_file=str(metadata_path),
                )
            )

    return sorted(clips, key=lambda item: item.video_name)


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


def resolve_sources_and_targets(clips: list[RequiredClip]) -> None:
    for clip in clips:
        if not clip.source_path.exists():
            clip.status = "missing_source"
            continue
        if not clip.source_path.is_file():
            clip.status = "invalid_source"
            clip.notes.append(f"source is not a file: {clip.source_path}")
            continue
        if not ffprobe_ok(clip.source_path):
            clip.status = "invalid_source"
            clip.notes.append(f"ffprobe failed: {clip.source_path}")
            continue
        clip.status = "resolved"


def create_clip(clip: RequiredClip) -> None:
    if clip.status != "resolved":
        return

    clip.target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = clip.target_path.with_name(f".{clip.target_path.stem}.tmp.{os.getpid()}{clip.target_path.suffix}")
    if tmp_path.exists():
        raise FileExistsError(tmp_path)

    vf = f"trim=start_frame={clip.start_frame}:end_frame={clip.end_frame + 1},setpts=PTS-STARTPTS"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(clip.source_path),
                "-an",
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                str(tmp_path),
            ],
            check=True,
    )
        tmp_path.replace(clip.target_path)
        clip.status = "created"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def validate_targets(clips: list[RequiredClip]) -> None:
    for clip in clips:
        if clip.status not in {"created", "exists"}:
            continue
        if not ffprobe_ok(clip.target_path):
            clip.status = "invalid_target"
            clip.notes.append(f"ffprobe failed: {clip.target_path}")


def summarize(clips: list[RequiredClip]) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    by_metadata_file: dict[str, int] = defaultdict(int)
    for clip in clips:
        by_status[clip.status] += 1
        by_metadata_file[clip.metadata_file] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "required_clips": len(clips),
        "by_metadata_file": dict(sorted(by_metadata_file.items())),
        "by_status": dict(sorted(by_status.items())),
        "missing_sources": by_status.get("missing_source", 0),
        "invalid_sources": by_status.get("invalid_source", 0),
        "invalid_targets": by_status.get("invalid_target", 0),
        "failed": by_status.get("failed", 0),
    }


def main() -> int:
    out_root = OUT_ROOT
    source_video_dir = out_root / DOMAIN / "videos" / "clips"
    target_clip_dir = out_root / DOMAIN / "videos" / "clips4spatial"
    meta_dir = out_root / DOMAIN / "meta-data"

    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2
    if not source_video_dir.exists():
        print(f"source video directory not found: {source_video_dir}", file=sys.stderr)
        return 2

    clips = collect_required_clips(meta_dir, source_video_dir, target_clip_dir)
    resolve_sources_and_targets(clips)

    for index, clip in enumerate(clips, start=1):
            if clip.status == "resolved":
                try:
                    create_clip(clip)
                except (FileExistsError, subprocess.CalledProcessError) as exc:
                    clip.status = "failed"
                    clip.notes.append(str(exc))
            if index % 50 == 0:
                print(f"processed {index}/{len(clips)} clips", file=sys.stderr)

    validate_targets(clips)
    summary = summarize(clips)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if (
        summary["missing_sources"]
        or summary["invalid_sources"]
        or summary["invalid_targets"]
        or summary["failed"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
