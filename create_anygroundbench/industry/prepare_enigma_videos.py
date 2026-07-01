#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_DATASET = "enigma"
DOMAIN = "industry"
DEFAULT_RAW_ROOT = Path("data/ENIGMA")
ENIGMA_VIDEO_ID = re.compile(r"^ENIGMA_\d{3}_\d{3}$")


@dataclass
class RequiredVideo:
    video_name: str
    target_path: Path
    source_path: Path
    clip_start_sec: float
    clip_end_sec: float
    metadata_files: set[str] = field(default_factory=set)
    query_ids: set[str] = field(default_factory=set)
    status: str = "pending"
    notes: list[str] = field(default_factory=list)

    def key(self) -> str:
        return self.video_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct ENIGMA videos for AnyGroundBench industry domain metadata."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Path to the ENIGMA dataset root. Default: data/ENIGMA",
    )
    parser.add_argument(
        "--full-video-dir",
        type=Path,
        default=Path("videos/enigma-data/videos"),
        help="Directory containing ENIGMA-51 videos. Relative paths are resolved under --raw-root. Default: videos/enigma-data/videos",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data"),
        help="Root containing AnyGroundBench domain directories. Default: data",
    )
    parser.add_argument(
        "--mode",
        choices=("clip",),
        default="clip",
        help="How to materialize resolved videos. Default: clip",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate source resolution; do not materialize videos.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True, help="Replace an existing target file.")
    parser.add_argument("--only-video-id", action="append", default=[], help="Materialize only this video id or filename. Repeatable.")
    parser.add_argument("--skip-ffprobe", action="store_true", help="Skip ffprobe validation.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        return [(str(key), value) for key, value in payload.items() if isinstance(value, dict)]
    if isinstance(payload, list):
        return [(str(index), value) for index, value in enumerate(payload) if isinstance(value, dict)]
    raise TypeError(f"Unsupported metadata container: {type(payload).__name__}")


def strip_mp4(name: str) -> str:
    return name[:-4] if name.endswith(".mp4") else name


def ensure_mp4(name: str) -> str:
    return name if name.endswith(".mp4") else f"{name}.mp4"


def resolve_source_dir(raw_root: Path, source_dir: Path) -> Path:
    return source_dir if source_dir.is_absolute() else raw_root / source_dir


def canonical_video_name(entry: dict[str, Any]) -> str | None:
    value = entry.get("video_name")
    if isinstance(value, str) and ENIGMA_VIDEO_ID.match(strip_mp4(value)):
        return strip_mp4(value)
    return None


def source_clip_info(entry: dict[str, Any]) -> tuple[str, float, float]:
    meta_info = entry.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("meta_info is missing")
    source_video_name = meta_info.get("source_video_name")
    if not isinstance(source_video_name, str):
        raise ValueError("meta_info.source_video_name is missing")
    start = meta_info.get("clip_start_sec", meta_info.get("clip_start_sec_original"))
    end = meta_info.get("clip_end_sec", meta_info.get("clip_end_sec_original"))
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValueError("meta_info clip start/end seconds are missing")
    if float(end) <= float(start):
        raise ValueError(f"invalid clip range: {start} {end}")
    return source_video_name, float(start), float(end)


def merge_required_video(
    required: dict[str, RequiredVideo],
    item: RequiredVideo,
    metadata_file: Path,
    query_id: str,
) -> None:
    key = item.key()
    if key not in required:
        required[key] = item
    required[key].metadata_files.add(str(metadata_file))
    required[key].query_ids.add(query_id)


def collect_required_videos(meta_dir: Path, source_dir: Path, out_root: Path) -> list[RequiredVideo]:
    required: dict[str, RequiredVideo] = {}
    for metadata_path in sorted(meta_dir.glob("st_*.json")):
        payload = load_json(metadata_path)
        for query_id, entry in iter_entries(payload):
            video_id = canonical_video_name(entry)
            if video_id is None:
                continue
            source_video_name, clip_start_sec, clip_end_sec = source_clip_info(entry)
            item = RequiredVideo(
                video_name=video_id,
                target_path=out_root / DOMAIN / "videos" / "clips" / ensure_mp4(video_id),
                source_path=source_dir / ensure_mp4(source_video_name),
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
            )
            merge_required_video(required, item, metadata_path, query_id)
    return sorted(required.values(), key=lambda item: item.video_name)


def resolve_sources(items: list[RequiredVideo]) -> None:
    for item in items:
        if item.source_path.exists() and item.source_path.is_file():
            item.status = "resolved"
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


def materialize(item: RequiredVideo, mode: str, overwrite: bool) -> None:
    target = item.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            item.status = "exists"
            return
        target.unlink()

    if mode != "clip":
        raise ValueError(f"Unsupported mode: {mode}")
    duration = item.clip_end_sec - item.clip_start_sec
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{item.clip_start_sec:.6f}",
            "-i",
            str(item.source_path),
            "-t",
            f"{duration:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            str(target),
        ],
        check=True,
    )
    item.status = "created"


def validate_videos(items: list[RequiredVideo], dry_run: bool, skip_ffprobe: bool) -> None:
    if skip_ffprobe:
        return
    for item in items:
        check_path = item.source_path if dry_run else item.target_path
        if not check_path.exists():
            continue
        if not ffprobe_ok(check_path):
            item.status = "invalid_video"
            item.notes.append(f"ffprobe failed: {check_path}")


def summarize(items: list[RequiredVideo], mode: str, dry_run: bool, source_dir: Path) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "source_dir": str(source_dir),
        "mode": mode,
        "dry_run": dry_run,
        "required_videos": len(items),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    args = parse_args()
    source_dir = resolve_source_dir(args.raw_root, args.full_video_dir)
    meta_dir = args.out_root / DOMAIN / "meta-data"
    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2

    items = collect_required_videos(meta_dir=meta_dir, source_dir=source_dir, out_root=args.out_root)
    if args.only_video_id:
        requested = {ensure_mp4(video_id) for video_id in args.only_video_id}
        items = [item for item in items if ensure_mp4(item.video_name) in requested or item.target_path.name in requested]
    resolve_sources(items)
    if not args.dry_run:
        for item in items:
            if item.status == "resolved":
                try:
                    materialize(item, args.mode, args.overwrite)
                except subprocess.CalledProcessError as exc:
                    item.status = "materialize_failed"
                    item.notes.append(str(exc))

    validate_videos(items, dry_run=args.dry_run, skip_ffprobe=args.skip_ffprobe)
    summary = summarize(items, mode=args.mode, dry_run=args.dry_run, source_dir=source_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing_videos"] or summary["invalid_videos"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
