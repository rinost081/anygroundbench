#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_DATASET = "egosurgery"
DOMAIN = "surgery"
DEFAULT_RAW_ROOT = Path("data/egosurgery")
DEFAULT_IMAGE_DIR = Path("images")
SOURCE_FPS = 0.5
VIDEO_ID = re.compile(r"^\d{2}_\d_\d+$")


@dataclass
class RequiredVideo:
    video_name: str
    source_video_name: str
    target_path: Path
    clip_start_sec: float
    clip_end_sec: float
    source_video_path: Path | None = None
    metadata_files: set[str] = field(default_factory=set)
    query_ids: set[str] = field(default_factory=set)
    status: str = "pending"
    notes: list[str] = field(default_factory=list)

    def key(self) -> str:
        return self.video_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct EgoSurgery videos for AnyGroundBench surgery domain metadata."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Path to the EgoSurgery dataset root. Default: data/egosurgery")
    parser.add_argument(
        "--source-image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing EgoSurgery extracted frames. Relative paths are resolved under --raw-root. Default: images",
    )
    parser.add_argument("--out-root", type=Path, default=Path("data"), help="Root containing AnyGroundBench domain directories. Default: data")
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
    if isinstance(value, str) and VIDEO_ID.match(strip_mp4(value)):
        return strip_mp4(value)
    return None


def source_clip_info(entry: dict[str, Any]) -> tuple[str, float, float] | None:
    meta_info = entry.get("meta_info")
    if not isinstance(meta_info, dict):
        return None
    source_video_name = meta_info.get("source_video_name")
    if not isinstance(source_video_name, str):
        return None
    start = meta_info.get("clip_start_sec", meta_info.get("clip_start_sec_original"))
    end = meta_info.get("clip_end_sec", meta_info.get("clip_end_sec_original"))
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if float(end) <= float(start):
        raise ValueError(f"invalid clip range for {entry.get('video_name')}: {start} {end}")
    return ensure_mp4(source_video_name), float(start), float(end)


def source_stem(source_video_name: str) -> str:
    return strip_mp4(source_video_name)


def source_case_dir(image_dir: Path, source_video_name: str) -> Path:
    stem = source_stem(source_video_name)
    return image_dir / stem.split("_", 1)[0]


def image_sequence_pattern(image_dir: Path, source_video_name: str) -> tuple[Path, Path]:
    stem = source_stem(source_video_name)
    case_dir = source_case_dir(image_dir, source_video_name)
    return case_dir / f"{stem}_%04d.jpg", case_dir / f"{stem}_0001.jpg"


def merge_required_video(required: dict[str, RequiredVideo], item: RequiredVideo, metadata_file: Path, query_id: str) -> None:
    key = item.key()
    if key not in required:
        required[key] = item
    else:
        existing = required[key]
        if (
            existing.source_video_name != item.source_video_name
            or abs(existing.clip_start_sec - item.clip_start_sec) > 1e-6
            or abs(existing.clip_end_sec - item.clip_end_sec) > 1e-6
        ):
            existing.status = "conflict"
            existing.notes.append(f"conflicting source clip info in {metadata_file}:{query_id}")
    required[key].metadata_files.add(str(metadata_file))
    required[key].query_ids.add(query_id)


def collect_required_videos(meta_dir: Path, out_root: Path) -> list[RequiredVideo]:
    required: dict[str, RequiredVideo] = {}
    for metadata_path in sorted(meta_dir.glob("st_*.json")):
        payload = load_json(metadata_path)
        for query_id, entry in iter_entries(payload):
            video_id = canonical_video_name(entry)
            if video_id is None:
                continue
            clip_info = source_clip_info(entry)
            if clip_info is None:
                continue
            source_video_name, clip_start_sec, clip_end_sec = clip_info
            item = RequiredVideo(
                video_name=video_id,
                source_video_name=source_video_name,
                target_path=out_root / DOMAIN / "videos" / "clips" / ensure_mp4(video_id),
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
            )
            merge_required_video(required, item, metadata_path, query_id)
    return sorted(required.values(), key=lambda item: item.video_name)


def resolve_sources(items: list[RequiredVideo], image_dir: Path) -> None:
    for item in items:
        if item.status == "conflict":
            continue
        _, first_frame = image_sequence_pattern(image_dir, item.source_video_name)
        if first_frame.exists():
            item.status = "resolved"
        else:
            item.status = "missing"
            item.notes.append(f"source image sequence not found: {first_frame}")


def ffprobe_ok(path: Path) -> bool:
    try:
        subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def build_long_source(source_video_name: str, image_dir: Path, tmp_dir: Path) -> Path:
    output = tmp_dir / ensure_mp4(source_stem(source_video_name))
    if output.exists():
        return output
    pattern, _ = image_sequence_pattern(image_dir, source_video_name)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(SOURCE_FPS),
        "-start_number", "1",
        "-i", str(pattern),
        "-an", "-c:v", "libx264", "-profile:v", "baseline", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
        str(output),
    ], check=True)
    return output


def materialize(item: RequiredVideo, source_video_path: Path, overwrite: bool) -> None:
    target = item.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            item.status = "exists"
            return
        target.unlink()
    duration = item.clip_end_sec - item.clip_start_sec
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{item.clip_start_sec:.6f}",
        "-i", str(source_video_path),
        "-t", f"{duration:.6f}",
        "-an", "-c:v", "libx264", "-profile:v", "baseline", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
        str(target),
    ], check=True)
    item.status = "created"


def validate_videos(items: list[RequiredVideo], skip_ffprobe: bool) -> None:
    if skip_ffprobe:
        return
    for item in items:
        if item.target_path.exists() and not ffprobe_ok(item.target_path):
            item.status = "invalid_video"
            item.notes.append(f"ffprobe failed: {item.target_path}")


def summarize(items: list[RequiredVideo], image_dir: Path) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "source_image_dir": str(image_dir),
        "required_videos": len(items),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "conflict_videos": by_status.get("conflict", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    args = parse_args()
    image_dir = resolve_source_dir(args.raw_root, args.source_image_dir)
    meta_dir = args.out_root / DOMAIN / "meta-data"
    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2
    if not image_dir.exists():
        print(f"source image directory not found: {image_dir}", file=sys.stderr)
        return 2

    items = collect_required_videos(meta_dir=meta_dir, out_root=args.out_root)
    if args.only_video_id:
        requested = {ensure_mp4(video_id) for video_id in args.only_video_id}
        items = [item for item in items if ensure_mp4(item.video_name) in requested or item.target_path.name in requested]
    resolve_sources(items, image_dir=image_dir)

    if not args.dry_run:
        with tempfile.TemporaryDirectory(prefix="egosurgery_long_sources_") as tmp_name:
            tmp_dir = Path(tmp_name)
            source_cache: dict[str, Path] = {}
            for item in items:
                if item.status != "resolved":
                    continue
                source_path = source_cache.get(item.source_video_name)
                if source_path is None:
                    source_path = build_long_source(item.source_video_name, image_dir=image_dir, tmp_dir=tmp_dir)
                    source_cache[item.source_video_name] = source_path
                materialize(item, source_video_path=source_path, overwrite=args.overwrite)

    validate_videos(items, skip_ffprobe=args.skip_ffprobe)
    summary = summarize(items, image_dir=image_dir)
    summary["dry_run"] = args.dry_run
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["missing_videos"] or summary["conflict_videos"] or summary["invalid_videos"]:
        for item in items:
            if item.status in {"missing", "conflict", "invalid_video"}:
                print(f"{item.video_name}: {item.status}: {'; '.join(item.notes)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
