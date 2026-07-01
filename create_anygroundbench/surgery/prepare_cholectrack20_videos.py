#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SOURCE_DATASET = "CholecTrack20"
DOMAIN = "surgery"
DEFAULT_RAW_ROOT = Path("data/CholecTrack20")
VIDEO_ID = re.compile(r"^VID\d+_\d+$")


@dataclass
class RequiredVideo:
    video_name: str
    source_video_name: str
    target_path: Path
    clip_start_sec: float
    clip_end_sec: float
    source_fps: float
    source_candidates: list[Path] = field(default_factory=list)
    resolved_source_path: Path | None = None
    source_frame_paths: list[Path] = field(default_factory=list)
    render_fps: float | None = None
    metadata_files: set[str] = field(default_factory=set)
    query_ids: set[str] = field(default_factory=set)
    status: str = "pending"
    notes: list[str] = field(default_factory=list)

    def key(self) -> str:
        return self.video_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct CholecTrack20 videos for AnyGroundBench surgery domain metadata."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Path to the CholecTrack20 dataset root. Default: data/CholecTrack20")
    parser.add_argument("--source-video-dir", type=Path, default=Path("."), help="Directory containing extracted CholecTrack20 source videos. Relative paths are resolved under --raw-root. Default: .")
    parser.add_argument("--out-root", type=Path, default=Path("data"), help="Root containing AnyGroundBench domain directories. Default: data")
    parser.add_argument("--source-meta-dir", type=Path, default=Path("data/surgery/meta-data"), help="Metadata directory containing CholecTrack20 meta_info clip timing. Default: data/surgery/meta-data")
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


def source_clip_info(entry: dict[str, Any]) -> tuple[str, float, float, float] | None:
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
    fps = meta_info.get("source_fps", meta_info.get("fps", 25.0))
    if not isinstance(fps, (int, float)) or float(fps) <= 0:
        raise ValueError("meta_info source_fps/fps is missing")
    if float(end) <= float(start):
        raise ValueError(f"invalid clip range for {entry.get('video_name')}: {start} {end}")
    return ensure_mp4(source_video_name), float(start), float(end), float(fps)


def build_source_index(source_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source_dir.rglob("*.mp4")):
        if path.is_file():
            index[path.name].append(path)
    return index




def frame_id(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def frame_dir(raw_root: Path, source_video_name: str) -> Path | None:
    source_id = strip_mp4(source_video_name)
    for split in ("Training", "Validation"):
        path = raw_root / split / source_id / "Frames"
        if path.exists() and any(path.glob("*.png")):
            return path
    return None


def frame_paths_in_range(raw_root: Path, source_video_name: str, start_sec: float, end_sec: float, source_fps: float) -> list[Path]:
    frames = frame_dir(raw_root, source_video_name)
    if frames is None:
        return []
    start_frame = math.floor(start_sec * source_fps)
    end_frame = math.ceil(end_sec * source_fps)
    selected: list[Path] = []
    for path in sorted(frames.glob("*.png")):
        idx = frame_id(path)
        if idx is not None and start_frame <= idx <= end_frame:
            selected.append(path)
    return selected


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


def collect_required_ids(meta_dir: Path) -> set[str]:
    required: set[str] = set()
    for metadata_path in sorted(meta_dir.glob("st_*.json")):
        payload = load_json(metadata_path)
        for _, entry in iter_entries(payload):
            video_id = canonical_video_name(entry)
            if video_id is not None:
                required.add(video_id)
    return required


def collect_source_clip_info(source_meta_dir: Path, required_ids: set[str], source_index: dict[str, list[Path]], out_root: Path) -> list[RequiredVideo]:
    required: dict[str, RequiredVideo] = {}
    for metadata_path in sorted(source_meta_dir.glob("st_*.json")):
        payload = load_json(metadata_path)
        for query_id, entry in iter_entries(payload):
            video_id = canonical_video_name(entry)
            if video_id is None or video_id not in required_ids:
                continue
            clip_info = source_clip_info(entry)
            if clip_info is None:
                continue
            source_video_name, clip_start_sec, clip_end_sec, source_fps = clip_info
            item = RequiredVideo(
                video_name=video_id,
                source_video_name=source_video_name,
                target_path=out_root / DOMAIN / "videos" / "clips" / ensure_mp4(video_id),
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
                source_fps=source_fps,
                source_candidates=source_index.get(source_video_name, []),
            )
            merge_required_video(required, item, metadata_path, query_id)
    missing_info = sorted(required_ids - set(required))
    for video_id in missing_info:
        required[video_id] = RequiredVideo(
            video_name=video_id,
            source_video_name="",
            target_path=out_root / DOMAIN / "videos" / "clips" / ensure_mp4(video_id),
            clip_start_sec=0.0,
            clip_end_sec=0.0,
            source_fps=25.0,
            status="missing_clip_info",
            notes=["clip timing not found in --source-meta-dir"],
        )
    return sorted(required.values(), key=lambda item: item.video_name)


def resolve_sources(items: list[RequiredVideo], raw_root: Path) -> None:
    for item in items:
        if item.status == "conflict":
            continue
        if len(item.source_candidates) == 1:
            item.resolved_source_path = item.source_candidates[0]
            item.status = "resolved"
            continue
        if len(item.source_candidates) > 1:
            item.status = "ambiguous_source"
            item.notes.append("multiple source files: " + ", ".join(str(path) for path in item.source_candidates))
            continue
        frame_paths = frame_paths_in_range(raw_root, item.source_video_name, item.clip_start_sec, item.clip_end_sec, item.source_fps)
        if frame_paths:
            item.source_frame_paths = frame_paths
            item.render_fps = len(frame_paths) / max(item.clip_end_sec - item.clip_start_sec, 1e-6)
            item.status = "resolved"
        else:
            item.status = "missing"
            item.notes.append(f"source video or frames not found: {item.source_video_name}")


def ffprobe_ok(path: Path) -> bool:
    try:
        subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def materialize(item: RequiredVideo, overwrite: bool) -> None:
    target = item.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            item.status = "exists"
            return
        target.unlink()
    duration = item.clip_end_sec - item.clip_start_sec
    if item.resolved_source_path is not None:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{item.clip_start_sec:.6f}",
            "-i", str(item.resolved_source_path),
            "-t", f"{duration:.6f}",
            "-an", "-c:v", "libx264", "-profile:v", "baseline", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
            str(target),
        ], check=True)
    elif item.source_frame_paths:
        fps = item.render_fps or (len(item.source_frame_paths) / max(duration, 1e-6))
        with tempfile.TemporaryDirectory(prefix="cholectrack20_frames_") as tmp_name:
            tmp = Path(tmp_name)
            for index, source in enumerate(item.source_frame_paths):
                os.symlink(source.resolve(), tmp / f"frame_{index:06d}.png")
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", f"{fps:.6f}",
                "-i", str(tmp / "frame_%06d.png"),
                "-an", "-c:v", "libx264", "-profile:v", "baseline", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
                str(target),
            ], check=True)
    else:
        return
    item.status = "created"


def validate_videos(items: list[RequiredVideo], dry_run: bool, skip_ffprobe: bool) -> None:
    if skip_ffprobe:
        return
    for item in items:
        check_path = item.resolved_source_path if dry_run else item.target_path
        if check_path is None or not check_path.exists():
            continue
        if not ffprobe_ok(check_path):
            item.status = "invalid_video"
            item.notes.append(f"ffprobe failed: {check_path}")


def summarize(items: list[RequiredVideo], dry_run: bool, source_dir: Path) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "source_dir": str(source_dir),
        "dry_run": dry_run,
        "required_videos": len(items),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "ambiguous_videos": by_status.get("ambiguous_source", 0),
        "conflict_videos": by_status.get("conflict", 0),
        "missing_clip_info": by_status.get("missing_clip_info", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    args = parse_args()
    source_dir = resolve_source_dir(args.raw_root, args.source_video_dir)
    meta_dir = args.out_root / DOMAIN / "meta-data"
    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2
    if not source_dir.exists():
        print(f"source video directory not found: {source_dir}", file=sys.stderr)
        return 2
    if not args.source_meta_dir.exists():
        print(f"source metadata directory not found: {args.source_meta_dir}", file=sys.stderr)
        return 2

    source_index = build_source_index(source_dir)
    required_ids = collect_required_ids(meta_dir)
    items = collect_source_clip_info(source_meta_dir=args.source_meta_dir, required_ids=required_ids, source_index=source_index, out_root=args.out_root)
    if args.only_video_id:
        requested = {ensure_mp4(video_id) for video_id in args.only_video_id}
        items = [item for item in items if ensure_mp4(item.video_name) in requested or item.target_path.name in requested]
    resolve_sources(items, raw_root=args.raw_root)

    if not args.dry_run:
        for item in items:
            if item.status == "resolved":
                materialize(item, args.overwrite)

    validate_videos(items, dry_run=args.dry_run, skip_ffprobe=args.skip_ffprobe)
    summary = summarize(items, dry_run=args.dry_run, source_dir=source_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["missing_videos"] or summary["ambiguous_videos"] or summary["conflict_videos"] or summary["missing_clip_info"] or summary["invalid_videos"]:
        for item in items:
            if item.status in {"missing", "ambiguous_source", "conflict", "missing_clip_info", "invalid_video"}:
                print(f"{item.video_name}: {item.status}: {'; '.join(item.notes)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
