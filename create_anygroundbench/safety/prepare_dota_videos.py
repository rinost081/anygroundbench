#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SOURCE_DATASET = "DoTA"
DOMAIN = "safety"
RAW_ROOT = Path("data/DoTA")
FRAME_ROOT = Path("frames")
FPS = 10.0
OUT_ROOT = Path("data")

@dataclass
class RequiredVideo:
    video_name: str
    clip_id: str
    frame_dir: Path
    target_path: Path
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


def collect_required_videos(mapping_path: Path, frame_root: Path, out_root: Path) -> list[RequiredVideo]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    required: dict[str, RequiredVideo] = {}
    for row in mapping:
        if row.get("dataset") != SOURCE_DATASET:
            continue
        domain_name = str(row["domain_name"])
        video_name = domain_name if domain_name.endswith(".mp4") else f"{domain_name}.mp4"
        clip_id = video_name.removesuffix(".mp4")
        required[video_name] = RequiredVideo(
            video_name=video_name,
            clip_id=clip_id,
            frame_dir=frame_root / clip_id / "images",
            target_path=out_root / DOMAIN / "videos" / "clips" /  video_name,
        )
    return sorted(required.values(), key=lambda item: item.video_name)


def resolve_frames(items: list[RequiredVideo]) -> None:
    for item in items:
        if not item.frame_dir.exists():
            item.status = "missing_frames"
            item.notes.append(f"frame directory not found: {item.frame_dir}")
            continue
        frames = sorted(item.frame_dir.glob("*.jpg"))
        if not frames:
            item.status = "missing_frames"
            item.notes.append(f"no jpg frames found: {item.frame_dir}")
            continue
        item.status = "resolved"


def ffprobe_ok(path: Path) -> bool:
    try:
        subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def materialize(item: RequiredVideo, fps: float, overwrite: bool) -> None:
    target = item.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            item.status = "exists"
            return
        target.unlink()
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps:g}",
        "-pattern_type", "glob",
        "-i", str(item.frame_dir / "*.jpg"),
        "-an", "-c:v", "libx264", "-profile:v", "baseline", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
        str(target),
    ], check=True)
    item.status = "created"


def validate_videos(items: list[RequiredVideo]) -> None:
    for item in items:
        if item.status not in {"created", "exists"}:
            continue
        if not ffprobe_ok(item.target_path):
            item.status = "invalid_video"
            item.notes.append(f"ffprobe failed: {item.target_path}")


def summarize(items: list[RequiredVideo], frame_root: Path, fps: float) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
    return {
        "domain": DOMAIN,
        "source_dataset": SOURCE_DATASET,
        "frame_root": str(frame_root),
        "fps": fps,
        "required_videos": len(items),
        "by_status": dict(sorted(by_status.items())),
        "missing_frames": by_status.get("missing_frames", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct DoTA videos for AnyGroundBench safety domain metadata.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT, help="Path to the DoTA dataset root. Default: data/DoTA")
    parser.add_argument("--frame-root", type=Path, default=FRAME_ROOT, help="Frame directory. Relative paths are resolved under --raw-root. Default: frames")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT, help="Root containing AnyGroundBench domain directories. Default: data")
    parser.add_argument("--fps", type=float, default=FPS, help="Output FPS. Default: 10")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True, help="Replace existing target files.")
    parser.add_argument("--only-video-id", action="append", default=[], help="Materialize only this video id or filename. Repeatable.")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    frame_root = resolve_path(args.raw_root, args.frame_root)
    mapping_path = out_root / DOMAIN / "filename_mapping.json"
    if not mapping_path.exists():
        print(f"filename mapping not found: {mapping_path}", file=sys.stderr)
        return 2
    if not frame_root.exists():
        print(f"frame root not found: {frame_root}", file=sys.stderr)
        return 2
    items = collect_required_videos(mapping_path, frame_root, out_root)
    if args.only_video_id:
        requested = {video_id if video_id.endswith(".mp4") else f"{video_id}.mp4" for video_id in args.only_video_id}
        items = [item for item in items if item.video_name in requested]
    resolve_frames(items)
    for item in items:
        if item.status == "resolved":
            materialize(item, args.fps, args.overwrite)
    validate_videos(items)
    summary = summarize(items, frame_root=frame_root, fps=args.fps)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["missing_frames"] or summary["invalid_videos"]:
        for item in items:
            if item.status in {"missing_frames", "invalid_video"}:
                print(f"{item.video_name}: {item.status}: {'; '.join(item.notes)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
