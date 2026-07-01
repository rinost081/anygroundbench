#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DOMAIN = "industry"
OUT_ROOT = Path("data")
MECCANO_PREFIX = "MECCANO_"
ENIGMA_PREFIX = "ENIGMA_"


@dataclass
class RequiredClip:
    query_id: str
    target_video_name: str
    source_video_name: str
    source_path: Path
    target_path: Path
    start_frame: int
    end_frame: int
    dataset_name: str
    metadata_files: set[str] = field(default_factory=set)
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


def dataset_name(video_name: str) -> str | None:
    if video_name.startswith(MECCANO_PREFIX):
        return "MECCANO"
    if video_name.startswith(ENIGMA_PREFIX):
        return "ENIGMA"
    return None


def spatio_temporal_bbox_window(entry: dict[str, Any], metadata_path: Path, query_id: str) -> tuple[int, int]:
    label = entry.get("spatio_temporal_label")
    if not isinstance(label, dict):
        raise ValueError(f"{metadata_path}:{query_id}: spatio_temporal_label is missing")

    tubes = label.get("tubes")
    if not isinstance(tubes, list) or not tubes:
        raise ValueError(f"{metadata_path}:{query_id}: tubes is missing")

    frame_ids: list[int] = []
    for tube in tubes:
        bbox = tube.get("bbox") if isinstance(tube, dict) else None
        if isinstance(bbox, dict):
            frame_ids.extend(int(float(frame_id)) for frame_id in bbox)
    if not frame_ids:
        raise ValueError(f"{metadata_path}:{query_id}: bbox frame ids are missing")

    return min(frame_ids), max(frame_ids)


def meccano_temporal_range_window(entry: dict[str, Any], metadata_path: Path, query_id: str) -> tuple[int, int]:
    temporal_range = entry.get("temporal_range")
    if not isinstance(temporal_range, str):
        raise ValueError(f"{metadata_path}:{query_id}: temporal_range is missing")
    parts = temporal_range.split()
    if len(parts) != 2:
        raise ValueError(f"{metadata_path}:{query_id}: invalid temporal_range: {temporal_range!r}")

    meta_info = entry.get("meta_info")
    fps = 12.0
    if isinstance(meta_info, dict) and meta_info.get("fps") is not None:
        fps = float(meta_info["fps"])

    start_frame = int(round(float(parts[0]) * fps))
    end_frame = int(round(float(parts[1]) * fps))
    if end_frame < start_frame:
        raise ValueError(f"{metadata_path}:{query_id}: invalid temporal_range frame window")
    return start_frame, end_frame


def frame_window(dataset: str, entry: dict[str, Any], metadata_path: Path, query_id: str) -> tuple[int, int]:
    if dataset == "MECCANO":
        return meccano_temporal_range_window(entry, metadata_path, query_id)
    if dataset == "ENIGMA":
        return spatio_temporal_bbox_window(entry, metadata_path, query_id)
    raise ValueError(f"{metadata_path}:{query_id}: unsupported dataset {dataset!r}")


def collect_required_clips(meta_dir: Path, source_video_dir: Path, target_clip_dir: Path) -> list[RequiredClip]:
    required: list[RequiredClip] = []
    for split in ("train", "test"):
        s_path = meta_dir / f"s_{split}.json"
        temporal_path = meta_dir / f"st_{split}.json"
        if not s_path.exists():
            raise FileNotFoundError(s_path)
        if not temporal_path.exists():
            raise FileNotFoundError(temporal_path)

        s_data_payload = json.loads(s_path.read_text(encoding="utf-8"))
        s_data = {
            str(key): value
            for key, value in (
                s_data_payload.items() if isinstance(s_data_payload, dict) else enumerate(s_data_payload)
            )
            if isinstance(value, dict)
        }
        temporal_data_payload = json.loads(temporal_path.read_text(encoding="utf-8"))
        temporal_data = {
            str(key): value
            for key, value in (
                temporal_data_payload.items() if isinstance(temporal_data_payload, dict) else enumerate(temporal_data_payload)
            )
            if isinstance(value, dict)
        }
        for query_id, s_entry in s_data.items():
            target_name = s_entry.get("video_name")
            if not isinstance(target_name, str):
                raise ValueError(f"{s_path}:{query_id}: video_name is missing")
            dataset = dataset_name(target_name)
            if dataset is None:
                continue
            source_entry = temporal_data.get(query_id)
            if not isinstance(source_entry, dict):
                raise ValueError(f"{temporal_path}:{query_id}: matching st_* entry not found")
            source_name = source_entry.get("video_name")
            if not isinstance(source_name, str):
                raise ValueError(f"{temporal_path}:{query_id}: video_name is missing")

            video_file = (target_name if target_name.endswith(".mp4") else f"{target_name}.mp4")
            source_file = (source_name if source_name.endswith(".mp4") else f"{source_name}.mp4")
            start_frame, end_frame = frame_window(dataset, source_entry, temporal_path, query_id)
            required.append(
                RequiredClip(
                    query_id=query_id,
                    target_video_name=target_name.removesuffix(".mp4"),
                    source_video_name=source_name.removesuffix(".mp4"),
                    source_path=source_video_dir / source_file,
                    target_path=target_clip_dir / video_file,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    dataset_name=dataset,
                    metadata_files={str(s_path), str(temporal_path)},
                )
            )
    return required


def resolve_sources(items: list[RequiredClip]) -> None:
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


def render_clip(item: RequiredClip) -> None:
    target = item.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()

    vf = f"trim=start_frame={item.start_frame}:end_frame={item.end_frame + 1},setpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(item.source_path),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
    ]
    if item.dataset_name == "MECCANO":
        cmd.extend(["-preset", "veryfast", "-crf", "18"])
    cmd.extend(["-pix_fmt", "yuv420p", str(tmp)])
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(target)
        item.status = "created"
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def validate_videos(items: list[RequiredClip]) -> None:
    for item in items:
        if not item.target_path.exists():
            continue
        if not ffprobe_ok(item.target_path):
            item.status = "invalid_video"
            item.notes.append(f"ffprobe failed: {item.target_path}")


def summarize(items: list[RequiredClip], source_video_dir: Path, target_clip_dir: Path) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
        by_dataset[item.dataset_name] += 1
    return {
        "domain": DOMAIN,
        "source_video_dir": str(source_video_dir),
        "target_clip_dir": str(target_clip_dir),
        "required_clips": len(items),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    out_root = OUT_ROOT
    source_video_dir = out_root / DOMAIN / "videos" / "clips"
    target_clip_dir = out_root / DOMAIN / "videos" / "clips4spatial"
    meta_dir = out_root / DOMAIN / "meta-data"
    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2

    items = collect_required_clips(meta_dir, source_video_dir, target_clip_dir)
    resolve_sources(items)
    for item in items:
            if item.status == "resolved":
                try:
                    render_clip(item)
                except subprocess.CalledProcessError as exc:
                    item.status = "render_failed"
                    item.notes.append(str(exc))

    validate_videos(items)
    summary = summarize(items, source_video_dir, target_clip_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing_videos"] or summary["invalid_videos"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
