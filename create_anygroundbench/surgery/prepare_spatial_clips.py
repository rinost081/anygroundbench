#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DOMAIN = "surgery"
EGOSURGERY_PREFIX = "egosurgery_"
CHOLECTRACK20_PREFIX = "CholecTrack20_"
DEFAULT_OUT_ROOT = Path("data")
DEFAULT_CHOLECTRACK20_RAW_ROOT = Path("data/CholecTrack20")


@dataclass
class RequiredClip:
    query_id: str
    dataset_name: str
    target_video_name: str
    target_path: Path
    source_video_name: str
    source_path: Path | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    source_video_id: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    source_fps: float | None = None
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct surgery clips4spatial from prepared EgoSurgery clips and raw CholecTrack20 source."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="AnyGroundBench data root. Default: data")
    parser.add_argument(
        "--source-video-dir",
        type=Path,
        default=None,
        help="Directory containing prepared surgery clips for EgoSurgery. Default: <out-root>/surgery/videos/clips",
    )
    parser.add_argument(
        "--target-clip-dir",
        type=Path,
        default=None,
        help="Directory for reconstructed clips4spatial. Default: <out-root>/surgery/videos/clips4spatial",
    )
    parser.add_argument(
        "--cholectrack20-raw-root",
        type=Path,
        default=DEFAULT_CHOLECTRACK20_RAW_ROOT,
        help="Path to the extracted CholecTrack20 official dataset root. Default: data/CholecTrack20",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs but do not write clips.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True, help="Replace existing target clips.")
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


def dataset_name(video_name: str) -> str | None:
    if video_name.startswith(EGOSURGERY_PREFIX):
        return "egosurgery"
    if video_name.startswith(CHOLECTRACK20_PREFIX):
        return "CholecTrack20"
    return None


def bbox_frame_window(entry: dict[str, Any], metadata_path: Path, query_id: str) -> tuple[int, int]:
    label = entry.get("spatio_temporal_label")
    if not isinstance(label, dict):
        raise ValueError(f"{metadata_path}:{query_id}: spatio_temporal_label is missing")

    start = label.get("st_frame")
    end = label.get("ed_frame")
    if isinstance(start, int) and isinstance(end, int):
        if end < start:
            raise ValueError(f"{metadata_path}:{query_id}: invalid st_frame/ed_frame")
        return start, end

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


def collect_required_clips(meta_dir: Path, source_video_dir: Path, target_clip_dir: Path) -> list[RequiredClip]:
    required: list[RequiredClip] = []
    for split in ("train", "test"):
        s_path = meta_dir / f"s_{split}.json"
        st_path = meta_dir / f"st_{split}.json"
        if not s_path.exists():
            raise FileNotFoundError(s_path)
        if not st_path.exists():
            raise FileNotFoundError(st_path)

        s_data = dict(iter_entries(load_json(s_path)))
        st_data = dict(iter_entries(load_json(st_path)))
        missing_st = sorted(set(s_data) - set(st_data))
        if missing_st:
            raise ValueError(f"{split}: s_* entries missing in st_*: {missing_st[:5]}")

        for query_id, s_entry in sorted(s_data.items()):
            target_name = s_entry.get("video_name")
            if not isinstance(target_name, str):
                raise ValueError(f"{s_path}:{query_id}: video_name is missing")
            dataset = dataset_name(target_name)
            if dataset is None:
                continue

            st_entry = st_data.get(query_id)
            if not isinstance(st_entry, dict):
                raise ValueError(f"{st_path}:{query_id}: matching st_* entry not found")
            source_name = st_entry.get("video_name")
            if not isinstance(source_name, str):
                raise ValueError(f"{st_path}:{query_id}: video_name is missing")

            if dataset == "egosurgery":
                start_frame, end_frame = bbox_frame_window(st_entry, st_path, query_id)
                required.append(
                    RequiredClip(
                        query_id=query_id,
                        dataset_name=dataset,
                        target_video_name=strip_mp4(target_name),
                        target_path=target_clip_dir / ensure_mp4(target_name),
                        source_video_name=strip_mp4(source_name),
                        source_path=source_video_dir / ensure_mp4(source_name),
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                )
                continue

            meta = st_entry.get("meta_info")
            if not isinstance(meta, dict):
                raise ValueError(f"{st_path}:{query_id}: CholecTrack20 meta_info is missing")
            source_video_id = meta.get("source_video")
            start_sec = meta.get("gt_start_sec_original")
            end_sec = meta.get("gt_end_sec_original")
            source_fps = meta.get("source_fps", 25.0)
            if not isinstance(source_video_id, str):
                raise ValueError(f"{st_path}:{query_id}: source_video is missing")
            if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)):
                raise ValueError(f"{st_path}:{query_id}: gt_start/end seconds are missing")
            if float(end_sec) <= float(start_sec):
                raise ValueError(f"{st_path}:{query_id}: invalid gt_start/end seconds")
            if not isinstance(source_fps, (int, float)) or float(source_fps) <= 0:
                raise ValueError(f"{st_path}:{query_id}: invalid source_fps")

            required.append(
                RequiredClip(
                    query_id=query_id,
                    dataset_name=dataset,
                    target_video_name=strip_mp4(target_name),
                    target_path=target_clip_dir / ensure_mp4(target_name),
                    source_video_name=source_video_id,
                    source_video_id=source_video_id,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    source_fps=float(source_fps),
                )
            )
    return required


def chole_frame_dir(raw_root: Path, source_video_id: str) -> Path | None:
    for split in ("Training", "Validation"):
        frames = raw_root / split / source_video_id / "Frames"
        if frames.exists() and any(frames.glob("*.png")):
            return frames
    return None


def chole_source_video_path(raw_root: Path, source_video_id: str) -> Path | None:
    for split in ("Testing", "Training", "Validation"):
        path = raw_root / split / source_video_id / f"{source_video_id}.mp4"
        if path.exists():
            return path
    return None


def chole_frame_map(raw_root: Path, source_video_id: str) -> dict[int, Path]:
    frames = chole_frame_dir(raw_root, source_video_id)
    if frames is None:
        return {}
    out: dict[int, Path] = {}
    for path in frames.glob("*.png"):
        try:
            out[int(path.stem)] = path
        except ValueError:
            pass
    return out


def resolve_sources(items: list[RequiredClip], chole_raw_root: Path, overwrite: bool) -> None:
    for item in items:
        if item.target_path.exists() and not overwrite:
            item.status = "exists"
            continue
        if item.dataset_name == "egosurgery":
            if item.source_path and item.source_path.exists() and item.source_path.is_file():
                item.status = "resolved"
            else:
                item.status = "missing"
                item.notes.append(f"source video not found: {item.source_path}")
            continue

        assert item.source_video_id is not None
        if chole_frame_map(chole_raw_root, item.source_video_id):
            item.status = "resolved"
            continue
        source_path = chole_source_video_path(chole_raw_root, item.source_video_id)
        if source_path is None:
            item.status = "missing"
            item.notes.append(f"source media not found for CholecTrack20 video: {item.source_video_id}")
        else:
            item.source_path = source_path
            item.status = "resolved"


def ffprobe_ok(path: Path) -> bool:
    try:
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def render_egosurgery(item: RequiredClip, tmp: Path) -> None:
    assert item.source_path is not None
    assert item.start_frame is not None and item.end_frame is not None
    vf = f"trim=start_frame={item.start_frame}:end_frame={item.end_frame + 1},setpts=PTS-STARTPTS"
    subprocess.run(
        [
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
            "-pix_fmt",
            "yuv420p",
            str(tmp),
        ],
        check=True,
    )


def render_chole_from_frames(frame_paths: list[Path], tmp: Path, fps: float) -> None:
    with tempfile.TemporaryDirectory(prefix="cholectrack20_spatial_", dir="/tmp") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for index, path in enumerate(frame_paths):
            os.symlink(path.resolve(), tmp_dir / f"frame_{index:06d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-framerate",
                f"{fps:.6f}",
                "-i",
                str(tmp_dir / "frame_%06d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                str(tmp),
            ],
            check=True,
        )


def render_chole(item: RequiredClip, raw_root: Path, tmp: Path) -> None:
    assert item.source_video_id is not None
    assert item.start_sec is not None and item.end_sec is not None
    source_fps = item.source_fps if item.source_fps is not None else 25.0
    frames = chole_frame_map(raw_root, item.source_video_id)
    if frames:
        start_frame = math.floor(item.start_sec * source_fps)
        end_frame = math.ceil(item.end_sec * source_fps)
        frame_ids = [idx for idx in sorted(frames) if start_frame <= idx <= end_frame]
        if not frame_ids:
            raise FileNotFoundError(f"no CholecTrack20 frames in range for {item.source_video_id}")
        fps = len(frame_ids) / max(item.end_sec - item.start_sec, 1e-6)
        render_chole_from_frames([frames[idx] for idx in frame_ids], tmp, fps)
        return

    source_path = chole_source_video_path(raw_root, item.source_video_id)
    if source_path is None:
        raise FileNotFoundError(item.source_video_id)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ss",
            f"{item.start_sec:.6f}",
            "-t",
            f"{item.end_sec - item.start_sec:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            str(tmp),
        ],
        check=True,
    )


def render_clip(item: RequiredClip, chole_raw_root: Path, overwrite: bool) -> None:
    target = item.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            item.status = "exists"
            return
        target.unlink()

    tmp = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    try:
        if item.dataset_name == "egosurgery":
            render_egosurgery(item, tmp)
        elif item.dataset_name == "CholecTrack20":
            render_chole(item, chole_raw_root, tmp)
        else:
            raise ValueError(item.dataset_name)
        tmp.replace(target)
        item.status = "created"
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def validate_videos(items: list[RequiredClip], dry_run: bool, skip_ffprobe: bool) -> None:
    if skip_ffprobe:
        return
    for item in items:
        check_path = item.target_path
        if dry_run and item.status == "resolved" and item.dataset_name == "egosurgery":
            check_path = item.source_path or item.target_path
        if not check_path.exists():
            continue
        if not ffprobe_ok(check_path):
            item.status = "invalid_video"
            item.notes.append(f"ffprobe failed: {check_path}")


def summarize(items: list[RequiredClip], dry_run: bool, source_video_dir: Path, target_clip_dir: Path, chole_raw_root: Path) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    by_dataset: dict[str, int] = defaultdict(int)
    for item in items:
        by_status[item.status] += 1
        by_dataset[item.dataset_name] += 1
    return {
        "domain": DOMAIN,
        "source_video_dir": str(source_video_dir),
        "target_clip_dir": str(target_clip_dir),
        "cholectrack20_raw_root": str(chole_raw_root),
        "dry_run": dry_run,
        "required_clips": len(items),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_status": dict(sorted(by_status.items())),
        "missing_videos": by_status.get("missing", 0),
        "invalid_videos": by_status.get("invalid_video", 0),
    }


def main() -> int:
    args = parse_args()
    meta_dir = args.out_root / DOMAIN / "meta-data"
    source_video_dir = args.source_video_dir or args.out_root / DOMAIN / "videos" / "clips"
    target_clip_dir = args.target_clip_dir or args.out_root / DOMAIN / "videos" / "clips4spatial"
    if not meta_dir.exists():
        print(f"metadata directory not found: {meta_dir}", file=sys.stderr)
        return 2

    items = collect_required_clips(meta_dir, source_video_dir, target_clip_dir)
    resolve_sources(items, args.cholectrack20_raw_root, args.overwrite)

    if not args.dry_run:
        for item in items:
            if item.status == "resolved":
                try:
                    render_clip(item, args.cholectrack20_raw_root, args.overwrite)
                except subprocess.CalledProcessError as exc:
                    item.status = "render_failed"
                    item.notes.append(str(exc))
                except FileNotFoundError as exc:
                    item.status = "missing"
                    item.notes.append(str(exc))

    validate_videos(items, dry_run=args.dry_run, skip_ffprobe=args.skip_ffprobe)
    summary = summarize(items, args.dry_run, source_video_dir, target_clip_dir, args.cholectrack20_raw_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing_videos"] or summary["invalid_videos"] or summary["by_status"].get("render_failed", 0):
        for item in items:
            if item.status in {"missing", "invalid_video", "render_failed"}:
                print(f"{item.query_id} {item.target_video_name}: {item.status}: {'; '.join(item.notes)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
