"""
Stitch multiple video clips into one output MP4 using ffmpeg (via imageio).

All clips must have the same resolution and fps — guaranteed because every clip
is generated with the same Replicate model settings.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("video_chain_service")


def stitch(clips: list[Path], output_path: Path) -> Path:
    """
    Concatenate clips in order using ffmpeg concat demuxer.
    No re-encoding — copy streams directly (instant, lossless).
    """
    if not clips:
        raise ValueError("No clips to stitch")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(clips) == 1:
        import shutil
        shutil.copy2(clips[0], output_path)
        return output_path


    # Write ffmpeg concat list file
    list_file = Path(tempfile.mktemp(suffix=".txt", dir="/tmp/stylevid2/tmp"))
    list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(list_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]

    log.info(f"Stitching {len(clips)} clips → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        log.error(f"ffmpeg concat failed:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg stitch failed: {result.stderr[-500:]}")

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"Stitched video: {output_path}  ({size_mb:.1f} MB)")
    return output_path
