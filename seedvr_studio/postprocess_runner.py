"""Run tools/postprocess_tensor_video.py once for many decoded batches (models load a single time)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable

from .media import MediaError
from .paths import ROOT, VENV_PYTHON

TRT_POSTPROCESS = ROOT / "tools" / "postprocess_tensor_video.py"
ProgressCallback = Callable[[float, str], None]


def _safe_print(value: str) -> None:
    try:
        print(value, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(value.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def run_postprocess_jobs(
    jobs: list[dict[str, Any]], common_args: list[str], manifest_path: Path, child_env: dict[str, str],
    progress_callback: ProgressCallback | None = None, *, progress_base: float = 0.0, progress_span: float = 0.0, label: str = "Post-processing",
) -> None:
    """`jobs` = [{"input", "output", "frame_start"}]; `common_args` = effect flags shared by all batches."""
    if not jobs:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    command = [str(VENV_PYTHON), "-u", str(TRT_POSTPROCESS), "--jobs", str(manifest_path), *common_args]
    process = subprocess.Popen(
        command, cwd=ROOT, env=child_env, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    tail: deque[str] = deque(maxlen=80)
    assert process.stdout is not None
    if progress_callback:
        progress_callback(progress_base, f"{label}: loading models for {len(jobs)} batches")
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        tail.append(line)
        _safe_print(line)
        match = re.search(r"POSTPROCESS_PROGRESS (\d+)/(\d+)", line)
        if match and progress_callback:
            current, total = int(match.group(1)), max(1, int(match.group(2)))
            progress_callback(progress_base + progress_span * current / total, f"{label}: batch {current} of {total}")
    if process.wait():
        raise MediaError(f"TensorRT post-processing failed:\n{chr(10).join(tail)}")
    missing = [job["output"] for job in jobs if not Path(str(job["output"])).exists()]
    if missing:
        raise MediaError("Post-processing did not create: " + ", ".join(map(str, missing)))
