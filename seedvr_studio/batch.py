"""Sequential batch queue: full renders of many source videos, one at a time, persisted to disk."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable
from uuid import uuid4

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
FINISHED = {"done", "failed", "skipped", "cancelled"}

ProgressHook = Callable[[float, str], None]
Runner = Callable[[dict[str, Any], dict[str, Any], ProgressHook], dict[str, Any]]


OUTPUT_SUBFOLDER = "VRUpscale"


def output_folder(source: Path) -> Path:
    """Batch results live in a VRUpscale subfolder beside the sources."""
    return source.parent / OUTPUT_SUBFOLDER


def expected_output(source: Path) -> Path:
    return output_folder(source) / f"{source.stem}-seed.mp4"


def expand_paths(paths: list[str]) -> tuple[list[Path], list[str]]:
    """Turn folder and file paths into a sorted, de-duplicated list of video files."""
    found: list[Path] = []
    missing: list[str] = []
    for raw in paths:
        candidate = Path(str(raw).strip()).expanduser()
        if not str(raw).strip():
            continue
        if candidate.is_dir():
            found.extend(sorted(p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and not p.stem.endswith("-seed")))
        elif candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
            found.append(candidate)
        else:
            missing.append(str(candidate))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique, missing


class BatchQueue:
    """Ordered list of source videos rendered one after another by a worker thread.

    `runner(item, settings, progress)` performs one full render and returns a dict with at
    least `status` ("done", "failed" or "cancelled"), plus optional `message`, `output`, `job_id`.
    `cancel()` is called to abort the render in flight when the queue is stopped.
    """

    def __init__(self, store: Path, runner: Runner, cancel: Callable[[], Any] | None = None) -> None:
        self.store = store
        self.runner = runner
        self.cancel = cancel or (lambda: None)
        self.items: list[dict[str, Any]] = []
        self.state = "idle"
        self.current: str | None = None
        self.lock = Lock()
        self.thread: Thread | None = None
        self.load()

    # ----- persistence -----
    def load(self) -> None:
        try:
            data = json.loads(self.store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return
        with self.lock:
            self.items = []
            for item in items:
                if not isinstance(item, dict) or not item.get("path"):
                    continue
                if item.get("status") == "running":
                    item["status"], item["message"], item["progress"] = "queued", "Interrupted by restart; queued again", 0.0
                self.items.append(item)

    def save(self) -> None:
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            self.store.write_text(json.dumps({"version": 1, "items": self.items}, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ----- editing -----
    def add(self, paths: list[str], settings: dict[str, Any]) -> dict[str, Any]:
        files, missing = expand_paths(paths)
        added: list[dict[str, Any]] = []
        skipped: list[str] = []
        duplicates: list[str] = []
        with self.lock:
            queued_paths = {str(item["path"]) for item in self.items if item.get("status") not in FINISHED}
            for path in files:
                if str(path) in queued_paths:
                    duplicates.append(path.name)
                    continue
                item = {
                    "id": uuid4().hex[:8], "path": str(path), "name": path.name, "folder": str(path.parent),
                    "status": "queued", "message": "Queued", "progress": 0.0, "output": None, "job_id": None,
                    "added_at": time.time(), "settings": dict(settings),
                }
                target = expected_output(path)
                if target.exists():
                    item.update(status="skipped", message=f"{target.name} already exists", output=str(target))
                    skipped.append(path.name)
                self.items.append(item)
                added.append(item)
                queued_paths.add(str(path))
        self.save()
        return {"added": len(added) - len(skipped), "skipped": skipped, "missing": missing, "duplicates": duplicates}

    def remove(self, item_id: str) -> bool:
        with self.lock:
            for index, item in enumerate(self.items):
                if item["id"] == item_id:
                    if item["status"] == "running":
                        return False
                    del self.items[index]
                    self.save()
                    return True
        return False

    def retry(self, item_id: str) -> bool:
        with self.lock:
            for item in self.items:
                if item["id"] == item_id and item["status"] in FINISHED:
                    item.update(status="queued", message="Queued again", progress=0.0, output=None, job_id=None)
                    self.save()
                    return True
        return False

    def clear_finished(self) -> int:
        with self.lock:
            before = len(self.items)
            self.items = [item for item in self.items if item["status"] not in FINISHED]
            removed = before - len(self.items)
        self.save()
        return removed

    def move(self, item_id: str, direction: int) -> bool:
        with self.lock:
            ids = [item["id"] for item in self.items]
            if item_id not in ids:
                return False
            index = ids.index(item_id)
            target = index + direction
            if not 0 <= target < len(self.items) or self.items[index]["status"] == "running" or self.items[target]["status"] == "running":
                return False
            self.items[index], self.items[target] = self.items[target], self.items[index]
        self.save()
        return True

    # ----- running -----
    def start(self) -> bool:
        with self.lock:
            if self.state == "running":
                return True
            if not any(item["status"] == "queued" for item in self.items):
                return False
            self.state = "running"
            self.thread = Thread(target=self._loop, name="batch-worker", daemon=True)
            self.thread.start()
            return True

    def pause(self) -> None:
        with self.lock:
            if self.state == "running":
                self.state = "pausing"

    def stop(self) -> None:
        self.pause()
        self.cancel()

    def skip_current(self) -> None:
        """Abort the render in flight and continue with the next item."""
        self.cancel()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            items = [{key: value for key, value in item.items() if key != "settings"} for item in self.items]
            counts = {status: sum(1 for item in self.items if item["status"] == status) for status in ("queued", "running", "done", "failed", "skipped", "cancelled")}
            return {"state": self.state, "current": self.current, "items": items, "counts": counts}

    def _next_item(self) -> dict[str, Any] | None:
        with self.lock:
            if self.state != "running":
                return None
            for item in self.items:
                if item["status"] == "queued":
                    item.update(status="running", message="Starting", progress=0.0)
                    self.current = item["id"]
                    return item
        return None

    def _loop(self) -> None:
        try:
            while True:
                item = self._next_item()
                if item is None:
                    break
                self.save()

                def progress(value: float, message: str, item=item) -> None:
                    with self.lock:
                        item["progress"] = float(value)
                        item["message"] = message

                try:
                    result = self.runner(item, dict(item.get("settings") or {}), progress)
                except Exception as exc:  # the queue must survive a broken item
                    result = {"status": "failed", "message": str(exc)}
                with self.lock:
                    status = str(result.get("status") or "failed")
                    item.update(status=status if status in FINISHED else "failed", message=str(result.get("message") or status),
                                progress=1.0 if status == "done" else item["progress"], output=result.get("output"), job_id=result.get("job_id"),
                                finished_at=time.time())
                    self.current = None
                self.save()
        finally:
            with self.lock:
                self.state = "idle"
                self.current = None
            self.save()
