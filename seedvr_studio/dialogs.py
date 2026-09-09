"""Native file-picker dialogs opened by the local server on the user's desktop."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Lock

from .paths import VENV_PYTHON

VIDEO_EXTENSIONS = ("mp4", "mov", "mkv", "avi", "webm", "m4v", "mpeg", "mpg")
_DIALOG_LOCK = Lock()


class DialogUnavailable(RuntimeError):
    """No native dialog tool works in this environment."""


def _clean(result: subprocess.CompletedProcess[str]) -> str | None:
    if result.returncode != 0:
        return None
    value = result.stdout.strip().splitlines()
    return value[0].strip() if value and value[0].strip() else None


def _clean_many(result: subprocess.CompletedProcess[str]) -> list[str] | None:
    if result.returncode != 0:
        return None
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return values or None


def _has_display() -> bool:
    return os.name == "nt" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _kdialog(initial: Path) -> str | None:
    filters = f"Video files ({' '.join(f'*.{ext}' for ext in VIDEO_EXTENSIONS)})|All files (*)"
    return _clean(subprocess.run(["kdialog", "--title", "Choose a source video", "--getopenfilename", str(initial), filters], capture_output=True, text=True, timeout=3600))


def _zenity(initial: Path) -> str | None:
    pattern = " ".join(f"*.{ext}" for ext in VIDEO_EXTENSIONS)
    return _clean(subprocess.run(["zenity", "--file-selection", "--title=Choose a source video", f"--filename={initial}{os.sep}", f"--file-filter=Video files | {pattern}", "--file-filter=All files | *"], capture_output=True, text=True, timeout=3600))


def _yad(initial: Path) -> str | None:
    pattern = " ".join(f"*.{ext}" for ext in VIDEO_EXTENSIONS)
    return _clean(subprocess.run(["yad", "--file", "--title=Choose a source video", f"--filename={initial}{os.sep}", f"--file-filter=Video files | {pattern}", "--file-filter=All files | *"], capture_output=True, text=True, timeout=3600))


def _tkinter(initial: Path) -> str | None:
    script = (
        "import sys, tkinter\nfrom tkinter import filedialog\n"
        "root = tkinter.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
        "exts = ' '.join('*.' + e for e in sys.argv[2].split(','))\n"
        "path = filedialog.askopenfilename(title='Choose a source video', initialdir=sys.argv[1], filetypes=[('Video files', exts), ('All files', '*')])\n"
        "print(path or '')\n"
    )
    return _clean(subprocess.run([str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable), "-c", script, str(initial), ",".join(VIDEO_EXTENSIONS)], capture_output=True, text=True, timeout=3600))


def _powershell(initial: Path) -> str | None:
    filter_text = "Video files|" + ";".join(f"*.{ext}" for ext in VIDEO_EXTENSIONS) + "|All files|*.*"
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.OpenFileDialog; "
        f"$d.InitialDirectory = '{str(initial).replace(chr(39), chr(39) * 2)}'; "
        f"$d.Filter = '{filter_text}'; $d.Title = 'Choose a source video'; "
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }"
    )
    return _clean(subprocess.run(["powershell.exe", "-NoProfile", "-STA", "-Command", script], capture_output=True, text=True, timeout=3600))


# ----- multi-file and folder variants -----
def _kdialog_files(initial: Path) -> list[str] | None:
    filters = f"Video files ({' '.join(f'*.{ext}' for ext in VIDEO_EXTENSIONS)})|All files (*)"
    return _clean_many(subprocess.run(["kdialog", "--title", "Choose source videos", "--multiple", "--separate-output", "--getopenfilename", str(initial), filters], capture_output=True, text=True, timeout=3600))


def _zenity_files(initial: Path) -> list[str] | None:
    pattern = " ".join(f"*.{ext}" for ext in VIDEO_EXTENSIONS)
    return _clean_many(subprocess.run(["zenity", "--file-selection", "--multiple", "--separator=\n", "--title=Choose source videos", f"--filename={initial}{os.sep}", f"--file-filter=Video files | {pattern}", "--file-filter=All files | *"], capture_output=True, text=True, timeout=3600))


def _yad_files(initial: Path) -> list[str] | None:
    pattern = " ".join(f"*.{ext}" for ext in VIDEO_EXTENSIONS)
    return _clean_many(subprocess.run(["yad", "--file", "--multiple", "--separator=\n", "--title=Choose source videos", f"--filename={initial}{os.sep}", f"--file-filter=Video files | {pattern}", "--file-filter=All files | *"], capture_output=True, text=True, timeout=3600))


def _tkinter_files(initial: Path) -> list[str] | None:
    script = (
        "import sys, tkinter\nfrom tkinter import filedialog\n"
        "root = tkinter.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
        "exts = ' '.join('*.' + e for e in sys.argv[2].split(','))\n"
        "paths = filedialog.askopenfilenames(title='Choose source videos', initialdir=sys.argv[1], filetypes=[('Video files', exts), ('All files', '*')])\n"
        "print('\\n'.join(paths))\n"
    )
    return _clean_many(subprocess.run([str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable), "-c", script, str(initial), ",".join(VIDEO_EXTENSIONS)], capture_output=True, text=True, timeout=3600))


def _powershell_files(initial: Path) -> list[str] | None:
    filter_text = "Video files|" + ";".join(f"*.{ext}" for ext in VIDEO_EXTENSIONS) + "|All files|*.*"
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.OpenFileDialog; $d.Multiselect = $true; "
        f"$d.InitialDirectory = '{str(initial).replace(chr(39), chr(39) * 2)}'; "
        f"$d.Filter = '{filter_text}'; $d.Title = 'Choose source videos'; "
        "if ($d.ShowDialog() -eq 'OK') { $d.FileNames | ForEach-Object { Write-Output $_ } }"
    )
    return _clean_many(subprocess.run(["powershell.exe", "-NoProfile", "-STA", "-Command", script], capture_output=True, text=True, timeout=3600))


def _kdialog_folder(initial: Path) -> str | None:
    return _clean(subprocess.run(["kdialog", "--title", "Choose a folder of videos", "--getexistingdirectory", str(initial)], capture_output=True, text=True, timeout=3600))


def _zenity_folder(initial: Path) -> str | None:
    return _clean(subprocess.run(["zenity", "--file-selection", "--directory", "--title=Choose a folder of videos", f"--filename={initial}{os.sep}"], capture_output=True, text=True, timeout=3600))


def _yad_folder(initial: Path) -> str | None:
    return _clean(subprocess.run(["yad", "--file", "--directory", "--title=Choose a folder of videos", f"--filename={initial}{os.sep}"], capture_output=True, text=True, timeout=3600))


def _tkinter_folder(initial: Path) -> str | None:
    script = (
        "import sys, tkinter\nfrom tkinter import filedialog\n"
        "root = tkinter.Tk(); root.withdraw(); root.attributes('-topmost', True)\n"
        "print(filedialog.askdirectory(title='Choose a folder of videos', initialdir=sys.argv[1]) or '')\n"
    )
    return _clean(subprocess.run([str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable), "-c", script, str(initial)], capture_output=True, text=True, timeout=3600))


def _powershell_folder(initial: Path) -> str | None:
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
        f"$d.SelectedPath = '{str(initial).replace(chr(39), chr(39) * 2)}'; $d.Description = 'Choose a folder of videos'; "
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
    )
    return _clean(subprocess.run(["powershell.exe", "-NoProfile", "-STA", "-Command", script], capture_output=True, text=True, timeout=3600))


PICKERS = {
    "file": {"kdialog": _kdialog, "zenity": _zenity, "yad": _yad, "tkinter": _tkinter, "powershell": _powershell},
    "files": {"kdialog": _kdialog_files, "zenity": _zenity_files, "yad": _yad_files, "tkinter": _tkinter_files, "powershell": _powershell_files},
    "folder": {"kdialog": _kdialog_folder, "zenity": _zenity_folder, "yad": _yad_folder, "tkinter": _tkinter_folder, "powershell": _powershell_folder},
}


def available_pickers() -> list[str]:
    if os.name == "nt":
        return ["powershell"] if shutil.which("powershell.exe") else []
    if not _has_display():
        return []
    order = ["kdialog", "zenity", "yad"] if "kde" in os.environ.get("XDG_CURRENT_DESKTOP", "").lower() else ["zenity", "kdialog", "yad"]
    found = [name for name in order if shutil.which(name)]
    found.append("tkinter")
    return found


def _pick(kind: str, initial_dir: str | Path | None):
    pickers = PICKERS[kind]
    initial = Path(initial_dir).expanduser() if initial_dir else Path.home()
    if not initial.is_dir():
        initial = initial.parent if initial.parent.is_dir() else Path.home()
    names = available_pickers()
    if not names:
        raise DialogUnavailable("No desktop file dialog is available (no display, or no kdialog/zenity/yad/tkinter/PowerShell).")
    if not _DIALOG_LOCK.acquire(blocking=False):
        raise DialogUnavailable("A file dialog is already open.")
    try:
        errors: list[str] = []
        for name in names:
            try:
                return pickers[name](initial)
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{name}: {exc}")
        raise DialogUnavailable("File dialog failed: " + "; ".join(errors))
    finally:
        _DIALOG_LOCK.release()


def pick_video_file(initial_dir: str | Path | None = None) -> str | None:
    """Open a native "choose video" dialog; return the chosen path or None when cancelled."""
    return _pick("file", initial_dir)


def pick_video_files(initial_dir: str | Path | None = None) -> list[str] | None:
    """Multi-select variant; returns a list of paths or None when cancelled."""
    return _pick("files", initial_dir)


def pick_folder(initial_dir: str | Path | None = None) -> str | None:
    """Choose a folder; returns its path or None when cancelled."""
    return _pick("folder", initial_dir)
