from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - hardware dependency
    sd = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from asciivisualizerpy.config import DEFAULT_CONFIG_PATH, save_config  # noqa: E402


def _camera_names_from_pyobjc() -> Dict[int, str]:
    if platform.system() != "Darwin":
        return {}
    try:
        import AVFoundation  # type: ignore[import-not-found]
    except Exception:
        return {}

    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    return {index: str(device.localizedName()) for index, device in enumerate(devices)}


def _camera_names_from_ffmpeg_avfoundation() -> Dict[int, str]:
    if platform.system() != "Darwin" or shutil.which("ffmpeg") is None:
        return {}
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    output = result.stderr or ""
    names: Dict[int, str] = {}
    in_video_section = False
    for line in output.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s(.+)$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    return names


def _camera_names_from_ffmpeg_dshow() -> Dict[int, str]:
    if platform.system() != "Windows" or shutil.which("ffmpeg") is None:
        return {}
    result = subprocess.run(
        ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
    )
    output = result.stderr or ""
    names: Dict[int, str] = {}
    in_video_section = False
    index = 0
    for line in output.splitlines():
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = re.search(r"\"([^\"]+)\"", line)
        if match:
            names[index] = match.group(1).strip()
            index += 1
    return names


def _camera_names_from_v4l2() -> Dict[int, str]:
    if platform.system() != "Linux":
        return {}
    names: Dict[int, str] = {}
    root = Path("/sys/class/video4linux")
    if not root.exists():
        return {}
    for device_dir in root.glob("video*"):
        name_file = device_dir / "name"
        if name_file.exists():
            index = int(device_dir.name.replace("video", ""))
            names[index] = name_file.read_text(encoding="utf-8").strip()
    return names


def _camera_name_map() -> Dict[int, str]:
    name_map = _camera_names_from_pyobjc()
    if name_map:
        return name_map
    name_map = _camera_names_from_ffmpeg_avfoundation()
    if name_map:
        return name_map
    name_map = _camera_names_from_v4l2()
    if name_map:
        return name_map
    return _camera_names_from_ffmpeg_dshow()


def _find_cameras(max_index: int) -> List[Tuple[int, str]]:
    cameras = []
    name_map = _camera_name_map()
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            name = name_map.get(index, f"Camera {index}")
            cameras.append((index, name))
        cap.release()
    return cameras


def _prompt_index(prompt: str, default: Optional[int]) -> Optional[int]:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            return default
        if raw.lower() in {"none", "n"}:
            return None
        if raw.isdigit():
            return int(raw)
        print("Enter a numeric index, or 'none'.")


def _list_microphones() -> List[Tuple[int, str, int, float]]:
    devices = sd.query_devices() if sd else []
    inputs = []
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            inputs.append(
                (index, device["name"], device["max_input_channels"], device["default_samplerate"])
            )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Select camera and microphone devices.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Config path to write (default: config.local.json)",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
        help="Maximum camera index to probe",
    )
    args = parser.parse_args()

    cameras = _find_cameras(args.max_camera_index)
    if cameras:
        print("Available cameras:")
        for index, name in cameras:
            print(f"  [{index}] {name}")
    else:
        print("No cameras found.")

    camera_default = cameras[0][0] if cameras else 0
    camera_index = _prompt_index("Select camera index", camera_default)

    audio_device_index = None
    audio_sample_rate = None
    audio_block_size = 1024

    if sd is None:
        print("sounddevice not installed; skipping microphone selection.")
    else:
        microphones = _list_microphones()
        if microphones:
            print("Available microphones:")
            for index, name, channels, sample_rate in microphones:
                print(f"  [{index}] {name} (inputs: {channels}, {int(sample_rate)} Hz)")
        else:
            print("No microphone inputs found.")

        default_input = sd.default.device[0]
        if default_input is None or default_input < 0:
            default_input = microphones[0][0] if microphones else None
        audio_device_index = _prompt_index("Select microphone index", default_input)

        if audio_device_index is not None:
            device_info = sd.query_devices(audio_device_index, "input")
            audio_sample_rate = int(device_info["default_samplerate"])

    config = {
        "camera_index": camera_index,
        "audio_device_index": audio_device_index,
        "audio_sample_rate": audio_sample_rate,
        "audio_block_size": audio_block_size,
    }

    config_path = save_config(config, Path(args.config))
    print(f"Saved device config to {config_path}")


if __name__ == "__main__":
    main()
