from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG_PATH, load_config
from .visualizer import (
    DEFAULT_SEGMENTATION_MODEL,
    MOG2_LEARNING_RATE,
    SEGMENTATION_THRESHOLD,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal ASCII visualizer with camera silhouette + kick-reactive density."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.local.json created by scripts/select_devices.py",
    )
    parser.add_argument("--camera", type=int, help="Camera index override")
    parser.add_argument("--mic", type=int, help="Microphone device index override")
    parser.add_argument("--sample-rate", type=int, help="Audio sample rate override")
    parser.add_argument("--block-size", type=int, help="Audio block size override")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio input")
    parser.add_argument(
        "--segmentation",
        choices=["auto", "mediapipe", "mog2"],
        help="Segmentation mode (auto uses mediapipe if installed)",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        help=f"Segmentation threshold for mediapipe (default: {SEGMENTATION_THRESHOLD})",
    )
    parser.add_argument(
        "--mog2-learning-rate",
        type=float,
        help=f"MOG2 learning rate (default: {MOG2_LEARNING_RATE})",
    )
    parser.add_argument(
        "--seg-model",
        help=f"Segmentation model path for mediapipe tasks (default: {DEFAULT_SEGMENTATION_MODEL})",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    camera_index = args.camera if args.camera is not None else config.get("camera_index", 0)
    audio_device = None
    if not args.no_audio:
        audio_device = args.mic if args.mic is not None else config.get("audio_device_index")
    sample_rate = (
        args.sample_rate if args.sample_rate is not None else config.get("audio_sample_rate")
    )
    block_size = (
        args.block_size if args.block_size is not None else config.get("audio_block_size", 1024)
    )
    segmentation = args.segmentation if args.segmentation is not None else config.get(
        "segmentation", "auto"
    )
    segmentation_threshold = (
        args.seg_threshold
        if args.seg_threshold is not None
        else config.get("segmentation_threshold", SEGMENTATION_THRESHOLD)
    )
    mog2_learning_rate = (
        args.mog2_learning_rate
        if args.mog2_learning_rate is not None
        else config.get("mog2_learning_rate", MOG2_LEARNING_RATE)
    )
    segmentation_model_path = (
        args.seg_model
        if args.seg_model is not None
        else config.get("segmentation_model_path", str(DEFAULT_SEGMENTATION_MODEL))
    )

    run(
        camera_index=camera_index,
        audio_device_index=audio_device,
        audio_sample_rate=sample_rate,
        audio_block_size=block_size,
        segmentation=segmentation,
        segmentation_threshold=segmentation_threshold,
        mog2_learning_rate=mog2_learning_rate,
        segmentation_model_path=segmentation_model_path,
    )


if __name__ == "__main__":
    main()
