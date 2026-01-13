from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen


DEFAULT_URLS = [
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float32/1/selfie_segmenter.tflite",
    "https://storage.googleapis.com/mediapipe-models/"
    "selfie_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
    "https://storage.googleapis.com/mediapipe-models/"
    "selfie_segmenter/selfie_segmenter/float32/1/selfie_segmenter.tflite",
    "https://storage.googleapis.com/mediapipe-assets/selfie_segmenter.tflite",
    "https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite",
]


def download(urls: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in urls:
        try:
            with urlopen(url) as response, destination.open("wb") as handle:
                handle.write(response.read())
            return
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
    if last_error:
        raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MediaPipe segmentation model.")
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="Model URL to download (can be supplied multiple times)",
    )
    parser.add_argument(
        "--output",
        default="assets/models/selfie_segmenter.tflite",
        help="Output path for the model",
    )
    args = parser.parse_args()

    destination = Path(args.output)
    if destination.exists():
        print(f"Model already exists at {destination}")
        return

    urls = args.urls or DEFAULT_URLS
    print(f"Downloading model to {destination} ...")
    print("Trying URLs:")
    for url in urls:
        print(f"  - {url}")
    try:
        download(urls, destination)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print("You can pass a custom URL with --url <model_url>.", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Download complete.")


if __name__ == "__main__":
    main()
