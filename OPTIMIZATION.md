# Optimization Guide

This document lists the main performance hotspots in `asciivisualizerpy`, why they are costly, and
concrete optimization ideas. The goal is to keep smooth frame rates while preserving the visuals.

## High‑Cost Areas (Most Likely Bottlenecks)

### 1) Camera → Resize → Grayscale pipeline
- File: `src/asciivisualizerpy/visualizer.py`
- Why it’s expensive:
  - Full‑resolution frames are captured then resized every frame.
  - Multiple color conversions (BGR→RGB) are done for segmentation + hands.
- Optimization ideas:
  - Reduce capture resolution via `cv2.CAP_PROP_FRAME_WIDTH/HEIGHT`.
  - Reuse the resized frame for both segmentation and hand detection.
  - Avoid repeated BGR→RGB conversions by sharing the converted buffer.
  - Skip frames for heavy models (e.g., run hand detection every N frames).

### 2) Segmentation (MediaPipe or MOG2)
- File: `src/asciivisualizerpy/visualizer.py`
- Why it’s expensive:
  - MediaPipe segmentation runs a model per frame.
  - MOG2 background subtraction still does full‑frame operations.
- Optimization ideas:
  - Use `--segmentation mog2` if MediaPipe is too slow.
  - Lower `SEGMENTATION_THRESHOLD` or reduce morphological operations.
  - Apply segmentation on a smaller downscaled frame and upsample the mask.
  - Skip segmentation on alternating frames and reuse the last mask.

### 3) Hand Detection (MediaPipe Tasks HandLandmarker)
- Files: `src/asciivisualizerpy/visualizer.py`, model download
- Why it’s expensive:
  - Model inference runs every frame by default.
  - HandLandmarker Tasks backend is heavier than simple landmarks.
- Optimization ideas:
  - Run detection every N frames and interpolate positions.
  - Drop `HAND_MODEL_COMPLEXITY` or lower confidences for faster tracking.
  - Use `--hand-backend solutions` if Tasks backend is slow on your machine.
  - Use a smaller `render_w`/`render_h` so the model sees fewer pixels.

### 4) Mesh Rendering and Hand‑Driven Warping
- File: `src/asciivisualizerpy/visualizer.py`
- Why it’s expensive:
  - Mesh draws many lines and points every frame.
  - Per‑point loops are Python‑level operations.
- Optimization ideas:
  - Reduce `MESH_PARTICLE_COUNT`.
  - Reduce line draw calls by only drawing edges, not all connections.
  - Cache per‑row neighbor indices to avoid recomputing them each frame.
  - Consider lowering `MESH_POINT_RADIUS`.

### 5) Halo + Ripple Effects
- File: `src/asciivisualizerpy/visualizer.py`
- Why it’s expensive:
  - Halo spike generation does multiple per‑frame geometry operations.
  - Ripples use dense numpy grids and exponentials.
- Optimization ideas:
  - Reduce `HALO_BANDS`, `HALO_CIRCLE_CURVE_SAMPLES`.
  - Disable `HALO_SOLID_FILL` and/or `HALO_TIP_OUTLINE`.
  - Reduce `RIPPLE_MAX_RINGS`, `RIPPLE_MAX_RADIUS_RATIO`.
  - Reuse grid caches aggressively (already partially cached).

### 6) Braille Dithering + Colorization
- File: `src/asciivisualizerpy/visualizer.py`
- Why it’s expensive:
  - Dithering and cell mapping are full‑frame operations.
  - Highlight/overlay colorization walks every character.
- Optimization ideas:
  - Increase `BRAILLE_CELL` to reduce output resolution.
  - Skip highlight overlay when not needed.
  - Consider disabling color during performance‑critical runs.

### 7) Terminal Output (I/O Bound)
- File: `src/asciivisualizerpy/visualizer.py`
- Why it’s expensive:
  - Writing the full frame to stdout each frame is costly.
- Optimization ideas:
  - Reduce output size by shrinking the terminal.
  - Cap FPS by sleeping for a small duration each frame.
  - Batch output with `sys.stdout.write` only once per frame (already done).

## Library‑Specific Opportunities

### OpenCV
- Use `cv2.resize` with smaller target sizes.
- Prefer `INTER_AREA` for downscaling (already used).
- Reduce expensive morphological operations or kernel sizes.

### MediaPipe
- Prefer `solutions.hands` if the Tasks backend is too heavy.
- Lower model complexity and tracking confidence thresholds.
- Run detection on downscaled frames.

### NumPy
- Avoid Python loops where possible; keep operations vectorized.
- Cache arrays (already using `grid_cache` and `braille_dither_cache`).

### sounddevice / audio
- If audio is not critical, run with `--no-audio` to reduce CPU.
- Use smaller FFT/feature bandwidth (reduce `HALO_BANDS`).

## Practical Tuning Checklist

1) Reduce terminal size and output resolution.
2) Reduce `MESH_PARTICLE_COUNT` and `MESH_POINT_RADIUS`.
3) Switch to `--segmentation mog2` if MediaPipe is heavy.
4) Run hand detection every N frames (simple throttle).
5) Disable halo/ripples if needed for performance.

## Suggested Future Improvements (Code Changes)

- Add a `--fps` cap and sleep to smooth CPU spikes.
- Add a `--hand-interval` to run hand detection every N frames.
- Cache hand landmarks for short bursts of missing frames.
- Add a `--low-power` preset that reduces all heavy effects at once.
