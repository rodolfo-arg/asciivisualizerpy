from __future__ import annotations

import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .audio import KickMeter

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None  # type: ignore[assignment]

CHAR_ASPECT = 0.5
FG_THRESHOLD = 200
SOLID_SILHOUETTE = False
SEGMENTATION_THRESHOLD = 0.35
MOG2_LEARNING_RATE = 0.002
RENDER_MODE = "braille"
COLOR_ENABLED = True
COLOR_PALETTE = [
    (80, 160, 255),
    (120, 220, 120),
    (255, 230, 120),
    (255, 180, 80),
    (255, 100, 80),
    (255, 80, 180),
    (200, 120, 255),
    (100, 120, 255),
    (80, 220, 220),
    (240, 240, 240),
]
COLOR_KICK_THRESHOLD = 0.25
COLOR_MIN_INTERVAL = 0.12
BRAILLE_WHITE_SPIKES = True
WHITE_COLOR = "\x1b[38;2;255;255;255m"
BRAILLE_CELL = (4, 2)
BRAILLE_KICK_BOOST = 0.25
BRAILLE_DITHER = (np.array([[0, 4], [6, 2], [3, 7], [5, 1]], dtype=np.float32) + 0.5) / 8.0
BRAILLE_LOOKUP = tuple(chr(0x2800 + value) for value in range(256))
RIPPLES_ENABLED = True
RIPPLE_KICK_THRESHOLD = 0.25
RIPPLE_MIN_INTERVAL = 0.12
RIPPLE_SPEED_RATIO = 0.55
RIPPLE_WIDTH_RATIO = 0.02
RIPPLE_DECAY = 1.4
RIPPLE_BOOST = 0.35
RIPPLE_MASK_THRESHOLD = 0.35
RIPPLE_MAX_RINGS = 3
RIPPLE_MAX_RADIUS_RATIO = 1.2
RIPPLE_INSIDE_ONLY = False
RIPPLE_ORIGIN = "centroid"
SHELF_ENABLED = False
SHELF_BANDS = 3
SHELF_BASE = 0.12
SHELF_GAIN = 0.85
SHELF_MAX_HEIGHT_RATIO = 0.4
SHELF_MIN_HEIGHT = 1
SHELF_WAVE_FREQ = 2.6
SHELF_SPEED = 2.0
SHELF_PARABOLA_POWER = 1.0
SHELF_OUTSIDE_ONLY = True
SHELF_OUTSET = 2
SHELF_ARC_RATIO = 0.25
SHELF_SAMPLE_STEP = 1
SHELF_MODE = "distributed"
SHELF_SWEEP_SPEED = 0.08
SHELF_RANDOM_HOLD = 0.5
SHELF_THICKNESS = 2
SHELF_SMOOTH_WINDOW = 7
SHELF_BASE_OFFSET_RATIO = 0.18
SHELF_SPIKE_RATIO = 0.55
HALO_ENABLED = True
HALO_BANDS = 12
HALO_LOW_HZ = 80.0
HALO_HIGH_HZ = 8000.0
HALO_MAX_HEIGHT_RATIO = 0.45
HALO_MIN_HEIGHT = 1
HALO_BASE_OFFSET_RATIO = 0.12
HALO_OUTSET = 2
HALO_SAMPLE_STEP = 1
HALO_SMOOTH_WINDOW = 5
HALO_THICKNESS = 1
HALO_CONNECT_TIPS = False
HALO_CONNECT_THICKNESS = 1
HALO_CONNECT_STRENGTH = 0.3
HALO_GAMMA = 1.6
HALO_KICK_GAIN = 0.35
HALO_ROTATE_SPEED = 0.0
HALO_OUTSIDE_ONLY = True
HALO_INTENSITY_GAIN = 0.9
HALO_INTENSITY_KICK = 0.35
HALO_INTERP_BANDS = False
HALO_SOLID_FILL = True
HALO_SPIKE_COUNT = 1
HALO_SPIKE_WIDTH_RATIO = 1.6
HALO_SPIKE_MIN_WIDTH = 7
HALO_SPIKE_MAX_RATIO = 0.2
HALO_SPIKE_PARABOLA = 1.4
HALO_TIP_OUTLINE = True
HALO_GEOMETRY = "circle"
HALO_CIRCLE_TOP_START_ANGLE = math.pi * 0.85
HALO_CIRCLE_TOP_END_ANGLE = 0.0
HALO_CIRCLE_BOTTOM_START_ANGLE = 0.0
HALO_CIRCLE_BOTTOM_END_ANGLE = math.tau - HALO_CIRCLE_TOP_START_ANGLE
HALO_CIRCLE_TOP_SPIKES = 1
HALO_CIRCLE_BOTTOM_SPIKES = 6
HALO_CIRCLE_SPIKE_ANGLE = 0.6
HALO_CIRCLE_BASE_WIDTH_RATIO = 0.16
HALO_CIRCLE_BASE_WIDTH_MIN = 6
HALO_CIRCLE_CURVE_SAMPLES = 11
HALO_CIRCLE_PARABOLA = 1.0
HALO_CIRCLE_CENTER_X_OFFSET = 0.0
HALO_CIRCLE_CENTER_Y_OFFSET = 0.5
HALO_CIRCLE_RADIUS_SCALE = 0.7
RAMP_LEVELS = [
    "░▒▓",
    "░▒▓█",
    "░▒▓██",
    "░▒▓███",
    "░▒▓████",
]
SPIKES_ENABLED = True
SPIKE_MIN_KICK = 0.05
SPIKE_MAX_RADIUS = 6
SPIKE_GAMMA = 1.4
SPIKE_DENSITY_BASE = 0.08
SPIKE_DENSITY_GAIN = 0.45
SPIKE_OUTSIDE_ONLY = True
SPIKE_BOOST_VALUE = 255
DEFAULT_SEGMENTATION_MODEL = (
    Path(__file__).resolve().parents[2] / "assets" / "models" / "selfie_segmenter.tflite"
)


def _build_lookup(ramp: str) -> Tuple[str, ...]:
    steps = len(ramp) - 1
    return tuple(ramp[int(value / 255 * steps)] for value in range(256))


def _pick_ramp(level: float) -> str:
    index = int(round(level * (len(RAMP_LEVELS) - 1)))
    index = max(0, min(index, len(RAMP_LEVELS) - 1))
    return RAMP_LEVELS[index]


def _color_for_rgb(color: Tuple[int, int, int]) -> str:
    if not COLOR_ENABLED:
        return ""
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m"


def _grid_for_shape(
    shape: Tuple[int, int],
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    cached = cache.get(shape)
    if cached is not None:
        return cached
    height, width = shape
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    cache[shape] = (x_grid, y_grid)
    return x_grid, y_grid


def _ripple_origin(mask: np.ndarray, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if not np.any(mask):
        return fallback
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        return fallback
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _mask_bounds(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, width, height = cv2.boundingRect(coords)
    return x, x + width - 1, y, y + height - 1


def _ray_hit_mask(
    mask: np.ndarray,
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    max_steps: int,
) -> Tuple[float, float] | None:
    height, width = mask.shape
    ox, oy = origin
    dx, dy = direction
    last = None
    for step in range(max_steps):
        x = int(round(ox + dx * step))
        y = int(round(oy + dy * step))
        if x < 0 or x >= width or y < 0 or y >= height:
            break
        if mask[y, x] > 0:
            last = (float(x), float(y))
        elif last is not None:
            break
    return last


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 6:
        return None
    return contour[:, 0, :].astype(np.float32)


def _smooth_closed_curve(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or points.shape[0] <= window:
        return points
    window = window if window % 2 == 1 else window + 1
    pad = window // 2
    extended = np.vstack([points[-pad:], points, points[:pad]])
    kernel = np.ones(window, dtype=np.float32) / float(window)
    xs = np.convolve(extended[:, 0], kernel, mode="valid")
    ys = np.convolve(extended[:, 1], kernel, mode="valid")
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _braille_dither(shape: Tuple[int, int], cache: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
    cached = cache.get(shape)
    if cached is not None:
        return cached
    height, width = shape
    tile_h = int(np.ceil(height / BRAILLE_CELL[0]))
    tile_w = int(np.ceil(width / BRAILLE_CELL[1]))
    dither = np.tile(BRAILLE_DITHER, (tile_h, tile_w))[:height, :width]
    cache[shape] = dither
    return dither


def _frame_to_braille(
    gray: np.ndarray,
    mask: np.ndarray,
    kick_level: float,
    spike_mask: np.ndarray | None,
    ripple_intensity: np.ndarray | None,
    dither_cache: Dict[Tuple[int, int], np.ndarray],
    highlight_mask: np.ndarray | None = None,
) -> Tuple[list[str], np.ndarray | None]:
    cell_h, cell_w = BRAILLE_CELL
    h_cells = gray.shape[0] // cell_h
    w_cells = gray.shape[1] // cell_w
    if h_cells <= 0 or w_cells <= 0:
        return [], None

    height = h_cells * cell_h
    width = w_cells * cell_w
    gray = gray[:height, :width]
    mask = mask[:height, :width]
    if spike_mask is not None:
        spike_mask = spike_mask[:height, :width]

    ink = (255 - gray).astype(np.float32) / 255.0
    if BRAILLE_KICK_BOOST > 0.0:
        ink = np.clip(ink + kick_level * BRAILLE_KICK_BOOST, 0.0, 1.0)
    if ripple_intensity is not None:
        ripple_intensity = ripple_intensity[:height, :width]
        ink = np.clip(ink + ripple_intensity * RIPPLE_BOOST, 0.0, 1.0)
    if spike_mask is not None:
        ink[spike_mask > 0] = 1.0

    thresholds = _braille_dither((height, width), dither_cache)
    dots = (ink > thresholds) & (mask > 0)
    dots = dots.reshape(h_cells, cell_h, w_cells, cell_w).astype(np.uint8)

    bits = (
        dots[:, 0, :, 0] * 1
        + dots[:, 1, :, 0] * 2
        + dots[:, 2, :, 0] * 4
        + dots[:, 0, :, 1] * 8
        + dots[:, 1, :, 1] * 16
        + dots[:, 2, :, 1] * 32
        + dots[:, 3, :, 0] * 64
        + dots[:, 3, :, 1] * 128
    )

    lines = ["".join(BRAILLE_LOOKUP[value] for value in row) for row in bits]
    highlight_cells = None
    if highlight_mask is not None:
        highlight_mask = highlight_mask[:height, :width]
        highlight = highlight_mask > 0
        highlight_cells = highlight.reshape(h_cells, cell_h, w_cells, cell_w).any(
            axis=(1, 3)
        )

    return lines, highlight_cells


def _colorize_braille(
    lines: list[str],
    highlight_cells: np.ndarray | None,
    base_color: str,
    highlight_color: str,
) -> str:
    if highlight_cells is None or not np.any(highlight_cells):
        return "\n".join(lines)

    reset = "\x1b[0m"
    rendered = []
    for row_index, line in enumerate(lines):
        row_mask = highlight_cells[row_index]
        out = []
        if base_color:
            out.append(base_color)
        in_highlight = False
        for col_index, ch in enumerate(line):
            want_highlight = bool(row_mask[col_index])
            if want_highlight and not in_highlight:
                out.append(highlight_color)
                in_highlight = True
            elif not want_highlight and in_highlight:
                out.append(base_color if base_color else reset)
                in_highlight = False
            out.append(ch)
        if in_highlight:
            out.append(base_color if base_color else reset)
        rendered.append("".join(out))

    return "\n".join(rendered)


def _compute_output_size(
    frame_size: Tuple[int, int],
    term_size: Tuple[int, int],
    status_lines: int = 1,
) -> Tuple[int, int]:
    frame_w, frame_h = frame_size
    term_w, term_h = term_size
    usable_h = max(1, term_h - status_lines)

    aspect = frame_h / max(frame_w, 1)
    target_w = max(1, term_w)
    target_h = int(target_w * aspect * CHAR_ASPECT)

    if target_h > usable_h:
        target_h = usable_h
        target_w = max(1, int(target_h / (aspect * CHAR_ASPECT)))

    return max(1, target_w), max(1, target_h)


def _frame_to_ascii(
    gray: np.ndarray,
    mask: np.ndarray,
    lookup: Tuple[str, ...],
    solid_silhouette: bool,
    boost_mask: np.ndarray | None = None,
    boost_value: int = SPIKE_BOOST_VALUE,
) -> str:
    lines = []
    if boost_mask is None:
        for row_gray, row_mask in zip(gray, mask):
            line_chars = [
                lookup[255 if solid_silhouette else pixel] if mask_value else " "
                for pixel, mask_value in zip(row_gray, row_mask)
            ]
            lines.append("".join(line_chars))
    else:
        for row_gray, row_mask, row_boost in zip(gray, mask, boost_mask):
            line = []
            for pixel, mask_value, boost_value_flag in zip(row_gray, row_mask, row_boost):
                if not mask_value:
                    line.append(" ")
                    continue
                value = 255 if solid_silhouette else int(pixel)
                if boost_value_flag:
                    value = max(value, boost_value)
                line.append(lookup[value])
            lines.append("".join(line))
    return "\n".join(lines)


def _format_status(fps: float, kick_level: float, audio_enabled: bool) -> str:
    if audio_enabled:
        return f"FPS: {fps:5.1f}  Kick: {kick_level:0.2f}"
    return f"FPS: {fps:5.1f}  Kick: --"


def run(
    camera_index: int = 0,
    audio_device_index: int | None = None,
    audio_sample_rate: int | None = None,
    audio_block_size: int = 1024,
    segmentation: str = "auto",
    segmentation_threshold: float = SEGMENTATION_THRESHOLD,
    mog2_learning_rate: float = MOG2_LEARNING_RATE,
    segmentation_model_path: str | Path | None = None,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    segmentation_mode = segmentation.lower()
    if segmentation_mode == "auto":
        segmentation_mode = "mediapipe" if mp is not None else "mog2"
    if segmentation_mode not in {"mediapipe", "mog2"}:
        raise ValueError("segmentation must be 'auto', 'mediapipe', or 'mog2'")

    selfie = None
    segmenter = None
    segmenter_backend = "mog2"
    mediapipe_warning = None
    if segmentation_mode in {"mediapipe", "auto"}:
        if mp is None:
            if segmentation_mode == "mediapipe":
                raise RuntimeError("mediapipe is not installed; use --segmentation mog2")
            mediapipe_warning = "mediapipe is not installed"
        else:
            has_solutions = hasattr(mp, "solutions") and hasattr(
                mp.solutions, "selfie_segmentation"
            )
            if has_solutions:
                selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                segmenter_backend = "solutions"
            else:
                try:
                    from mediapipe.tasks.python import vision
                    from mediapipe.tasks.python.core import base_options
                except Exception as exc:  # pragma: no cover - optional dependency
                    if segmentation_mode == "mediapipe":
                        raise RuntimeError("mediapipe tasks are unavailable") from exc
                    mediapipe_warning = "mediapipe tasks are unavailable"
                else:
                    model_path = Path(segmentation_model_path or DEFAULT_SEGMENTATION_MODEL)
                    if not model_path.exists():
                        if segmentation_mode == "mediapipe":
                            raise RuntimeError(
                                "Segmentation model not found at "
                                f"{model_path}. Run scripts/download_segmentation_model.py "
                                "or pass --seg-model to point at a local .tflite file."
                            )
                        mediapipe_warning = f"segmentation model not found at {model_path}"
                    else:
                        options = vision.ImageSegmenterOptions(
                            base_options=base_options.BaseOptions(
                                model_asset_path=str(model_path)
                            ),
                            running_mode=vision.RunningMode.IMAGE,
                            output_confidence_masks=True,
                            output_category_mask=False,
                        )
                        segmenter = vision.ImageSegmenter.create_from_options(options)
                        segmenter_backend = "tasks"

    if segmentation_mode == "auto" and segmenter_backend == "mog2" and mediapipe_warning:
        print(
            f"MediaPipe unavailable ({mediapipe_warning}); falling back to MOG2.",
            file=sys.stderr,
        )

    back_sub = None
    if segmenter_backend == "mog2":
        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=False,
        )

    band_count = HALO_BANDS if HALO_ENABLED else 0
    kick_meter = KickMeter(
        device=audio_device_index,
        samplerate=audio_sample_rate,
        blocksize=audio_block_size,
        band_count=band_count,
        band_low_hz=HALO_LOW_HZ,
        band_high_hz=HALO_HIGH_HZ,
    )
    audio_enabled = True
    try:
        kick_meter.start()
    except Exception as exc:  # pragma: no cover - runtime hardware dependency
        audio_enabled = False
        print(f"Audio disabled: {exc}", file=sys.stderr)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Unable to read from camera")

    term_size = shutil.get_terminal_size((80, 24))
    out_w, out_h = _compute_output_size(
        (frame.shape[1], frame.shape[0]),
        (term_size.columns, term_size.lines),
    )

    lookup_cache: Dict[str, Tuple[str, ...]] = {}
    braille_dither_cache: Dict[Tuple[int, int], np.ndarray] = {}
    grid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    spike_kernels: Dict[int, np.ndarray] = {}
    rng = np.random.default_rng()
    ripples: list[Tuple[float, float, float, float]] = []
    last_ripple_time = -1.0
    last_kick_level = 0.0
    color_index = 0
    last_color_time = -1.0
    last_color_kick = 0.0
    shelf_center_index = 0
    last_shelf_change = -1.0
    last_time = time.monotonic()
    fps = 0.0

    sys.stdout.write("\x1b[2J\x1b[H\x1b[?25l")
    sys.stdout.flush()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_time = time.monotonic()

            term_size = shutil.get_terminal_size((80, 24))
            new_w, new_h = _compute_output_size(
                (frame.shape[1], frame.shape[0]),
                (term_size.columns, term_size.lines),
            )
            if (new_w, new_h) != (out_w, out_h):
                out_w, out_h = new_w, new_h

            if RENDER_MODE == "braille":
                render_w = out_w * BRAILLE_CELL[1]
                render_h = out_h * BRAILLE_CELL[0]
            else:
                render_w = out_w
                render_h = out_h
            resized = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            if segmenter_backend == "solutions":
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                results = selfie.process(rgb)
                if results.segmentation_mask is None:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                else:
                    mask = (results.segmentation_mask > segmentation_threshold).astype(
                        np.uint8
                    ) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            elif segmenter_backend == "tasks":
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = segmenter.segment(image)
                if not results.confidence_masks:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                else:
                    confidence = results.confidence_masks[0].numpy_view()
                    mask = (confidence > segmentation_threshold).astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            else:
                fgmask = back_sub.apply(resized, learningRate=mog2_learning_rate)
                _, mask = cv2.threshold(fgmask, FG_THRESHOLD, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

            kick_level = kick_meter.read_level() if audio_enabled else 0.0
            band_levels = (
                kick_meter.read_bands() if audio_enabled else np.empty(0, dtype=np.float32)
            )
            if RENDER_MODE != "braille":
                ramp = _pick_ramp(kick_level)
                lookup = lookup_cache.setdefault(ramp, _build_lookup(ramp))
            if COLOR_ENABLED and COLOR_PALETTE:
                if (
                    kick_level >= COLOR_KICK_THRESHOLD
                    and last_color_kick < COLOR_KICK_THRESHOLD
                    and (frame_time - last_color_time) >= COLOR_MIN_INTERVAL
                ):
                    color_index = (color_index + 1) % len(COLOR_PALETTE)
                    last_color_time = frame_time
                last_color_kick = kick_level
                color = _color_for_rgb(COLOR_PALETTE[color_index])
            else:
                color = ""

            base_mask = mask
            ripple_intensity = None
            if RIPPLES_ENABLED:
                if (
                    kick_level >= RIPPLE_KICK_THRESHOLD
                    and last_kick_level < RIPPLE_KICK_THRESHOLD
                    and (frame_time - last_ripple_time) >= RIPPLE_MIN_INTERVAL
                    and len(ripples) < RIPPLE_MAX_RINGS
                ):
                    fallback = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
                    if RIPPLE_ORIGIN == "centroid":
                        origin_x, origin_y = _ripple_origin(base_mask, fallback)
                    else:
                        origin_x, origin_y = fallback
                    ripples.append((frame_time, origin_x, origin_y, kick_level))
                    last_ripple_time = frame_time
                last_kick_level = kick_level

                if ripples:
                    height, width = gray.shape
                    x_grid, y_grid = _grid_for_shape((height, width), grid_cache)
                    min_dim = max(1.0, float(min(height, width)))
                    speed = max(1.0, RIPPLE_SPEED_RATIO * min_dim)
                    width_sigma = max(1.0, RIPPLE_WIDTH_RATIO * min_dim)
                    max_radius = max(height, width) * RIPPLE_MAX_RADIUS_RATIO
                    intensity = np.zeros((height, width), dtype=np.float32)
                    active_ripples: list[Tuple[float, float, float, float]] = []
                    for start_time, origin_x, origin_y, amplitude in ripples:
                        age = frame_time - start_time
                        if age < 0:
                            continue
                        radius = speed * age
                        if radius > max_radius:
                            continue
                        strength = amplitude * np.exp(-RIPPLE_DECAY * age)
                        if strength <= 0.01:
                            continue
                        dist = np.hypot(x_grid - origin_x, y_grid - origin_y)
                        ring = np.exp(-0.5 * ((dist - radius) / width_sigma) ** 2)
                        intensity += ring * strength
                        active_ripples.append((start_time, origin_x, origin_y, amplitude))
                    ripples = active_ripples
                    if np.any(intensity):
                        if RIPPLE_INSIDE_ONLY:
                            intensity *= (base_mask > 0)
                        else:
                            intensity *= (base_mask == 0)
                        ripple_intensity = np.clip(intensity, 0.0, 1.0)

            halo_mask = None
            halo_intensity = None
            if HALO_ENABLED and band_levels.size > 0 and np.any(base_mask):
                bounds = _mask_bounds(base_mask)
                contour = _largest_contour(base_mask)
                if bounds is not None and contour is not None:
                    x_min, x_max, y_min, y_max = bounds
                    width = x_max - x_min + 1
                    height = y_max - y_min + 1
                    if width > 4 and height > 4 and contour.shape[0] > 12:
                        halo_mask = np.zeros_like(base_mask)
                        halo_intensity = np.zeros_like(gray, dtype=np.float32)
                        max_height = max(1.0, min(width, height) * HALO_MAX_HEIGHT_RATIO)
                        base_offset = HALO_OUTSET + max_height * HALO_BASE_OFFSET_RATIO
                        band_count = band_levels.size
                        spike_count = max(1, HALO_SPIKE_COUNT)
                        def band_level_for_fraction(frac: float) -> float:
                            if band_count <= 1:
                                return float(band_levels[0]) if band_count else 0.0
                            pos = frac * (band_count - 1)
                            if HALO_INTERP_BANDS:
                                idx = int(math.floor(pos))
                                next_idx = min(idx + 1, band_count - 1)
                                t = pos - idx
                                level = (
                                    (1.0 - t) * float(band_levels[idx])
                                    + t * float(band_levels[next_idx])
                                )
                            else:
                                idx = int(round(pos))
                                level = float(band_levels[idx])
                            return max(0.0, min(1.0, level))

                        if HALO_GEOMETRY == "circle":
                            center = _ripple_origin(
                                base_mask,
                                (x_min + width / 2.0, y_min + height / 2.0),
                            )
                            center = (
                                center[0] + width * HALO_CIRCLE_CENTER_X_OFFSET,
                                center[1] + height * HALO_CIRCLE_CENTER_Y_OFFSET,
                            )
                            center = (
                                float(np.clip(center[0], 0, gray.shape[1] - 1)),
                                float(np.clip(center[1], 0, gray.shape[0] - 1)),
                            )
                            max_steps = int(max(width, height) * 2)
                            top_spikes = max(0, HALO_CIRCLE_TOP_SPIKES)
                            bottom_spikes = max(0, HALO_CIRCLE_BOTTOM_SPIKES)
                            arcs: list[Tuple[float, float, int]] = []
                            if top_spikes:
                                arcs.append(
                                    (
                                        HALO_CIRCLE_TOP_START_ANGLE,
                                        HALO_CIRCLE_TOP_END_ANGLE,
                                        top_spikes,
                                    )
                                )
                            if bottom_spikes:
                                arcs.append(
                                    (
                                        HALO_CIRCLE_BOTTOM_START_ANGLE,
                                        HALO_CIRCLE_BOTTOM_END_ANGLE,
                                        bottom_spikes,
                                    )
                                )
                            total_spikes = sum(spikes for _, _, spikes in arcs)
                            if total_spikes == 0:
                                arcs = []

                            global_index = 0
                            for start_angle, end_angle, spikes in arcs:
                                if spikes == 1:
                                    angles = [start_angle]
                                else:
                                    angles = [
                                        start_angle
                                        + (idx / (spikes - 1)) * (end_angle - start_angle)
                                        for idx in range(spikes)
                                    ]
                                for angle in angles:
                                    frac = (
                                        global_index / max(1, total_spikes - 1)
                                        if total_spikes > 1
                                        else 0.0
                                    )
                                    level = band_level_for_fraction(frac)
                                    global_index += 1
                                    dx = math.cos(angle)
                                    dy = -math.sin(angle)
                                    hit = _ray_hit_mask(base_mask, center, (dx, dy), max_steps)
                                    if hit is not None:
                                        base_radius = math.hypot(
                                            hit[0] - center[0], hit[1] - center[1]
                                        )
                                    else:
                                        base_radius = min(width, height) * 0.25
                                    base_radius = (
                                        base_radius * HALO_CIRCLE_RADIUS_SCALE + base_offset
                                    )

                                    spike_height = (level**HALO_GAMMA) * max_height
                                    spike_height += kick_level * HALO_KICK_GAIN * max_height
                                    tip_radius = base_radius + spike_height
                                    base_width = max(
                                        HALO_CIRCLE_BASE_WIDTH_MIN,
                                        min(width, height) * HALO_CIRCLE_BASE_WIDTH_RATIO,
                                    )
                                    perp_x = -dy
                                    perp_y = dx
                                    sample_count = max(3, int(HALO_CIRCLE_CURVE_SAMPLES))
                                    if sample_count % 2 == 0:
                                        sample_count += 1
                                    curve_points = []
                                    for sample_idx in range(sample_count):
                                        t = sample_idx / max(1, sample_count - 1)
                                        offset = (t - 0.5) * base_width
                                        parabola = max(0.0, 1.0 - (2.0 * t - 1.0) ** 2)
                                        shape = parabola**HALO_CIRCLE_PARABOLA
                                        radius = base_radius + spike_height * shape
                                        x = center[0] + dx * radius + perp_x * offset
                                        y = center[1] + dy * radius + perp_y * offset
                                        curve_points.append(
                                            (
                                                int(np.clip(x, 0, gray.shape[1] - 1)),
                                                int(np.clip(y, 0, gray.shape[0] - 1)),
                                            )
                                        )
                                    tri = np.array(curve_points, dtype=np.int32)
                                    strength = min(
                                        1.0,
                                        level * HALO_INTENSITY_GAIN
                                        + kick_level * HALO_INTENSITY_KICK,
                                    )
                                    if HALO_SOLID_FILL:
                                        cv2.fillPoly(halo_mask, [tri], 255)
                                        cv2.fillPoly(halo_intensity, [tri], float(strength))
                                    if HALO_TIP_OUTLINE:
                                        base_mid = (
                                            int(
                                                np.clip(
                                                    center[0] + dx * base_radius,
                                                    0,
                                                    gray.shape[1] - 1,
                                                )
                                            ),
                                            int(
                                                np.clip(
                                                    center[1] + dy * base_radius,
                                                    0,
                                                    gray.shape[0] - 1,
                                                )
                                            ),
                                        )
                                        tip_point = (
                                            int(
                                                np.clip(
                                                    center[0] + dx * tip_radius,
                                                    0,
                                                    gray.shape[1] - 1,
                                                )
                                            ),
                                            int(
                                                np.clip(
                                                    center[1] + dy * tip_radius,
                                                    0,
                                                    gray.shape[0] - 1,
                                                )
                                            ),
                                        )
                                        cv2.line(
                                            halo_mask,
                                            base_mid,
                                            tip_point,
                                            255,
                                            HALO_THICKNESS,
                                        )
                                        cv2.line(
                                            halo_intensity,
                                            base_mid,
                                            tip_point,
                                            float(strength),
                                            HALO_THICKNESS,
                                        )
                        else:
                            points = _smooth_closed_curve(contour, HALO_SMOOTH_WINDOW)
                            count = points.shape[0]
                            normals = np.zeros_like(points)
                            for idx in range(count):
                                prev_point = points[(idx - 1) % count]
                                next_point = points[(idx + 1) % count]
                                tangent = next_point - prev_point
                                length = math.hypot(float(tangent[0]), float(tangent[1]))
                                if length < 1e-3:
                                    continue
                                normal = np.array(
                                    [-tangent[1], tangent[0]], dtype=np.float32
                                ) / length
                                sample = points[idx] + normal * 2.0
                                sx = int(np.clip(sample[0], 0, gray.shape[1] - 1))
                                sy = int(np.clip(sample[1], 0, gray.shape[0] - 1))
                                if base_mask[sy, sx] > 0:
                                    normal = -normal
                                normals[idx] = normal

                            spike_step = max(1, int(count / spike_count))
                            spike_width = max(
                                HALO_SPIKE_MIN_WIDTH, int(spike_step * HALO_SPIKE_WIDTH_RATIO)
                            )
                            max_width = max(3, int(count * HALO_SPIKE_MAX_RATIO))
                            spike_width = min(spike_width, max_width)
                            if spike_width % 2 == 0:
                                spike_width += 1
                            half_width = spike_width // 2

                            center_indices = [
                                (idx * spike_step) % count for idx in range(spike_count)
                            ]
                            for spike_idx, center_index in enumerate(center_indices):
                                frac = (
                                    spike_idx / max(1, spike_count - 1)
                                    if spike_count > 1
                                    else 0.0
                                )
                                level = band_level_for_fraction(frac)
                                base_points: list[Tuple[int, int]] = []
                                tip_points: list[Tuple[int, int]] = []
                                tip_strengths: list[float] = []
                                for offset in range(
                                    -half_width, half_width + 1, HALO_SAMPLE_STEP
                                ):
                                    idx = (center_index + offset) % count
                                    normal = normals[idx]
                                    if normal[0] == 0 and normal[1] == 0:
                                        continue
                                    t = (offset + half_width) / max(1, spike_width - 1)
                                    parabola = max(0.0, 1.0 - (2.0 * t - 1.0) ** 2)
                                    shape = parabola**HALO_SPIKE_PARABOLA
                                    if shape <= 0.0:
                                        continue
                                    spike = (level**HALO_GAMMA) * shape * max_height
                                    spike += kick_level * HALO_KICK_GAIN * shape * max_height
                                    offset_px = base_offset + spike
                                    if offset_px < HALO_MIN_HEIGHT:
                                        continue
                                    point = points[idx]
                                    offset_point = point + normal * offset_px
                                    x0 = int(np.clip(point[0], 0, gray.shape[1] - 1))
                                    y0 = int(np.clip(point[1], 0, gray.shape[0] - 1))
                                    x1 = int(np.clip(offset_point[0], 0, gray.shape[1] - 1))
                                    y1 = int(np.clip(offset_point[1], 0, gray.shape[0] - 1))
                                    strength = min(
                                        1.0,
                                        (
                                            level * HALO_INTENSITY_GAIN
                                            + kick_level * HALO_INTENSITY_KICK
                                        )
                                        * shape,
                                    )
                                    base_points.append((x0, y0))
                                    tip_points.append((x1, y1))
                                    tip_strengths.append(strength)

                                if len(base_points) < 2:
                                    continue

                                for idx in range(len(base_points) - 1):
                                    p0 = base_points[idx]
                                    p1 = base_points[idx + 1]
                                    t0 = tip_points[idx]
                                    t1 = tip_points[idx + 1]
                                    strength = (
                                        tip_strengths[idx] + tip_strengths[idx + 1]
                                    ) * 0.5
                                    if HALO_SOLID_FILL:
                                        quad = np.array([p0, p1, t1, t0], dtype=np.int32)
                                        cv2.fillPoly(halo_mask, [quad], 255)
                                        cv2.fillPoly(
                                            halo_intensity, [quad], float(strength)
                                        )

                                if HALO_TIP_OUTLINE:
                                    center_idx = len(base_points) // 2
                                    p0 = base_points[center_idx]
                                    t0 = tip_points[center_idx]
                                    strength = tip_strengths[center_idx]
                                    cv2.line(halo_mask, p0, t0, 255, HALO_THICKNESS)
                                    cv2.line(
                                        halo_intensity,
                                        p0,
                                        t0,
                                        float(strength),
                                        HALO_THICKNESS,
                                    )

                                if HALO_CONNECT_TIPS:
                                    for idx in range(1, len(tip_points)):
                                        p0 = tip_points[idx - 1]
                                        p1 = tip_points[idx]
                                        strength = (
                                            tip_strengths[idx - 1] + tip_strengths[idx]
                                        ) * 0.5 * HALO_CONNECT_STRENGTH
                                        strength = min(1.0, strength)
                                        cv2.line(
                                            halo_mask,
                                            p0,
                                            p1,
                                            255,
                                            HALO_CONNECT_THICKNESS,
                                        )
                                        cv2.line(
                                            halo_intensity,
                                            p0,
                                            p1,
                                            float(strength),
                                            HALO_CONNECT_THICKNESS,
                                        )

                        if HALO_OUTSIDE_ONLY:
                            halo_mask = cv2.bitwise_and(
                                halo_mask, cv2.bitwise_not(base_mask)
                            )
                            halo_intensity *= (base_mask == 0)

            shelf_mask = None
            shelf_intensity = None
            if SHELF_ENABLED and np.any(base_mask):
                bounds = _mask_bounds(base_mask)
                contour = _largest_contour(base_mask)
                if bounds is not None and contour is not None:
                    x_min, x_max, y_min, y_max = bounds
                    width = x_max - x_min + 1
                    height = y_max - y_min + 1
                    if width > 4 and height > 4 and contour.shape[0] > 12:
                        shelf_mask = np.zeros_like(base_mask)
                        shelf_intensity = np.zeros_like(gray, dtype=np.float32)
                        max_height = max(1.0, min(width, height) * SHELF_MAX_HEIGHT_RATIO)
                        base_offset = SHELF_OUTSET + max_height * SHELF_BASE_OFFSET_RATIO
                        amplitude = SHELF_BASE + kick_level * SHELF_GAIN
                        phase_offset = math.pi / max(1, SHELF_BANDS)

                        points = _smooth_closed_curve(contour, SHELF_SMOOTH_WINDOW)
                        count = points.shape[0]
                        normals = np.zeros_like(points)
                        for idx in range(count):
                            prev_point = points[(idx - 1) % count]
                            next_point = points[(idx + 1) % count]
                            tangent = next_point - prev_point
                            length = math.hypot(float(tangent[0]), float(tangent[1]))
                            if length < 1e-3:
                                continue
                            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32) / length
                            sample = points[idx] + normal * 2.0
                            sx = int(np.clip(sample[0], 0, gray.shape[1] - 1))
                            sy = int(np.clip(sample[1], 0, gray.shape[0] - 1))
                            if base_mask[sy, sx] > 0:
                                normal = -normal
                            normals[idx] = normal

                        if SHELF_MODE == "random":
                            if frame_time - last_shelf_change >= SHELF_RANDOM_HOLD:
                                shelf_center_index = int(rng.integers(0, count))
                                last_shelf_change = frame_time
                            band_centers = [shelf_center_index]
                        else:
                            phase = (frame_time * SHELF_SWEEP_SPEED) % 1.0
                            band_centers = [
                                int((phase + band_idx / SHELF_BANDS) * count) % count
                                for band_idx in range(SHELF_BANDS)
                            ]

                        band_length = max(6, int(count * SHELF_ARC_RATIO))
                        half_band = band_length // 2
                        for band_index, center_index in enumerate(band_centers):
                            offset_points: list[Tuple[int, int]] = []
                            offset_strengths: list[float] = []
                            for offset in range(-half_band, half_band + 1, SHELF_SAMPLE_STEP):
                                idx = (center_index + offset) % count
                                point = points[idx]
                                normal = normals[idx]
                                if normal[0] == 0 and normal[1] == 0:
                                    continue
                                t = (offset + half_band) / max(1, band_length)
                                parabola = max(0.0, 1.0 - (2.0 * t - 1.0) ** 2)
                                wave = math.sin(
                                    frame_time * SHELF_SPEED
                                    + t * math.pi * SHELF_WAVE_FREQ
                                    + band_index * phase_offset
                                )
                                magnitude = abs(wave)
                                spike = (
                                    (parabola**SHELF_PARABOLA_POWER)
                                    * magnitude
                                    * amplitude
                                    * max_height
                                    * SHELF_SPIKE_RATIO
                                )
                                offset_px = base_offset + spike
                                if offset_px < SHELF_MIN_HEIGHT:
                                    continue
                                offset_point = point + normal * offset_px
                                x = int(np.clip(offset_point[0], 0, gray.shape[1] - 1))
                                y = int(np.clip(offset_point[1], 0, gray.shape[0] - 1))
                                offset_points.append((x, y))
                                offset_strengths.append(parabola * magnitude * amplitude)

                            for idx in range(1, len(offset_points)):
                                p0 = offset_points[idx - 1]
                                p1 = offset_points[idx]
                                strength = offset_strengths[idx]
                                cv2.line(shelf_mask, p0, p1, 255, SHELF_THICKNESS)
                                cv2.line(
                                    shelf_intensity,
                                    p0,
                                    p1,
                                    float(strength),
                                    SHELF_THICKNESS,
                                )

                        if SHELF_OUTSIDE_ONLY:
                            shelf_mask = cv2.bitwise_and(
                                shelf_mask, cv2.bitwise_not(base_mask)
                            )
                            shelf_intensity *= (base_mask == 0)

            spike_mask = None
            if SPIKES_ENABLED and kick_level >= SPIKE_MIN_KICK and np.any(mask):
                edges = cv2.morphologyEx(base_mask, cv2.MORPH_GRADIENT, kernel)
                radius = max(1, int((kick_level**SPIKE_GAMMA) * SPIKE_MAX_RADIUS))
                if radius > 0:
                    spike_kernel = spike_kernels.get(radius)
                    if spike_kernel is None:
                        size = radius * 2 + 1
                        spike_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                        spike_kernels[radius] = spike_kernel
                    spike_layer = cv2.dilate(edges, spike_kernel, iterations=1)
                    density = min(1.0, SPIKE_DENSITY_BASE + kick_level * SPIKE_DENSITY_GAIN)
                    noise = (rng.random(spike_layer.shape) < density).astype(np.uint8) * 255
                    spike_mask = cv2.bitwise_and(spike_layer, noise)
                    if SPIKE_OUTSIDE_ONLY:
                        spike_mask = cv2.bitwise_and(spike_mask, cv2.bitwise_not(base_mask))

            mask_out = base_mask
            boost_mask = spike_mask
            if spike_mask is not None:
                mask_out = cv2.bitwise_or(mask_out, spike_mask)
            if shelf_mask is not None:
                if boost_mask is None:
                    boost_mask = shelf_mask
                else:
                    boost_mask = cv2.bitwise_or(boost_mask, shelf_mask)
                mask_out = cv2.bitwise_or(mask_out, shelf_mask)
            if halo_mask is not None:
                if boost_mask is None:
                    boost_mask = halo_mask
                else:
                    boost_mask = cv2.bitwise_or(boost_mask, halo_mask)
                mask_out = cv2.bitwise_or(mask_out, halo_mask)

            extra_intensity = ripple_intensity
            if shelf_intensity is not None:
                if extra_intensity is None:
                    extra_intensity = shelf_intensity
                else:
                    extra_intensity = np.clip(
                        extra_intensity + shelf_intensity, 0.0, 1.0
                    )
            if halo_intensity is not None:
                if extra_intensity is None:
                    extra_intensity = halo_intensity
                else:
                    extra_intensity = np.clip(
                        extra_intensity + halo_intensity, 0.0, 1.0
                    )

            if ripple_intensity is not None:
                ripple_mask = (ripple_intensity > RIPPLE_MASK_THRESHOLD).astype(np.uint8) * 255
                if RIPPLE_INSIDE_ONLY:
                    ripple_mask = cv2.bitwise_and(ripple_mask, base_mask)
                else:
                    ripple_mask = cv2.bitwise_and(ripple_mask, cv2.bitwise_not(base_mask))
                if boost_mask is None:
                    boost_mask = ripple_mask
                else:
                    boost_mask = cv2.bitwise_or(boost_mask, ripple_mask)
                mask_out = cv2.bitwise_or(mask_out, ripple_mask)

            highlight_mask = None
            if BRAILLE_WHITE_SPIKES and RENDER_MODE == "braille":
                if halo_mask is not None and spike_mask is not None:
                    highlight_mask = cv2.bitwise_or(halo_mask, spike_mask)
                elif halo_mask is not None:
                    highlight_mask = halo_mask
                elif spike_mask is not None:
                    highlight_mask = spike_mask

            braille_spike_mask = spike_mask
            if RENDER_MODE == "braille" and HALO_SOLID_FILL and halo_mask is not None:
                if braille_spike_mask is None:
                    braille_spike_mask = halo_mask
                else:
                    braille_spike_mask = cv2.bitwise_or(braille_spike_mask, halo_mask)

            if RENDER_MODE == "braille":
                braille_lines, highlight_cells = _frame_to_braille(
                    gray,
                    mask_out,
                    kick_level,
                    braille_spike_mask,
                    extra_intensity,
                    braille_dither_cache,
                    highlight_mask,
                )
                has_highlight = (
                    highlight_cells is not None and np.any(highlight_cells)
                )
                if BRAILLE_WHITE_SPIKES and has_highlight:
                    ascii_frame = _colorize_braille(
                        braille_lines,
                        highlight_cells,
                        color,
                        WHITE_COLOR,
                    )
                    frame_prefix = ""
                    use_color = bool(color) or has_highlight
                else:
                    ascii_frame = "\n".join(braille_lines)
                    frame_prefix = color
                    use_color = bool(color)
            else:
                ascii_frame = _frame_to_ascii(
                    gray, mask_out, lookup, SOLID_SILHOUETTE, boost_mask
                )
                frame_prefix = color
                use_color = bool(color)

            dt = frame_time - last_time
            last_time = frame_time
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            status = _format_status(fps, kick_level, audio_enabled)
            reset = "\x1b[0m" if use_color else ""
            sys.stdout.write("\x1b[H" + frame_prefix + ascii_frame + "\n" + reset + status)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if audio_enabled:
            kick_meter.stop()
        if selfie is not None:
            selfie.close()
        if segmenter is not None:
            segmenter.close()
        cap.release()
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()
