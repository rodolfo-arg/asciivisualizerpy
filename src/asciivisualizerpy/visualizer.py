from __future__ import annotations

import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .audio import KickMeter
from .midi import MidiInput

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None  # type: ignore[assignment]

CHAR_ASPECT = 0.5
FG_THRESHOLD = 200
SEGMENTATION_THRESHOLD = 0.35
MOG2_LEARNING_RATE = 0.002
COLOR_ENABLED = True
BACKGROUND_ENABLED = True
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
BACKGROUND_PALETTE = [(0, 0, 0), *COLOR_PALETTE]
COLOR_KICK_THRESHOLD = 0.25
COLOR_MIN_INTERVAL = 0.12
HALO_HIGHLIGHT_COLOR = "\x1b[38;2;255;255;255m"
BLACK_FG = "\x1b[38;2;0;0;0m"
BRAILLE_CELL = (4, 2)
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
HALO_ENABLED = True
HALO_BANDS = 12
HALO_LOW_HZ = 80.0
HALO_HIGH_HZ = 8000.0
HALO_MAX_HEIGHT_RATIO = 0.45
HALO_BASE_OFFSET_RATIO = 0.12
HALO_OUTSET = 2
HALO_THICKNESS = 1
HALO_GAMMA = 1.6
HALO_KICK_GAIN = 0.35
HALO_OUTSIDE_ONLY = True
HALO_INTERP_BANDS = False
HALO_SOLID_FILL = True
HALO_TIP_OUTLINE = True
HALO_CIRCLE_TOP_START_ANGLE = math.pi * 0.85
HALO_CIRCLE_TOP_END_ANGLE = 0.0
HALO_CIRCLE_BOTTOM_START_ANGLE = 0.0
HALO_CIRCLE_BOTTOM_END_ANGLE = math.tau - HALO_CIRCLE_TOP_START_ANGLE
HALO_CIRCLE_TOP_SPIKES = 1
HALO_CIRCLE_BOTTOM_SPIKES = 6
HALO_CIRCLE_BASE_WIDTH_RATIO = 0.16
HALO_CIRCLE_BASE_WIDTH_MIN = 6
HALO_CIRCLE_CURVE_SAMPLES = 11
HALO_CIRCLE_PARABOLA = 1.0
HALO_CIRCLE_CENTER_X_OFFSET = 0.0
HALO_CIRCLE_CENTER_Y_OFFSET = 0.5
HALO_CIRCLE_RADIUS_SCALE = 0.7
HALO_PARTICLES_ENABLED = True
HALO_PARTICLE_MIN_KICK = 0.05
HALO_PARTICLE_MAX_RADIUS = 6
HALO_PARTICLE_GAMMA = 1.4
HALO_PARTICLE_DENSITY_BASE = 0.08
HALO_PARTICLE_DENSITY_GAIN = 0.45
HALO_PARTICLE_OUTSIDE_ONLY = True
DEFAULT_SEGMENTATION_MODEL = (
    Path(__file__).resolve().parents[2] / "assets" / "models" / "selfie_segmenter.tflite"
)

MIDI_CC_TOGGLE_HALO_PARTICLES = 24
MIDI_CC_TOGGLE_CUBE = 25
MIDI_CC_MOMENTARY_BG_COLOR = 26
MIDI_CC_MOMENTARY_UNUSED_27 = 27
MIDI_CC_TOGGLE_PARABOLA = 28
MIDI_CC_TOGGLE_MESH = 29
MIDI_CC_MOMENTARY_HEART = 30
MIDI_CC_MOMENTARY_UNUSED_31 = 31

CUBE_ENABLED_DEFAULT = False
CUBE_SIZE_RATIO = 0.22
CUBE_DEPTH_RATIO = 0.35
CUBE_SPEED_RATIO = 0.14
CUBE_LINE_THICKNESS = 1
CUBE_ROTATION_SPEED = 0.35
CUBE_TILT = 0.45

MESH_ENABLED_DEFAULT = False
MESH_PARTICLE_COUNT = 60
MESH_SPEED_RATIO = 0.12
MESH_POINT_RADIUS = 1
MESH_LINK_DISTANCE_RATIO = 0.25
MESH_LINE_THICKNESS = 1

HEART_SPEED_RATIO = 0.45
HEART_SIZE_RATIO = 0.06
HEART_LIFETIME = 3.2
HEART_MAX_ACTIVE = 12

HAND_ENABLED_DEFAULT = True
HAND_MAX_NUM_HANDS = 2
HAND_MODEL_COMPLEXITY = 1
HAND_MIN_DETECTION_CONFIDENCE = 0.4
HAND_MIN_TRACKING_CONFIDENCE = 0.4
HAND_SMOOTHING = 0.35
HAND_RESET_SECONDS = 0.6
HAND_INDEX_TIP = 8
HAND_BACKEND_DEFAULT = "auto"


@dataclass
class CubeState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    spin: float


@dataclass
class MeshState:
    positions: np.ndarray
    velocities: np.ndarray


@dataclass
class Heart:
    x: float
    y: float
    vx: float
    vy: float
    spawn_time: float
    size: int


@dataclass
class HandState:
    positions: np.ndarray
    velocities: np.ndarray
    valid: np.ndarray
    visible: np.ndarray
    last_seen: np.ndarray




def _color_for_rgb(color: Tuple[int, int, int]) -> str:
    if not COLOR_ENABLED:
        return ""
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m"


def _background_for_rgb(color: Tuple[int, int, int]) -> str:
    if not BACKGROUND_ENABLED:
        return ""
    r, g, b = color
    return f"\x1b[48;2;{r};{g};{b}m"


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


def _mask_centroid(mask: np.ndarray, fallback: Tuple[float, float]) -> Tuple[float, float]:
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


def _heart_polygon(size: int, cache: Dict[int, np.ndarray]) -> np.ndarray:
    cached = cache.get(size)
    if cached is not None:
        return cached
    t = np.linspace(0.0, math.tau, 40, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x = x / 18.0
    y = -y / 18.0
    points = np.stack([x, y], axis=1) * float(size)
    polygon = points.astype(np.int32)
    cache[size] = polygon
    return polygon


def _draw_heart(
    mask: np.ndarray,
    center: Tuple[float, float],
    size: int,
    cache: Dict[int, np.ndarray],
) -> None:
    if size <= 0:
        return
    polygon = _heart_polygon(size, cache)
    cx, cy = center
    translated = polygon + np.array([int(round(cx)), int(round(cy))], dtype=np.int32)
    cv2.fillPoly(mask, [translated], 255)


def _init_cube_state(rng: np.random.Generator, width: int, height: int, speed: float) -> CubeState:
    angle = rng.random() * math.tau
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    spin = (rng.random() * 0.5 + 0.5) * CUBE_ROTATION_SPEED
    if rng.random() < 0.5:
        spin = -spin
    return CubeState(x=width / 2.0, y=height / 2.0, vx=vx, vy=vy, angle=0.0, spin=spin)


def _update_cube(
    state: CubeState,
    dt: float,
    bounds: Tuple[int, int],
    size: float,
    depth: float,
) -> None:
    width, height = bounds
    if dt <= 0:
        return
    state.x += state.vx * dt
    state.y += state.vy * dt
    state.angle = (state.angle + state.spin * dt) % math.tau

    half = size / 2.0
    left = state.x - half
    right = state.x + half + depth
    top = state.y - half - depth
    bottom = state.y + half

    if left < 0:
        state.x += -left
        state.vx = abs(state.vx)
    if right > width - 1:
        state.x -= right - (width - 1)
        state.vx = -abs(state.vx)
    if top < 0:
        state.y += -top
        state.vy = abs(state.vy)
    if bottom > height - 1:
        state.y -= bottom - (height - 1)
        state.vy = -abs(state.vy)


def _draw_cube(
    mask: np.ndarray,
    state: CubeState,
    size: float,
    depth: float,
    thickness: int,
) -> None:
    half = size / 2.0
    cx, cy = state.x, state.y
    vertices = np.array(
        [
            (-half, -half, -half),
            (half, -half, -half),
            (half, half, -half),
            (-half, half, -half),
            (-half, -half, half),
            (half, -half, half),
            (half, half, half),
            (-half, half, half),
        ],
        dtype=np.float32,
    )

    cos_y = math.cos(state.angle)
    sin_y = math.sin(state.angle)
    cos_x = math.cos(CUBE_TILT)
    sin_x = math.sin(CUBE_TILT)
    projected = []
    camera_dist = max(1.0, size * 2.0 + depth * 2.0)
    for x, y, z in vertices:
        xz = x * cos_y + z * sin_y
        zz = -x * sin_y + z * cos_y
        yz = y * cos_x - zz * sin_x
        zz = y * sin_x + zz * cos_x
        factor = camera_dist / (camera_dist + zz)
        sx = cx + xz * factor
        sy = cy + yz * factor
        projected.append((int(round(sx)), int(round(sy))))

    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for a, b in edges:
        cv2.line(mask, projected[a], projected[b], 255, thickness)


def _init_mesh_state(
    rng: np.random.Generator,
    width: int,
    height: int,
    count: int,
    speed: float,
) -> MeshState:
    positions = np.column_stack(
        [
            rng.uniform(0, max(width - 1, 1), size=count),
            rng.uniform(0, max(height - 1, 1), size=count),
        ]
    ).astype(np.float32)
    angles = rng.uniform(0.0, math.tau, size=count)
    velocities = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    velocities *= float(speed)
    return MeshState(positions=positions, velocities=velocities)


def _update_mesh(
    state: MeshState,
    dt: float,
    bounds: Tuple[int, int],
) -> None:
    if dt <= 0:
        return
    width, height = bounds
    state.positions += state.velocities * float(dt)
    for idx in range(state.positions.shape[0]):
        x, y = state.positions[idx]
        dx, dy = state.velocities[idx]
        if x < 0:
            x = -x
            dx = abs(dx)
        elif x > width - 1:
            x = 2 * (width - 1) - x
            dx = -abs(dx)
        if y < 0:
            y = -y
            dy = abs(dy)
        elif y > height - 1:
            y = 2 * (height - 1) - y
            dy = -abs(dy)
        state.positions[idx] = (x, y)
        state.velocities[idx] = (dx, dy)

def _draw_mesh(
    mask: np.ndarray,
    state: MeshState,
    point_radius: int,
    max_link_distance: float,
    line_thickness: int,
) -> None:
    positions = state.positions
    count = positions.shape[0]
    max_dist_sq = max_link_distance * max_link_distance
    for i in range(count):
        x1, y1 = positions[i]
        for j in range(i + 1, count):
            x2, y2 = positions[j]
            dx = x1 - x2
            dy = y1 - y2
            if (dx * dx + dy * dy) <= max_dist_sq:
                cv2.line(
                    mask,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    255,
                    line_thickness,
                )
        cv2.circle(mask, (int(round(x1)), int(round(y1))), point_radius, 255, -1)


def _init_hand_state(max_hands: int) -> HandState:
    positions = np.zeros((max_hands, 2), dtype=np.float32)
    velocities = np.zeros_like(positions)
    valid = np.zeros(max_hands, dtype=bool)
    visible = np.zeros(max_hands, dtype=bool)
    last_seen = np.full(max_hands, -1.0, dtype=np.float32)
    return HandState(
        positions=positions,
        velocities=velocities,
        valid=valid,
        visible=visible,
        last_seen=last_seen,
    )


def _update_hand_state(
    state: HandState,
    raw_positions: list[Tuple[float, float]],
    dt: float,
    now: float,
    smoothing: float,
    reset_seconds: float,
) -> None:
    max_hands = state.positions.shape[0]
    state.visible[:] = False
    count = min(len(raw_positions), max_hands)
    for idx in range(count):
        state.visible[idx] = True
        state.last_seen[idx] = float(now)
        pos = np.array(raw_positions[idx], dtype=np.float32)
        if not state.valid[idx]:
            state.positions[idx] = pos
            state.velocities[idx] = 0.0
            state.valid[idx] = True
            continue
        prev = state.positions[idx].copy()
        smoothed = prev * (1.0 - smoothing) + pos * smoothing
        state.positions[idx] = smoothed
        if dt > 0:
            state.velocities[idx] = (smoothed - prev) / dt
        else:
            state.velocities[idx] = 0.0

    for idx in range(count, max_hands):
        state.visible[idx] = False
        state.velocities[idx] = 0.0
        if state.valid[idx] and state.last_seen[idx] >= 0:
            if (now - state.last_seen[idx]) > reset_seconds:
                state.valid[idx] = False


def _hand_landmark_to_pixel(
    landmark: object,
    width: int,
    height: int,
) -> Tuple[float, float]:
    x = float(np.clip(getattr(landmark, "x", 0.0), 0.0, 1.0)) * (width - 1)
    y = float(np.clip(getattr(landmark, "y", 0.0), 0.0, 1.0)) * (height - 1)
    return x, y


def _extract_hand_positions(
    results: object,
    width: int,
    height: int,
) -> list[Tuple[float, float]]:
    finger_positions: list[Tuple[float, float]] = []
    landmarks = getattr(results, "multi_hand_landmarks", None)
    if not landmarks:
        return finger_positions
    for hand_landmarks in landmarks:
        if len(hand_landmarks.landmark) > HAND_INDEX_TIP:
            finger_positions.append(
                _hand_landmark_to_pixel(
                    hand_landmarks.landmark[HAND_INDEX_TIP],
                    width,
                    height,
                )
            )
    return finger_positions


def _frame_to_braille(
    gray: np.ndarray,
    mask: np.ndarray,
    boost_mask: np.ndarray | None,
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
    if boost_mask is not None:
        boost_mask = boost_mask[:height, :width]

    ink = (255 - gray).astype(np.float32) / 255.0
    if ripple_intensity is not None:
        ripple_intensity = ripple_intensity[:height, :width]
        ink = np.clip(ink + ripple_intensity * RIPPLE_BOOST, 0.0, 1.0)
    if boost_mask is not None:
        ink[boost_mask > 0] = 1.0

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
    reset = "\x1b[0m"
    default_color = base_color if base_color else reset
    rendered = []
    for row_index, line in enumerate(lines):
        row_highlight = highlight_cells[row_index] if highlight_cells is not None else None
        out = []
        current_color = ""
        if base_color:
            out.append(base_color)
            current_color = base_color
        for col_index, ch in enumerate(line):
            want_highlight = bool(row_highlight[col_index]) if row_highlight is not None else False
            if want_highlight:
                desired_color = highlight_color
            else:
                desired_color = default_color
            if desired_color != current_color:
                out.append(desired_color)
                current_color = desired_color
            out.append(ch)
        if base_color:
            if current_color != base_color:
                out.append(base_color)
        elif current_color:
            out.append(reset)
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
    midi_enabled: bool = True,
    midi_input_port: str | None = None,
    hand_enabled: bool = HAND_ENABLED_DEFAULT,
    hand_backend: str = HAND_BACKEND_DEFAULT,
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

    hand_backend_mode = hand_backend.lower() if isinstance(hand_backend, str) else "auto"
    if hand_backend_mode not in {"auto", "solutions"}:
        raise ValueError("hand_backend must be 'auto' or 'solutions'")

    hands = None
    hand_backend_active = "none"
    if hand_enabled:
        if mp is None:
            print("Hand detection disabled: mediapipe is not installed.", file=sys.stderr)
        else:
            if hand_backend_mode in {"auto", "solutions"}:
                try:
                    hands = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=HAND_MAX_NUM_HANDS,
                        model_complexity=HAND_MODEL_COMPLEXITY,
                        min_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
                        min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
                    )
                    hand_backend_active = "solutions"
                except Exception as exc:  # pragma: no cover - optional dependency
                    print(f"Hand detection disabled: {exc}", file=sys.stderr)
                    hands = None

            if hand_backend_active == "none":
                print("Hand detection disabled: no backend available.", file=sys.stderr)

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

    midi_input = MidiInput(midi_input_port)
    if midi_enabled:
        midi_input.open()

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Unable to read from camera")

    term_size = shutil.get_terminal_size((80, 24))
    out_w, out_h = _compute_output_size(
        (frame.shape[1], frame.shape[0]),
        (term_size.columns, term_size.lines),
    )

    braille_dither_cache: Dict[Tuple[int, int], np.ndarray] = {}
    grid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    halo_particle_kernels: Dict[int, np.ndarray] = {}
    rng = np.random.default_rng()
    ripples: list[Tuple[float, float, float, float]] = []
    last_ripple_time = -1.0
    last_kick_level = 0.0
    color_index = 0
    background_index = 0
    last_color_time = -1.0
    last_color_kick = 0.0
    last_time = time.monotonic()
    fps = 0.0
    midi_cc_values: Dict[int, int] = {}
    halo_particles_enabled = HALO_PARTICLES_ENABLED
    halo_parabola_enabled = True
    cube_enabled = CUBE_ENABLED_DEFAULT
    mesh_enabled = MESH_ENABLED_DEFAULT
    cube_state: CubeState | None = None
    mesh_state: MeshState | None = None
    hearts: list[Heart] = []
    heart_cache: Dict[int, np.ndarray] = {}
    hand_state = _init_hand_state(HAND_MAX_NUM_HANDS) if hands is not None else None

    sys.stdout.write("\x1b[2J\x1b[H\x1b[?25l")
    sys.stdout.flush()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_time = time.monotonic()
            frame_dt = frame_time - last_time
            if frame_dt < 0:
                frame_dt = 0.0

            midi_background_bump = False
            heart_triggers = 0
            if midi_input.enabled:
                for message in midi_input.poll(frame_time):
                    if (
                        message.type != "control_change"
                        or message.control is None
                        or message.value is None
                    ):
                        continue
                    control = message.control
                    value = message.value
                    prev = midi_cc_values.get(control, 0)
                    midi_cc_values[control] = value
                    pressed = value > 0
                    was_pressed = prev > 0
                    if control == MIDI_CC_TOGGLE_HALO_PARTICLES:
                        halo_particles_enabled = pressed
                    elif control == MIDI_CC_TOGGLE_PARABOLA:
                        halo_parabola_enabled = pressed
                    elif control == MIDI_CC_TOGGLE_CUBE:
                        cube_enabled = pressed
                    elif control == MIDI_CC_TOGGLE_MESH:
                        mesh_enabled = pressed
                    elif control == MIDI_CC_MOMENTARY_BG_COLOR:
                        if pressed:
                            midi_background_bump = True
                    elif control == MIDI_CC_MOMENTARY_HEART:
                        if pressed:
                            heart_triggers += 1

            term_size = shutil.get_terminal_size((80, 24))
            new_w, new_h = _compute_output_size(
                (frame.shape[1], frame.shape[0]),
                (term_size.columns, term_size.lines),
            )
            if (new_w, new_h) != (out_w, out_h):
                out_w, out_h = new_w, new_h

            render_w = out_w * BRAILLE_CELL[1]
            render_h = out_h * BRAILLE_CELL[0]
            resized = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            frame_height, frame_width = gray.shape
            min_dim = min(frame_height, frame_width)
            if heart_triggers and min_dim > 0:
                size = max(3, int(min_dim * HEART_SIZE_RATIO))
                speed = HEART_SPEED_RATIO * min_dim
                cx = frame_width / 2.0
                cy = frame_height / 2.0
                for _ in range(heart_triggers):
                    if len(hearts) >= HEART_MAX_ACTIVE:
                        break
                    angle = rng.random() * math.tau
                    hearts.append(
                        Heart(
                            x=cx,
                            y=cy,
                            vx=math.cos(angle) * speed,
                            vy=math.sin(angle) * speed,
                            spawn_time=frame_time,
                            size=size,
                        )
                    )

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            if segmenter_backend == "solutions":
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

            raw_finger_positions: list[Tuple[float, float]] = []
            if hands is not None:
                hand_results = hands.process(rgb)
                raw_finger_positions = _extract_hand_positions(
                    hand_results,
                    frame_width,
                    frame_height,
                )

            finger_positions: list[Tuple[float, float]] = []
            if hand_state is not None:
                _update_hand_state(
                    hand_state,
                    raw_finger_positions,
                    frame_dt,
                    frame_time,
                    HAND_SMOOTHING,
                    HAND_RESET_SECONDS,
                )
                for idx in range(hand_state.positions.shape[0]):
                    if hand_state.visible[idx]:
                        finger_positions.append(
                            (
                                float(hand_state.positions[idx, 0]),
                                float(hand_state.positions[idx, 1]),
                            )
                        )

            kick_level = kick_meter.read_level() if audio_enabled else 0.0
            band_levels = (
                kick_meter.read_bands() if audio_enabled else np.empty(0, dtype=np.float32)
            )
            background_is_black = True
            if BACKGROUND_PALETTE:
                if midi_background_bump:
                    background_index = (background_index + 1) % len(BACKGROUND_PALETTE)
                background_rgb = BACKGROUND_PALETTE[background_index]
                background_is_black = background_rgb == (0, 0, 0)
                background_color = _background_for_rgb(background_rgb)
            else:
                background_color = ""

            highlight_color = HALO_HIGHLIGHT_COLOR if background_is_black else BLACK_FG

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
                        origin_x, origin_y = _mask_centroid(base_mask, fallback)
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
            if HALO_ENABLED and band_levels.size > 0 and np.any(base_mask):
                bounds = _mask_bounds(base_mask)
                contour = _largest_contour(base_mask)
                if bounds is not None and contour is not None:
                    x_min, x_max, y_min, y_max = bounds
                    width = x_max - x_min + 1
                    height = y_max - y_min + 1
                    if width > 4 and height > 4 and contour.shape[0] > 12:
                        halo_mask = np.zeros_like(base_mask)
                        max_height = max(1.0, min(width, height) * HALO_MAX_HEIGHT_RATIO)
                        base_offset = HALO_OUTSET + max_height * HALO_BASE_OFFSET_RATIO
                        band_count = band_levels.size

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

                        center = _mask_centroid(
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

                        parabola_exponent = (
                            HALO_CIRCLE_PARABOLA if halo_parabola_enabled else 0.0
                        )
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
                                base_radius = base_radius * HALO_CIRCLE_RADIUS_SCALE + base_offset

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
                                    shape = parabola**parabola_exponent
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
                                if HALO_SOLID_FILL:
                                    cv2.fillPoly(halo_mask, [tri], 255)
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

                        if HALO_OUTSIDE_ONLY:
                            halo_mask = cv2.bitwise_and(
                                halo_mask, cv2.bitwise_not(base_mask)
                            )

            halo_particle_mask = None
            if halo_particles_enabled and kick_level >= HALO_PARTICLE_MIN_KICK and np.any(base_mask):
                edges = cv2.morphologyEx(base_mask, cv2.MORPH_GRADIENT, kernel)
                radius = max(
                    1, int((kick_level**HALO_PARTICLE_GAMMA) * HALO_PARTICLE_MAX_RADIUS)
                )
                if radius > 0:
                    particle_kernel = halo_particle_kernels.get(radius)
                    if particle_kernel is None:
                        size = radius * 2 + 1
                        particle_kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (size, size)
                        )
                        halo_particle_kernels[radius] = particle_kernel
                    particle_layer = cv2.dilate(edges, particle_kernel, iterations=1)
                    density = min(
                        1.0,
                        HALO_PARTICLE_DENSITY_BASE
                        + kick_level * HALO_PARTICLE_DENSITY_GAIN,
                    )
                    noise = (rng.random(particle_layer.shape) < density).astype(np.uint8) * 255
                    halo_particle_mask = cv2.bitwise_and(particle_layer, noise)
                    if HALO_PARTICLE_OUTSIDE_ONLY:
                        halo_particle_mask = cv2.bitwise_and(
                            halo_particle_mask, cv2.bitwise_not(base_mask)
                        )

            bounds = (frame_width, frame_height)
            min_dim = max(1.0, float(min(bounds)))

            background_mask = None
            if cube_enabled or mesh_enabled or hearts:
                background_mask = np.zeros_like(base_mask)
                if cube_enabled:
                    size = CUBE_SIZE_RATIO * min_dim
                    depth = CUBE_DEPTH_RATIO * size
                    speed = CUBE_SPEED_RATIO * min_dim
                    if cube_state is None:
                        cube_state = _init_cube_state(rng, frame_width, frame_height, speed)
                    _update_cube(cube_state, frame_dt, bounds, size, depth)
                    _draw_cube(background_mask, cube_state, size, depth, CUBE_LINE_THICKNESS)
                if mesh_enabled:
                    mesh_speed = MESH_SPEED_RATIO * min_dim
                    if mesh_state is None:
                        mesh_state = _init_mesh_state(
                            rng,
                            frame_width,
                            frame_height,
                            MESH_PARTICLE_COUNT,
                            mesh_speed,
                        )
                    _update_mesh(mesh_state, frame_dt, bounds)
                    _draw_mesh(
                        background_mask,
                        mesh_state,
                        MESH_POINT_RADIUS,
                        MESH_LINK_DISTANCE_RATIO * min_dim,
                        MESH_LINE_THICKNESS,
                    )
                if hearts:
                    active_hearts: list[Heart] = []
                    for heart in hearts:
                        age = frame_time - heart.spawn_time
                        if age > HEART_LIFETIME:
                            continue
                        heart.x += heart.vx * frame_dt
                        heart.y += heart.vy * frame_dt
                        if (
                            heart.x < -heart.size
                            or heart.x > frame_width + heart.size
                            or heart.y < -heart.size
                            or heart.y > frame_height + heart.size
                        ):
                            continue
                        _draw_heart(
                            background_mask,
                            (heart.x, heart.y),
                            heart.size,
                            heart_cache,
                        )
                        active_hearts.append(heart)
                    hearts = active_hearts

                if background_mask is not None:
                    background_mask = cv2.bitwise_and(
                        background_mask, cv2.bitwise_not(base_mask)
                    )

            mask_out = base_mask
            boost_mask = None
            if halo_mask is not None:
                mask_out = cv2.bitwise_or(mask_out, halo_mask)
                boost_mask = halo_mask
            if halo_particle_mask is not None:
                mask_out = cv2.bitwise_or(mask_out, halo_particle_mask)
                if boost_mask is None:
                    boost_mask = halo_particle_mask
                else:
                    boost_mask = cv2.bitwise_or(boost_mask, halo_particle_mask)
            if ripple_intensity is not None:
                ripple_mask = (ripple_intensity > RIPPLE_MASK_THRESHOLD).astype(np.uint8) * 255
                if RIPPLE_INSIDE_ONLY:
                    ripple_mask = cv2.bitwise_and(ripple_mask, base_mask)
                else:
                    ripple_mask = cv2.bitwise_and(ripple_mask, cv2.bitwise_not(base_mask))
                mask_out = cv2.bitwise_or(mask_out, ripple_mask)

            if background_mask is not None:
                mask_out = cv2.bitwise_or(mask_out, background_mask)
                if boost_mask is None:
                    boost_mask = background_mask
                else:
                    boost_mask = cv2.bitwise_or(boost_mask, background_mask)

            highlight_mask = halo_mask
            if halo_particle_mask is not None:
                if highlight_mask is None:
                    highlight_mask = halo_particle_mask
                else:
                    highlight_mask = cv2.bitwise_or(highlight_mask, halo_particle_mask)

            braille_lines, highlight_cells = _frame_to_braille(
                gray,
                mask_out,
                boost_mask,
                ripple_intensity,
                braille_dither_cache,
                highlight_mask,
            )
            has_highlight = highlight_cells is not None and np.any(highlight_cells)
            base_color = background_color + color
            if has_highlight:
                ascii_frame = _colorize_braille(
                    braille_lines,
                    highlight_cells,
                    base_color,
                    highlight_color,
                )
                frame_prefix = ""
                use_color = bool(base_color) or has_highlight
            else:
                ascii_frame = "\n".join(braille_lines)
                frame_prefix = base_color
                use_color = bool(base_color)

            last_time = frame_time
            if frame_dt > 0:
                fps = fps * 0.9 + (1.0 / frame_dt) * 0.1

            status = _format_status(fps, kick_level, audio_enabled)
            reset = "\x1b[0m" if use_color else ""
            sys.stdout.write("\x1b[H" + frame_prefix + ascii_frame + "\n" + reset + status)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if audio_enabled:
            kick_meter.stop()
        if midi_input.enabled:
            midi_input.close()
        if selfie is not None:
            selfie.close()
        if segmenter is not None:
            segmenter.close()
        if hands is not None:
            hands.close()
        cap.release()
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()
