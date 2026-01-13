# asciivisualizerpy

Terminal braille visualizer with a camera silhouette, kick-reactive halo spikes, halo particles, and ripple rings.

## Requirements

- Python 3.10+
- A camera
- A microphone (optional if you run with `--no-audio`)
- A terminal that supports ANSI color

## Install

Create a virtual environment and install Python dependencies:

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows (cmd):

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### System dependencies

`sounddevice` needs PortAudio:

- macOS: `brew install portaudio`
- Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
- Fedora: `sudo dnf install portaudio-devel`
- Windows: PortAudio is usually bundled with `sounddevice` wheels; if you see audio device errors, install PortAudio via a system package manager or use a Python distribution that bundles it (for example, conda).

OpenCV on Linux may require extra system libraries. If import errors mention `libGL` or `glib`, install:

```bash
sudo apt-get install libgl1 libglib2.0-0
```

## Run

Optional: select camera/mic and write `config.local.json`:

```bash
python scripts/select_devices.py
```

Optional: download the MediaPipe segmentation model for the tasks backend:

```bash
python scripts/download_segmentation_model.py
```

Start the visualizer:

```bash
python -m asciivisualizerpy
```

Useful flags:

- `--camera N` to pick a camera index
- `--mic N` to pick a microphone index
- `--no-audio` to disable audio input
- `--segmentation mog2` if you do not have MediaPipe installed

## Tuning constants

All effect and rendering knobs live near the top of `src/asciivisualizerpy/visualizer.py`.
Edit them, save, and re-run.

Color:
- `COLOR_ENABLED`, `COLOR_PALETTE`, `COLOR_KICK_THRESHOLD`, `COLOR_MIN_INTERVAL`

Halo spikes:
- `HALO_ENABLED`, `HALO_BANDS`, `HALO_LOW_HZ`, `HALO_HIGH_HZ`
- `HALO_MAX_HEIGHT_RATIO`, `HALO_BASE_OFFSET_RATIO`, `HALO_OUTSET`, `HALO_THICKNESS`
- `HALO_GAMMA`, `HALO_KICK_GAIN`, `HALO_OUTSIDE_ONLY`, `HALO_INTERP_BANDS`
- `HALO_SOLID_FILL`, `HALO_TIP_OUTLINE`
- `HALO_CIRCLE_TOP_START_ANGLE`, `HALO_CIRCLE_TOP_END_ANGLE`
- `HALO_CIRCLE_BOTTOM_START_ANGLE`, `HALO_CIRCLE_BOTTOM_END_ANGLE`
- `HALO_CIRCLE_TOP_SPIKES`, `HALO_CIRCLE_BOTTOM_SPIKES`
- `HALO_CIRCLE_BASE_WIDTH_RATIO`, `HALO_CIRCLE_BASE_WIDTH_MIN`
- `HALO_CIRCLE_CURVE_SAMPLES`, `HALO_CIRCLE_PARABOLA`
- `HALO_CIRCLE_CENTER_X_OFFSET`, `HALO_CIRCLE_CENTER_Y_OFFSET`
- `HALO_CIRCLE_RADIUS_SCALE`

Halo particles:
- `HALO_PARTICLES_ENABLED`, `HALO_PARTICLE_MIN_KICK`, `HALO_PARTICLE_MAX_RADIUS`
- `HALO_PARTICLE_GAMMA`, `HALO_PARTICLE_DENSITY_BASE`, `HALO_PARTICLE_DENSITY_GAIN`
- `HALO_PARTICLE_OUTSIDE_ONLY`

Ripples:
- `RIPPLES_ENABLED`, `RIPPLE_KICK_THRESHOLD`, `RIPPLE_MIN_INTERVAL`
- `RIPPLE_SPEED_RATIO`, `RIPPLE_WIDTH_RATIO`, `RIPPLE_DECAY`, `RIPPLE_BOOST`
- `RIPPLE_MASK_THRESHOLD`, `RIPPLE_MAX_RINGS`, `RIPPLE_MAX_RADIUS_RATIO`
- `RIPPLE_INSIDE_ONLY`, `RIPPLE_ORIGIN`

Segmentation:
- `SEGMENTATION_THRESHOLD`, `MOG2_LEARNING_RATE`

Braille grid:
- `BRAILLE_CELL`, `BRAILLE_DITHER`
