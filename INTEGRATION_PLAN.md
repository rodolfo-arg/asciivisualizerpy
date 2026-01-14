# MIDI Control Integration Plan (Rodecaster Pro II)

This plan documents how the MIDI controls are integrated and which controls map to
which visual effects. It focuses on a scalable approach so additional controllers
can be added later without changing core rendering logic.

## Goals
- Read MIDI CC messages from the Rodecaster Pro II.
- Map specific CC numbers to effect toggles and triggers.
- Keep the visualizer stable if no MIDI device is present.

## Control Mapping

Latching (value > 0 = on, value = 0 = off):
- CC 24: Halo particles on/off
- CC 28: Parabolic spikes on/off
- CC 25: 3D cube background on/off (DVD bounce behavior)
- CC 29: Mesh particle background on/off

Momentary (press triggers action):
- CC 26: Cycle background canvas color
- CC 30: Shoot a heart in a random direction
- CC 27: Reserved (no action)
- CC 31: Reserved (no action)

## Phased Implementation

### Phase 1: MIDI input + event parsing
- Add `mido` + `python-rtmidi` dependencies.
- Open the selected MIDI input port and poll `iter_pending()` per frame.
- Normalize messages into a simple `MidiMessage` structure.

### Phase 2: Input mapping + state
- Track last CC values so momentary triggers only fire on press.
- Apply latching controls as direct on/off state (value > 0).
- Queue one-shot triggers (color bump, heart shot) for the current frame.

### Phase 3: Effect toggles + visuals
- Halo particles: gate `HALO_PARTICLES_ENABLED` with MIDI state.
- Parabolic spikes: toggle exponent between `HALO_CIRCLE_PARABOLA` and 0.
- 3D cube: draw a wireframe cube in the background with DVD-style bounce.
- Mesh particles: draw points + connections with simple bounce dynamics.
- Hearts: spawn directional hearts, update position and lifetime each frame.

### Phase 4: Scalability
- Keep the MIDI layer isolated (`midi.py`) so other devices can map into the same
  control structure.
- Reserve CC slots for future mappings (DJ deck controls, faders, jog wheels).

## Notes
- Effects render in the background (outside the silhouette mask).
- MIDI remains optional: the visualizer runs without a device connected.
