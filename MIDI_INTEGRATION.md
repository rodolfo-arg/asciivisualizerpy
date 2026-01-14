# MIDI Integration Plan (Rodecaster Pro II)

This document outlines requirements and a phased plan to integrate MIDI controls
for ripple and color control in asciivisualizerpy. The approach is device-agnostic
so we can scale to other controllers and, eventually, a DJ deck.

## Goals
- Use Rodecaster Pro II controls to drive ripple triggers and color changes.
- Keep the MIDI layer device-agnostic so other MIDI devices can be added later.
- Support future multi-device setups (DJ controllers, fader banks, etc.).

## Requirements

### Device setup
- Enable MIDI control on the Rodecaster Pro II (Settings > System > MIDI, or via
  the Rodecaster App).
- Connect the Rodecaster Pro II over USB and confirm the OS sees a MIDI input
  named "MIDI Function".
- Decide which controls map to ripples vs. colors (Smart Pads, REC, Mute, etc.).

### MIDI capabilities to leverage
- Smart Pads can send Control Change (CC) or Note messages, with custom channel,
  number, and value.
- Pads can be configured to send On, Off, or both On+Off messages.
- Default Smart Pad mapping (if not customized) sends CC #0-#7 on bank 1, then
  CC #8-#15 on bank 2, etc., all on channel 1.
- The device can also receive CC messages for certain functions (pad bank select,
  pad trigger, pad color, record, channel/listen/mute buttons).

### Software requirements
- MIDI input backend (e.g., `mido` + `python-rtmidi`) for cross-platform support.
- A mapping layer and config file (`config.local.json`) for per-device bindings.
- Developer tooling to list MIDI devices and log raw events for debugging.

## Scalable architecture

### Core components
- `midi_backend.py`: port discovery, open input, hot-plug handling.
- `midi_events.py`: normalize to `MidiEvent` with
  `{device_id, type, channel, number, value, timestamp}`.
- `midi_mapping.py`: map events to `Action` objects (data-driven).
- `midi_state.py`: latching, momentary, smoothing, and bank tracking.
- `visualizer_controls.py`: merge audio-driven and MIDI-driven modulation.

### Device profiles
- Store profiles as JSON (one per device) in `assets/midi/`.
- Each profile defines: device matching, control mappings, and optional labels.
- Add optional "learn mode" for quick remapping.

## Phased integration plan

### Phase 0: Discovery and mapping design
- Verify MIDI can be enabled and detected on the host OS.
- Decide the control surface layout for ripples/colors.
- Draft the Rodecaster Pro II profile based on observed CC/Note messages.

### Phase 1: MIDI input prototype
- Add MIDI dependency and a script to list devices and log messages.
- Verify the device shows up as "MIDI Function".
- Capture messages from pads and buttons to confirm event types and values.

### Phase 2: Mapping layer + config
- Introduce `MidiEvent` and `MidiMapping` classes.
- Add `midi` settings to `config.local.json`:
  - `enabled`
  - `input_port`
  - `device_profile`
  - `mappings`
- Implement action types:
  - `trigger_ripple`
  - `set_ripple_intensity`
  - `cycle_palette`
  - `set_palette_index`
  - `set_color_shift`

### Phase 3: Visualizer integration
- Add a `MidiState` object to the visualizer update loop.
- Apply mapping outputs to ripple generation and color selection.
- Blend MIDI modulation with audio-driven effects (override or additive).

### Phase 4: Rodecaster Pro II profile + UX
- Provide a default profile mapping pads to ripple triggers and palette control.
- Support pad bank selection to choose palette banks or effect modes.
- Add a "learn" mode to simplify remapping for end users.

### Phase 5: Multi-device + DJ deck readiness
- Support multiple inputs with a priority or merge strategy.
- Add shift layers and banks to the mapping system.
- Create a capability model (buttons, faders, encoders, jog wheels).
- Add throttling/debouncing for high-frequency controls.

## Acceptance criteria
- Visualizer runs with no MIDI device connected (no crash; one warning at most).
- A pad press triggers a ripple within ~100 ms.
- Color changes can be triggered and persist until changed again.
- Hot-plugging the device recovers without restart.

## Open questions
- Should Smart Pads be momentary (press = ripple) or latching (toggle state)?
- Should MIDI input override audio-driven ripples or blend with them?
- Which controls should be dedicated to palette vs. ripple density/strength?

## Rodecaster Pro II MIDI reference (from Rode docs)
- Smart Pads can send custom CC or Note messages on any channel/value.
- Default pad send mapping (if not customized): CC #0-#7 on bank 1, CC #8-#15 on
  bank 2, etc., all on channel 1.
- MIDI device name exposed to the host: "MIDI Function".
- The device can receive CC messages for:
  - Pad Bank select: CC 0 (value 0-7) on channel 1.
  - Record button: CC 17 (value 1) on channel 1.
  - Channel button: CC 20 (value 1) on channel 1-6.
  - Listen button: CC 24 (value 1) on channel 1-6.
  - Mute button: CC 27 (value 1) on channel 1-6.
  - Pad trigger: CC 35 (value 1) on channel 1-8.
  - Pad color: CC 37 (value 1-127) on channel 1.
