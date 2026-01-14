# Hand Detection Integration Plan

This document captures requirements and phased steps to add hand/finger detection
so gestures can drive particle motion. The goal is a reliable, debuggable workflow
that can be tuned for your lighting/camera setup.

## Requirements

### Detection approach
- Use MediaPipe Hands (hand landmark model) for reliable finger tracking.
- If using segmentation, include a calibration step to isolate hands cleanly.
- Support at least one finger per hand as the primary control signal.

### Visual control
- Particles should remain static until fingers are detected.
- Once fingers are detected, particles move in response to finger motion/position.
- Gestures should be robust to partial occlusion and rapid movement.

### Debugging
- Render a clear, orange diagnostic overlay for detected hands/fingers.
- Overlay should be separate from the existing silhouette/halo effects.
- Toggle-able debug mode to keep the visualizer clean during performances.

## Phased Implementation

### Phase 1: Model selection + baseline detection
- Integrate MediaPipe Hands for finger tracking (21 landmarks per hand).
- Add a detection module that outputs:
  - hand presence
  - key finger tip positions (index tip, optionally thumb tip)
- Provide CPU-only fallback behavior if the model is unavailable.

### Phase 2: Calibration workflow (if segmentation is used)
- Add a short calibration sequence:
  - prompt user to place hand in view
  - compute threshold parameters (mask confidence, morphology settings)
- Store calibration in `config.local.json`.
- Provide a `--hand-debug` mode to visualize detection results.

### Phase 3: Debug overlay
- Draw orange overlays for detected fingertips and/or hand contour.
- Keep overlay on a dedicated mask so it can be blended cleanly.
- Ensure overlay does not break existing halo/ripple drawing.

### Phase 4: Gesture → particle control
- Introduce a “gesture state” object with:
  - finger count
  - finger positions (normalized or pixel)
  - velocity/smoothing
- When no fingers detected, hold particles in place.
- When fingers detected:
  - move particles toward finger positions
  - optionally add force fields or attraction/repulsion

### Phase 5: Polish + performance
- Add smoothing (EMA) for finger positions to reduce jitter.
- Clamp to safe bounds and handle quick transitions.
- Measure FPS impact and optimize if needed.

## Open Questions
- Do you want left/right hands to control different particle groups?
- Should a single finger control one “leader” particle or a field force?
- What is the ideal debug toggle (CLI flag vs. config setting)?
