from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import mido
except ImportError:  # pragma: no cover - optional dependency
    mido = None  # type: ignore[assignment]


@dataclass
class MidiMessage:
    timestamp: float
    type: str
    channel: Optional[int]
    control: Optional[int]
    value: Optional[int]
    note: Optional[int]
    velocity: Optional[int]


def _to_message(now: float, message: object) -> MidiMessage:
    msg_type = getattr(message, "type", "unknown")
    channel = getattr(message, "channel", None)
    control = getattr(message, "control", None)
    value = getattr(message, "value", None)
    note = getattr(message, "note", None)
    velocity = getattr(message, "velocity", None)
    return MidiMessage(
        timestamp=now,
        type=msg_type,
        channel=channel if isinstance(channel, int) else None,
        control=control if isinstance(control, int) else None,
        value=value if isinstance(value, int) else None,
        note=note if isinstance(note, int) else None,
        velocity=velocity if isinstance(velocity, int) else None,
    )


class MidiInput:
    def __init__(self, port_name: Optional[str]) -> None:
        self.port_name = port_name
        self.enabled = False
        self.error: Optional[str] = None
        self.active_port_name: Optional[str] = None
        self._port = None

    def open(self) -> None:
        if mido is None:
            self.error = "mido not installed"
            return
        ports = mido.get_input_names()
        if not ports:
            self.error = "no MIDI inputs detected"
            return

        chosen = None
        if self.port_name:
            for name in ports:
                if name == self.port_name:
                    chosen = name
                    break
            if chosen is None:
                wanted = self.port_name.lower()
                for name in ports:
                    if wanted in name.lower():
                        chosen = name
                        break
            if chosen is None:
                self.error = f"input '{self.port_name}' not found"
                return
        else:
            chosen = ports[0]

        try:
            self._port = mido.open_input(chosen)
        except Exception as exc:  # pragma: no cover - runtime hardware dependency
            self.error = f"failed to open MIDI input: {exc}"
            return

        self.enabled = True
        self.active_port_name = chosen

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None
        self.enabled = False

    def poll(self, now: float) -> list[MidiMessage]:
        if not self.enabled or self._port is None:
            return []
        messages: list[MidiMessage] = []
        try:
            for message in self._port.iter_pending():
                messages.append(_to_message(now, message))
        except Exception as exc:  # pragma: no cover - runtime hardware dependency
            self.error = f"midi error: {exc}"
            self.close()
        return messages
