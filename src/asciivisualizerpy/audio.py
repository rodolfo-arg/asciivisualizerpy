from __future__ import annotations

import threading
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled by runtime checks
    sd = None  # type: ignore[assignment]


class KickMeter:
    def __init__(
        self,
        device: Optional[int] = None,
        samplerate: Optional[int] = None,
        blocksize: int = 1024,
        low_hz: float = 20.0,
        high_hz: float = 200.0,
        band_count: int = 24,
        band_low_hz: float = 80.0,
        band_high_hz: float = 8000.0,
    ) -> None:
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.band_count = max(0, int(band_count))
        self.band_low_hz = band_low_hz
        self.band_high_hz = band_high_hz
        self.level = 0.0
        self.fast = 0.0
        self.slow = 0.0
        self.max_energy = 1e-6
        self.band_levels = np.zeros(self.band_count, dtype=np.float32)
        self.band_max = np.ones(self.band_count, dtype=np.float32) * 1e-6
        self._band_bins: list[tuple[int, int]] | None = None
        self._band_bins_len = 0
        self._lock = threading.Lock()
        self._stream = None

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed")

        if self.samplerate is None:
            device_info = sd.query_devices(self.device, "input")
            self.samplerate = int(device_info["default_samplerate"])

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=1,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def read_level(self) -> float:
        with self._lock:
            return float(self.level)

    def read_bands(self) -> np.ndarray:
        with self._lock:
            return self.band_levels.copy()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status or indata.size == 0:
            return

        mono = np.mean(indata, axis=1)
        window = np.hanning(len(mono))
        mono = mono * window

        spectrum = np.fft.rfft(mono)
        mag = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(self.samplerate))
        band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        if not np.any(band):
            return

        band_mag = mag[band]
        band_energy = float(np.sqrt(np.mean(band_mag**2)))
        band_energy_values = None
        if self.band_count > 0:
            nyquist = 0.5 * float(self.samplerate)
            band_high = min(self.band_high_hz, nyquist)
            band_low = min(self.band_low_hz, band_high - 1.0)
            if band_high > band_low:
                if self._band_bins is None or self._band_bins_len != len(freqs):
                    edges = np.geomspace(max(1.0, band_low), band_high, self.band_count + 1)
                    bins = np.searchsorted(freqs, edges)
                    self._band_bins = [
                        (int(bins[i]), int(bins[i + 1])) for i in range(self.band_count)
                    ]
                    self._band_bins_len = len(freqs)
                band_energy_values = np.zeros(self.band_count, dtype=np.float32)
                for idx, (start, end) in enumerate(self._band_bins or []):
                    if end <= start:
                        continue
                    slice_mag = mag[start:end]
                    if slice_mag.size == 0:
                        continue
                    band_energy_values[idx] = float(np.sqrt(np.mean(slice_mag**2)))

        with self._lock:
            self.max_energy = max(band_energy, self.max_energy * 0.995)
            normalized = band_energy / (self.max_energy + 1e-9)
            self.fast = 0.6 * normalized + 0.4 * self.fast
            self.slow = 0.05 * normalized + 0.95 * self.slow
            kick = max(0.0, self.fast - self.slow)
            self.level = min(1.0, kick * 2.5)
            if band_energy_values is not None and band_energy_values.size == self.band_levels.size:
                self.band_max = np.maximum(band_energy_values, self.band_max * 0.995)
                band_normalized = band_energy_values / (self.band_max + 1e-9)
                band_normalized = np.clip(band_normalized, 0.0, 1.0)
                self.band_levels = 0.6 * band_normalized + 0.4 * self.band_levels
