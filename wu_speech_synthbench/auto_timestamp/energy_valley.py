from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CharSpan:
    character: str
    start_time: float
    end_time: float
    source: str


@dataclass(frozen=True)
class WordSpan:
    start_time: float
    end_time: float
    source: str
    threshold_db: float
    characters: list[CharSpan]


def bandpass_filter(
    signal: np.ndarray,
    sample_rate: int,
    low_hz: float = 300.0,
    high_hz: float = 3400.0,
) -> np.ndarray:
    """Zero-phase FFT speech-bandpass filter (pure NumPy, safe for any sample_rate)."""
    audio = np.asarray(signal, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) < 4 or sample_rate <= 0:
        return audio
    nyquist = sample_rate / 2.0
    effective_low = max(0.0, min(low_hz, nyquist * 0.8))
    effective_high = min(high_hz, nyquist * 0.98)
    if effective_low >= effective_high:
        return audio
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    mask = np.zeros_like(freqs, dtype=np.float64)
    in_band = (freqs >= effective_low) & (freqs <= effective_high)
    mask[in_band] = 1.0
    taper = max(25.0, 0.05 * (effective_high - effective_low))
    low_taper = (freqs >= max(0.0, effective_low - taper)) & (freqs < effective_low)
    if np.any(low_taper):
        mask[low_taper] = 0.5 * (1.0 + np.cos(np.pi * (effective_low - freqs[low_taper]) / taper))
    high_taper = (freqs > effective_high) & (freqs <= min(nyquist, effective_high + taper))
    if np.any(high_taper):
        mask[high_taper] = 0.5 * (1.0 + np.cos(np.pi * (freqs[high_taper] - effective_high) / taper))
    return np.fft.irfft(spectrum * mask, n=len(audio))


def _make_window(window_length: int, window_type: str = "hann") -> np.ndarray:
    if window_type == "hann":
        return np.hanning(window_length)
    if window_type == "hamming":
        return np.hamming(window_length)
    if window_type == "rect":
        return np.ones(window_length, dtype=np.float64)
    raise ValueError(f"unsupported window_type: {window_type!r}")


def _frame_audio(
    signal: np.ndarray,
    sample_rate: int,
    hop_seconds: float,
    window_seconds: float,
    window_type: str = "hann",
) -> tuple[np.ndarray, float, float]:
    audio = np.asarray(signal, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or len(audio) < 2 or sample_rate <= 0:
        raise ValueError("signal must be one-dimensional with a positive sample rate")
    hop = int(round(sample_rate * hop_seconds))
    window = int(round(sample_rate * window_seconds))
    if hop < 1 or window < 2:
        raise ValueError("sample_rate is too small for energy framing")
    count = 1 + max(0, (len(audio) - window) // hop)
    indices = np.arange(count)[:, None] * hop + np.arange(window)[None, :]
    indices = np.clip(indices, 0, len(audio) - 1)
    weights = _make_window(window, window_type)
    frames = audio[indices] * weights[None, :]
    return frames, float(hop / sample_rate), float(window / sample_rate)


def rms_energy_db(
    signal: np.ndarray,
    sample_rate: int,
    hop_seconds: float = 0.010,
    window_seconds: float = 0.025,
    window_type: str = "hann",
    bandpass: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Return windowed short-time RMS energy in dB and exact hop duration."""
    audio = np.asarray(signal, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if bandpass is not None:
        audio = bandpass_filter(audio, sample_rate, bandpass[0], bandpass[1])
    frames, hop, _ = _frame_audio(audio, sample_rate, hop_seconds, window_seconds, window_type)
    weights = _make_window(frames.shape[1], window_type)
    norm_factor = max(float(np.sum(weights ** 2)), 1e-12)
    rms = np.sqrt(np.sum(frames ** 2, axis=1) / norm_factor + 1e-12)
    return 20.0 * np.log10(rms), hop


def acoustic_features(
    signal: np.ndarray,
    sample_rate: int,
    hop_seconds: float = 0.010,
    window_seconds: float = 0.025,
    window_type: str = "hann",
    bandpass: tuple[float, float] | None = (300.0, 3400.0),
    high_band_hz: tuple[float, float] = (2000.0, 4000.0),
) -> dict[str, Any]:
    """Compute windowed speech-band energy, Parseval-consistent high-band energy, and spectral flux.

    In v3, ``high_band_db`` obeys Parseval's theorem for real-signal one-sided FFTs so that
    narrowband voiceless fricatives/aspirates are measured on the exact same physical dB
    scale as time-domain ``energy_db`` regardless of sample rate or window length.
    """
    raw = np.asarray(signal, dtype=np.float64)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    speech = bandpass_filter(raw, sample_rate, bandpass[0], bandpass[1]) if bandpass is not None else raw
    frames, hop, window_dur = _frame_audio(speech, sample_rate, hop_seconds, window_seconds, window_type)
    n_fft = frames.shape[1]
    weights = _make_window(n_fft, window_type)
    norm_factor = max(float(np.sum(weights ** 2)), 1e-12)
    rms = np.sqrt(np.sum(frames ** 2, axis=1) / norm_factor + 1e-12)
    energy_db = 20.0 * np.log10(rms)

    # Parseval-consistent high-frequency subband energy (protects voiceless fricatives/aspirates)
    raw_frames, _, _ = _frame_audio(raw, sample_rate, hop_seconds, window_seconds, window_type)
    mag = np.abs(np.fft.rfft(raw_frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    # One-sided rfft Parseval weights: 1 for DC and Nyquist (if n_fft even), 2 for interior bins
    parseval_w = np.full(len(freqs), 2.0, dtype=np.float64)
    parseval_w[0] = 1.0
    if n_fft % 2 == 0:
        parseval_w[-1] = 1.0

    nyquist = sample_rate / 2.0
    hi_low = min(high_band_hz[0], nyquist * 0.7)
    hi_high = min(high_band_hz[1], nyquist)
    hi_mask = (freqs >= hi_low) & (freqs <= hi_high)
    if np.any(hi_mask):
        # sum(|X_k|^2 * w_k) / (N * sum(win^2)) == time-domain windowed mean-square power in band
        hi_power = np.sum((mag[:, hi_mask] ** 2) * parseval_w[hi_mask][None, :], axis=1) / (n_fft * norm_factor)
        high_band_db = 10.0 * np.log10(hi_power + 1e-12)
    else:
        high_band_db = energy_db.copy()

    # Half-wave rectified spectral flux
    mag_norm = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + 1e-12)
    diff = np.diff(mag_norm, axis=0, prepend=mag_norm[:1])
    spectral_flux = np.sum(np.maximum(diff, 0.0), axis=1)

    fused_db = np.maximum(energy_db, high_band_db - 6.0)
    return {
        "energy_db": energy_db,
        "high_band_db": high_band_db,
        "fused_db": fused_db,
        "spectral_flux": spectral_flux,
        "hop": hop,
        "window_seconds": window_dur,
        "duration": len(raw) / float(sample_rate),
    }


def _time_to_frame(time_s: float, hop: float, half_window: float, num_frames: int) -> int:
    frame = int(round((time_s - half_window) / hop))
    return max(0, min(frame, num_frames - 1))


def _frame_to_center_time(frame: int, hop: float, half_window: float, duration: float) -> float:
    return float(np.clip(frame * hop + half_window, 0.0, duration))


def _refine_edge(
    energy_db: np.ndarray,
    hop: float,
    half_window: float,
    anchor_time: float,
    direction: str,
    relative_db: float,
    hold_frames: int,
    local_peak_seconds: float,
    max_walk_seconds: float,
) -> tuple[int, float]:
    """Refine outer boundary without silence overshoot or boundary anchor lock."""
    n_frames = len(energy_db)
    anchor = _time_to_frame(anchor_time, hop, half_window, n_frames)
    peak_radius = max(1, int(round(local_peak_seconds / hop)))
    local_start = max(0, anchor - peak_radius)
    local_end = min(n_frames, anchor + peak_radius + 1)
    block = energy_db[local_start:local_end]
    quiet = float(np.percentile(energy_db, 5))
    threshold = max(float(block.max()) - relative_db, quiet + 3.0)

    frame = anchor
    last_active = anchor
    miss = 0
    step = -1 if direction == "left" else 1
    limit = max(1, int(round(max_walk_seconds / hop)))

    while abs(frame - anchor) < limit:
        candidate = frame + step
        if candidate < 0 or candidate >= n_frames:
            break
        if energy_db[candidate] < threshold:
            miss += 1
            if miss >= hold_frames:
                break
        else:
            miss = 0
            last_active = candidate
        frame = candidate

    return last_active, threshold


def _resample_posteriors(
    posterior_probs: np.ndarray,
    target_frames: int,
    posterior_hop_seconds: float | None,
    hop: float,
    half_window: float,
    posterior_offset_seconds: float = 0.0,
) -> np.ndarray:
    """Resample external frame posteriors onto the exact acoustic frame-center time grid."""
    arr = np.asarray(posterior_probs, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError("posterior_probs must be 1D [frames] or 2D [frames, classes]")
    src_hop = posterior_hop_seconds if posterior_hop_seconds is not None else hop
    src_times = np.arange(arr.shape[0], dtype=np.float64) * src_hop + posterior_offset_seconds
    # Align target timestamps to frame centers (frame * hop + half_window)
    dst_times = np.arange(target_frames, dtype=np.float64) * hop + half_window
    if (
        arr.shape[0] == target_frames
        and abs(src_hop - hop) < 1e-9
        and abs(posterior_offset_seconds - half_window) < 1e-9
    ):
        return arr
    resampled = np.empty((target_frames, arr.shape[1]), dtype=np.float64)
    for col in range(arr.shape[1]):
        resampled[:, col] = np.interp(dst_times, src_times, arr[:, col])
    return resampled


def _local_minmax(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    if span < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(values.min())) / span


def segment_word(
    signal: np.ndarray,
    sample_rate: int,
    characters: str,
    anchor_times: Sequence[float],
    relative_db: float = 18.0,
    hold_frames: int = 2,
    reach_seconds: float = 0.12,
    max_walk_seconds: float = 0.24,
    hop_seconds: float = 0.010,
    window_seconds: float = 0.025,
    window_type: str = "hann",
    bandpass: tuple[float, float] | None = (300.0, 3400.0),
    flux_weight: float = 0.25,
    posterior_probs: np.ndarray | None = None,
    posterior_hop_seconds: float | None = None,
    posterior_offset_seconds: float = 0.0,
    posterior_weight: float = 0.5,
) -> WordSpan:
    """Expand char anchors outward and cut internal boundaries using local energy + flux (+ posteriors).

    Improvements in v3:
    - Normalizes energy and spectral flux locally inside the word span ``[left : right + 1]``
      so utterance-edge digital silence (-120 dB) never dilutes in-word energy valleys.
    - Uses Parseval-consistent high-band energy for voiceless fricatives/aspirates.
    - Resamples external posteriors directly onto frame-center times (``frame * hop + half_window``).
    """
    if len(characters) != len(anchor_times):
        raise ValueError("characters and anchor_times must have equal length")
    if len(characters) < 1:
        raise ValueError("at least one character is required")
    anchors = np.asarray(anchor_times, dtype=np.float64)
    if np.any(np.diff(anchors) <= 0):
        raise ValueError("anchor_times must be strictly increasing")

    feats = acoustic_features(
        signal,
        sample_rate,
        hop_seconds=hop_seconds,
        window_seconds=window_seconds,
        window_type=window_type,
        bandpass=bandpass,
    )
    energy_db = feats["fused_db"]
    flux = feats["spectral_flux"]
    hop = feats["hop"]
    half_window = feats["window_seconds"] / 2.0
    duration = feats["duration"]
    n_frames = len(energy_db)

    left, left_threshold = _refine_edge(
        energy_db, hop, half_window, float(anchors[0]), "left", relative_db, hold_frames, reach_seconds, max_walk_seconds
    )
    right, right_threshold = _refine_edge(
        energy_db, hop, half_window, float(anchors[-1]), "right", relative_db, hold_frames, reach_seconds, max_walk_seconds
    )
    if right < left:
        right = left

    # Local normalization inside the active word span [left : right + 1]
    word_energy_norm = _local_minmax(energy_db[left : right + 1])
    word_flux_norm = _local_minmax(flux[left : right + 1])
    word_valley_cost = word_energy_norm - flux_weight * word_flux_norm

    resampled_post: np.ndarray | None = None
    has_posteriors = posterior_probs is not None
    if has_posteriors:
        resampled_post = _resample_posteriors(
            posterior_probs,
            target_frames=n_frames,
            posterior_hop_seconds=posterior_hop_seconds,
            hop=hop,
            half_window=half_window,
            posterior_offset_seconds=posterior_offset_seconds,
        )

    boundaries = [left]
    for char_idx, (left_anchor, right_anchor) in enumerate(zip(anchors[:-1], anchors[1:])):
        low = max(left, _time_to_frame(float(left_anchor), hop, half_window, n_frames))
        high = min(right, _time_to_frame(float(right_anchor), hop, half_window, n_frames))
        if high > low:
            rel_low = low - left
            rel_high = high - left
            segment_cost = word_valley_cost[rel_low : rel_high + 1].copy()
            if resampled_post is not None:
                if resampled_post.shape[1] >= len(characters):
                    p_left = resampled_post[low : high + 1, char_idx]
                    p_right = resampled_post[low : high + 1, char_idx + 1]
                    crossover_cost = np.abs(p_left - p_right)
                    segment_cost = (1.0 - posterior_weight) * segment_cost + posterior_weight * crossover_cost
                else:
                    p_val = resampled_post[low : high + 1, 0]
                    segment_cost = (1.0 - posterior_weight) * segment_cost + posterior_weight * p_val
            cut = low + int(np.argmin(segment_cost))
            boundaries.append(cut)
        else:
            boundaries.append((low + high) // 2)
    boundaries.append(right)
    boundaries = sorted(boundaries)

    word_source = "anchors+energy+posterior" if has_posteriors else "anchors+energy"
    char_source = "energy-valley+posterior" if has_posteriors else "energy-valley"

    word_start = _frame_to_center_time(boundaries[0], hop, half_window, duration)
    word_end = _frame_to_center_time(boundaries[-1], hop, half_window, duration)
    spans = [
        CharSpan(
            character=character,
            start_time=_frame_to_center_time(left_index, hop, half_window, duration),
            end_time=_frame_to_center_time(right_index, hop, half_window, duration),
            source=char_source,
        )
        for character, left_index, right_index in zip(characters, boundaries[:-1], boundaries[1:])
    ]
    threshold = min(left_threshold, right_threshold)
    return WordSpan(word_start, word_end, word_source, threshold, spans)
