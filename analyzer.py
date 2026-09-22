from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import essentia.standard as es
except ImportError as e:
    raise SystemExit(
        "Essentia is not installed. Install dependencies from requirements.txt."
    ) from e


@dataclass
class Event:
    time: float
    type: str
    strength: float
    confidence: float

@dataclass
class AnalysisResult:
    schema_version: str
    source_file: str
    duration_sec: float
    bpm: float
    bpm_confidence: float
    beats: List[float]
    events: List[Event]
    sync_points: List[Event]


def _robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        std = np.std(x)
        scale = std if std > 1e-9 else 1.0
    return (x - med) / scale


def _merge_events(events: List[Event], tolerance: float = 0.09) -> List[Event]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e.time)
    merged: List[Event] = [events[0]]
    priority = {"DROP": 5, "BASS_PEAK": 4, "STRONG_BEAT": 3, "TRANSITION": 2, "BEAT": 1}
    for e in events[1:]:
        prev = merged[-1]
        if abs(e.time - prev.time) <= tolerance:
            if (priority.get(e.type, 0), e.strength) > (priority.get(prev.type, 0), prev.strength):
                merged[-1] = e
        else:
            merged.append(e)
    return merged


def analyze_audio(path: str, max_sync_points: int = 24) -> AnalysisResult:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    loader = es.MonoLoader(filename=str(audio_path), sampleRate=44100)
    audio = loader()
    sr = 44100
    duration = len(audio) / sr

    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, ticks, confidence, _, _ = rhythm(audio)
    ticks = np.asarray(ticks, dtype=float)

    # Frame-level spectral analysis.
    frame_size, hop = 2048, 512
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)

    bass_mask = (freqs >= 25) & (freqs <= 160)
    lowmid_mask = (freqs > 160) & (freqs <= 500)

    times, bass_energy, total_energy, flux = [], [], [], []
    prev_mag = None
    for i, frame in enumerate(es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop, startFromZero=True)):
        mag = np.asarray(spectrum(window(frame)), dtype=float)
        power = mag * mag
        times.append((i * hop + frame_size / 2) / sr)
        bass_energy.append(float(power[bass_mask].sum()))
        total_energy.append(float(power.sum()))
        if prev_mag is None:
            flux.append(0.0)
        else:
            d = np.maximum(mag - prev_mag, 0.0)
            flux.append(float(np.dot(d, d)))
        prev_mag = mag

    times = np.asarray(times)
    bass_z = _robust_z(np.log1p(np.asarray(bass_energy)))
    energy_z = _robust_z(np.log1p(np.asarray(total_energy)))
    flux_z = _robust_z(np.log1p(np.asarray(flux)))

    events: List[Event] = []

    # Every beat is retained; strength is estimated from nearby frame energy/flux.
    for t in ticks:
        idx = int(np.argmin(np.abs(times - t))) if times.size else 0
        s = float(max(0.0, 0.55 * energy_z[idx] + 0.45 * flux_z[idx])) if times.size else 0.0
        events.append(Event(float(t), "BEAT", round(s, 4), round(float(confidence), 4)))

    # Strong beats: top local beat accents.
    if len(ticks):
        beat_scores = []
        for t in ticks:
            idx = int(np.argmin(np.abs(times - t)))
            score = 0.45 * energy_z[idx] + 0.35 * flux_z[idx] + 0.20 * bass_z[idx]
            beat_scores.append(score)
        beat_scores = np.asarray(beat_scores)
        threshold = max(1.0, float(np.percentile(beat_scores, 75)))
        for t, score in zip(ticks, beat_scores):
            if score >= threshold:
                conf = min(1.0, 0.55 + max(0.0, score) / 6.0)
                events.append(Event(float(t), "STRONG_BEAT", round(float(score), 4), round(conf, 4)))

    # Bass peaks: local maxima in robust low-frequency energy.
    for i in range(1, max(1, len(times) - 1)):
        if bass_z[i] > 1.8 and bass_z[i] >= bass_z[i-1] and bass_z[i] > bass_z[i+1]:
            events.append(Event(float(times[i]), "BASS_PEAK", round(float(bass_z[i]), 4),
                                round(min(1.0, 0.55 + bass_z[i] / 8.0), 4)))

    # Transitions / drops: spectral-flux + energy discontinuities.
    # "DROP" is a heuristic candidate, not a semantic guarantee.
    for i in range(2, max(2, len(times) - 2)):
        transition_score = 0.60 * flux_z[i] + 0.40 * abs(energy_z[i] - energy_z[i-2])
        if transition_score > 2.4:
            typ = "DROP" if energy_z[i] > energy_z[i-2] and bass_z[i] > 0.8 else "TRANSITION"
            conf = min(0.95, 0.50 + max(0.0, transition_score) / 10.0)
            events.append(Event(float(times[i]), typ, round(float(transition_score), 4), round(conf, 4)))

    events = _merge_events(events)

    # DIRECTOR-oriented shortlist. Favor meaningful high-strength events and spacing.
    candidates = [e for e in events if e.type != "BEAT"]
    candidates.sort(key=lambda e: (e.confidence * max(e.strength, 0.1)), reverse=True)
    selected: List[Event] = []
    for e in candidates:
        if all(abs(e.time - s.time) >= 0.35 for s in selected):
            selected.append(e)
        if len(selected) >= max_sync_points:
            break
    selected.sort(key=lambda e: e.time)

    return AnalysisResult(
        schema_version="1.0",
        source_file=audio_path.name,
        duration_sec=round(float(duration), 3),
        bpm=round(float(bpm), 3),
        bpm_confidence=round(float(confidence), 4),
        beats=[round(float(x), 4) for x in ticks],
        events=events,
        sync_points=selected,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Black Red Line AUDIO ANALYZER 1.0")
    parser.add_argument("audio", help="Path to MP3/WAV/AAC supported by Essentia/FFmpeg build")
    parser.add_argument("-o", "--output", default="analysis.json")
    parser.add_argument("--max-sync-points", type=int, default=24)
    args = parser.parse_args()

    result = analyze_audio(args.audio, args.max_sync_points)
    payload = asdict(result)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
