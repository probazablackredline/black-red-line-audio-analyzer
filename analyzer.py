from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import essentia.standard as es
import numpy as np

SCHEMA_VERSION = "1.1"
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 512
BASS_MIN_HZ = 25.0
BASS_MAX_HZ = 160.0


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
    beats: list[float]
    events: list[dict[str, Any]]
    sync_points: list[dict[str, Any]]


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad

    if scale < 1e-9:
        std = np.std(values)
        scale = std if std > 1e-9 else 1.0

    return (values - median) / scale


def _normalize_strength(value: float) -> float:
    return float(max(0.0, value))


def _merge_nearby(
    events: list[Event],
    window_sec: float = 0.32,
) -> list[Event]:
    """Merge detections that describe the same musical moment."""

    if not events:
        return []

    priority = {
        "DROP": 4,
        "TRANSITION": 3,
        "BASS_PEAK": 2,
        "STRONG_BEAT": 1,
        "BEAT": 0,
    }

    ordered = sorted(events, key=lambda e: e.time)
    groups: list[list[Event]] = [[ordered[0]]]

    for event in ordered[1:]:
        if event.time - groups[-1][-1].time <= window_sec:
            groups[-1].append(event)
        else:
            groups.append([event])

    merged: list[Event] = []

    for group in groups:
        best = max(
            group,
            key=lambda e: (
                e.strength + priority.get(e.type, 0) * 0.35,
                priority.get(e.type, 0),
            ),
        )

        combined_strength = max(e.strength for e in group)
        combined_confidence = max(e.confidence for e in group)

        merged.append(
            Event(
                time=best.time,
                type=best.type,
                strength=round(combined_strength, 4),
                confidence=round(combined_confidence, 4),
            )
        )

    return merged


def _suppress_close_drops(
    events: list[Event],
    min_gap_sec: float = 2.5,
) -> list[Event]:
    """Keep only the strongest DROP inside a short musical window."""

    drops = [e for e in events if e.type == "DROP"]
    others = [e for e in events if e.type != "DROP"]

    kept: list[Event] = []

    for event in sorted(
        drops,
        key=lambda e: e.strength,
        reverse=True,
    ):
        if all(
            abs(event.time - kept_event.time) >= min_gap_sec
            for kept_event in kept
        ):
            kept.append(event)

    return sorted(others + kept, key=lambda e: e.time)


def _balanced_sync_points(
    events: list[Event],
    max_points: int,
) -> list[dict[str, Any]]:
    """
    Build a montage-oriented shortlist instead of simply
    taking the loudest events.

    Major changes are preferred, but bass hits and strong
    beats are deliberately retained.
    """

    if max_points <= 0:
        return []

    quotas = {
        "DROP": max(1, round(max_points * 0.35)),
        "TRANSITION": max(1, round(max_points * 0.25)),
        "BASS_PEAK": max(1, round(max_points * 0.25)),
        "STRONG_BEAT": max(1, round(max_points * 0.15)),
    }

    selected: list[Event] = []
    min_spacing = 0.75

    def can_add(candidate: Event) -> bool:
        return all(
            abs(candidate.time - selected_event.time) >= min_spacing
            for selected_event in selected
        )

    for kind in (
        "DROP",
        "TRANSITION",
        "BASS_PEAK",
        "STRONG_BEAT",
    ):
        candidates = sorted(
            (e for e in events if e.type == kind),
            key=lambda e: (e.strength, e.confidence),
            reverse=True,
        )

        count = 0

        for event in candidates:
            if count >= quotas[kind]:
                break

            if can_add(event):
                selected.append(event)
                count += 1

    if len(selected) < max_points:
        leftovers = sorted(
            (
                e
                for e in events
                if e.type != "BEAT" and e not in selected
            ),
            key=lambda e: (e.strength, e.confidence),
            reverse=True,
        )

        for event in leftovers:
            if len(selected) >= max_points:
                break

            if can_add(event):
                selected.append(event)

    selected = sorted(
        selected[:max_points],
        key=lambda e: e.time,
    )

    return [
        {
            "time": round(e.time, 4),
            "type": e.type,
            "strength": round(e.strength, 4),
            "confidence": round(e.confidence, 4),
        }
        for e in selected
    ]


def analyze_audio(
    path: str | Path,
    max_sync_points: int = 24,
) -> AnalysisResult:

    path = Path(path)

    audio = es.MonoLoader(
        filename=str(path),
        sampleRate=SAMPLE_RATE,
    )()

    duration_sec = len(audio) / SAMPLE_RATE

    rhythm = es.RhythmExtractor2013(
        method="multifeature",
    )

    bpm, ticks, confidence, _, _ = rhythm(audio)

    beats = [
        round(float(t), 4)
        for t in ticks
    ]

    window = es.Windowing(type="hann")
    spectrum = es.Spectrum(size=FRAME_SIZE)

    freqs = np.fft.rfftfreq(
        FRAME_SIZE,
        d=1.0 / SAMPLE_RATE,
    )

    bass_mask = (
        (freqs >= BASS_MIN_HZ)
        & (freqs <= BASS_MAX_HZ)
    )

    bass_energy = []
    spectral_flux = []
    prev_spec = None

    for frame in es.FrameGenerator(
        audio,
        frameSize=FRAME_SIZE,
        hopSize=HOP_SIZE,
        startFromZero=True,
    ):
        spec = np.asarray(
            spectrum(window(frame)),
            dtype=np.float64,
        )

        bass_energy.append(
            float(np.sum(spec[bass_mask] ** 2))
        )

        if prev_spec is None:
            spectral_flux.append(0.0)
        else:
            diff = np.maximum(
                spec - prev_spec,
                0.0,
            )

            spectral_flux.append(
                float(np.sum(diff))
            )

        prev_spec = spec

    bass_z = robust_zscore(
        np.asarray(bass_energy)
    )

    flux_z = robust_zscore(
        np.asarray(spectral_flux)
    )

    frame_times = (
        np.arange(len(bass_z))
        * HOP_SIZE
        / SAMPLE_RATE
    )

    raw_events: list[Event] = []

    beat_interval = (
        60.0 / float(bpm)
        if bpm > 0
        else 0.5
    )

    for i, beat in enumerate(beats):
        raw_events.append(
            Event(
                beat,
                "BEAT",
                1.0,
                min(1.0, float(confidence)),
            )
        )

        if i % 4 == 0:
            raw_events.append(
                Event(
                    beat,
                    "STRONG_BEAT",
                    2.0,
                    min(1.0, float(confidence)),
                )
            )

    for i in range(1, len(bass_z) - 1):
        if (
            bass_z[i] >= 3.0
            and bass_z[i] >= bass_z[i - 1]
            and bass_z[i] > bass_z[i + 1]
        ):
            raw_events.append(
                Event(
                    float(frame_times[i]),
                    "BASS_PEAK",
                    _normalize_strength(
                        float(bass_z[i])
                    ),
                    0.85,
                )
            )

    for i in range(1, len(flux_z) - 1):
        if (
            flux_z[i] >= 4.0
            and flux_z[i] >= flux_z[i - 1]
            and flux_z[i] > flux_z[i + 1]
        ):
            raw_events.append(
                Event(
                    float(frame_times[i]),
                    "TRANSITION",
                    _normalize_strength(
                        float(flux_z[i])
                    ),
                    0.82,
                )
            )

    for i in range(
        1,
        min(len(bass_z), len(flux_z)) - 1,
    ):
        if (
            flux_z[i] >= 4.5
            and bass_z[i] >= 3.0
            and flux_z[i] >= flux_z[i - 1]
            and flux_z[i] > flux_z[i + 1]
        ):
            score = float(
                0.6 * flux_z[i]
                + 0.4 * bass_z[i]
            )

            raw_events.append(
                Event(
                    float(frame_times[i]),
                    "DROP",
                    _normalize_strength(score),
                    0.88,
                )
            )

    major = [
        e
        for e in raw_events
        if e.type != "BEAT"
    ]

    beats_only = [
        e
        for e in raw_events
        if e.type == "BEAT"
    ]

    major = _merge_nearby(
        major,
        window_sec=max(
            0.24,
            min(
                0.38,
                beat_interval * 0.8,
            ),
        ),
    )

    major = _suppress_close_drops(
        major,
        min_gap_sec=max(
            2.0,
            beat_interval * 4.0,
        ),
    )

    events = sorted(
        beats_only + major,
        key=lambda e: e.time,
    )

    # Исправлено:
    # функция ожидает max_points, а не max_sync_points.
    sync_points = _balanced_sync_points(
        major,
        max_points=max_sync_points,
    )

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        source_file=path.name,
        duration_sec=round(
            float(duration_sec),
            4,
        ),
        bpm=round(
            float(bpm),
            4,
        ),
        bpm_confidence=round(
            float(confidence),
            4,
        ),
        beats=beats,
        events=[
            {
                "time": round(e.time, 4),
                "type": e.type,
                "strength": round(
                    e.strength,
                    4,
                ),
                "confidence": round(
                    e.confidence,
                    4,
                ),
            }
            for e in events
        ],
        sync_points=sync_points,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Black Red Line Audio Analyzer 1.1"
    )

    parser.add_argument(
        "audio",
        help="Path to an audio file",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="analysis.json",
        help="Output JSON file",
    )

    parser.add_argument(
        "--max-sync-points",
        type=int,
        default=24,
    )

    args = parser.parse_args()

    result = analyze_audio(
        args.audio,
        max_sync_points=args.max_sync_points,
    )

    Path(args.output).write_text(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(args.output)


if __name__ == "__main__":
    main()
