"""导出标准 MIDI 文件（带速度与拍号）。"""
from __future__ import annotations

import io

import mido

PPQ = 480


def export_midi(notes: list[dict], bpm: float, beats_per_bar: int, origin: float,
                align: str = "bar", name: str = "OptMusic") -> bytes:
    """align="bar"：第 1 小节对齐到 MIDI 0 点（适合进 DAW 按小节编辑）；
    align="audio"：音频起点对齐到 MIDI 0 点。"""
    zero = origin if align == "bar" else 0.0
    ticks_per_sec = bpm / 60.0 * PPQ

    mid = mido.MidiFile(ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=beats_per_bar,
                                  denominator=8 if beats_per_bar in (6, 9, 12) else 4, time=0))

    events = []
    for n in notes:
        s = max(0, round((n["start"] - zero) * ticks_per_sec))
        e = max(s + 1, round((n["end"] - zero) * ticks_per_sec))
        v = int(max(1, min(127, n.get("velocity", 100))))
        events.append((s, 1, mido.Message("note_on", note=int(n["pitch"]), velocity=v)))
        events.append((e, 0, mido.Message("note_off", note=int(n["pitch"]), velocity=0)))
    events.sort(key=lambda x: (x[0], x[1]))

    last = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()
