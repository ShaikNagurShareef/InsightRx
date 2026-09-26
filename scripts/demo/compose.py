"""
Compose the final demo: narration placed on each scene's recorded start, burned-in captions and tech badges
(ASS via libass), upscaled to 1920x1080 H.264 + AAC. Also writes an SRT sidecar.

  python scripts/demo/compose.py [--out DIR] [--final PATH]
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from script import GAP_S, SCENES  # noqa: E402

SR = 24_000
DEFAULT_OUT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/out"
FONTS = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/fonts"
LEAD_IN = 0.35                 # narration starts shortly after the scene appears
MAX_CAPTION = 96               # characters per caption chunk (two lines at 44 px)
# broadcast-style voice polish: remove rumble, gentle compression, slight presence lift, then loudness normalisation
VOICE_FX = ("highpass=f=75,lowpass=f=12000,acompressor=threshold=-20dB:ratio=2.5:attack=8:release=160:makeup=2,"
            "equalizer=f=3200:t=q:w=1.2:g=2.5,equalizer=f=220:t=q:w=1.0:g=-1.5,loudnorm=I=-16:TP=-1.5:LRA=9")


def chunks(text):
    """Split a long caption at sentence, then clause boundaries into pieces of at most MAX_CAPTION characters."""
    if len(text) <= MAX_CAPTION:
        return [text]
    for pattern in (r"(?<=[.?!])\s+", r"(?<=[,;:])\s+"):
        parts = re.split(pattern, text)
        if len(parts) > 1:
            break
    else:                                                # no punctuation: split at the space nearest the middle
        mid = len(text) // 2
        cut = min((i for i, ch in enumerate(text) if ch == " "), key=lambda i: abs(i - mid), default=None)
        return [text] if cut is None else chunks(text[:cut]) + chunks(text[cut + 1:])
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > MAX_CAPTION:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    out.append(cur)
    return [piece for part in out for piece in (chunks(part) if len(part) > MAX_CAPTION and part != text else [part])]

ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Public Sans SemiBold,44,&H00FFFFFF,&H00FFFFFF,&H9A16100B,&H9A16100B,0,0,0,0,100,100,0,0,3,16,0,2,220,220,46,1
Style: Badge,Public Sans ExtraBold,27,&H00FFFFFF,&H00FFFFFF,&H206B6B0F,&H206B6B0F,0,0,0,0,100,100,0.4,0,3,11,0,9,40,40,34,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts_ass(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def ts_srt(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def ass_text(s):
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def build(out):
    timings = json.load(open(os.path.join(out, "timings.json")))
    rec = json.load(open(os.path.join(out, "scenes.json")))
    starts = {s["id"]: s for s in rec["scenes"]}
    dur = probe_duration(rec["video"])
    track = np.zeros(int((dur + 1) * SR), dtype=np.float32)
    events, srt, placed = [], [], []
    for sc in SCENES:
        s = starts[sc["id"]]
        t = s["start"] + LEAD_IN
        for row in timings[sc["id"]]["sentences"]:
            audio, sr = sf.read(row["file"], dtype="float32")
            assert sr == SR
            i = int(t * SR)
            track[i:i + len(audio)] += audio[: max(0, len(track) - i)]
            end = t + row["dur"]
            placed.append((t, end))
            pieces = chunks(row["caption"])
            total_chars = sum(len(p) for p in pieces)
            c0 = t
            for j, piece in enumerate(pieces):          # time each chunk by its share of the sentence
                c1 = end if j == len(pieces) - 1 else c0 + row["dur"] * len(piece) / total_chars
                tail = 0.15 if j == len(pieces) - 1 else 0.0
                events.append(f"Dialogue: 0,{ts_ass(c0)},{ts_ass(c1 + tail)},Caption,,0,0,0,,{ass_text(piece)}")
                srt.append((c0, c1 + tail, piece))
                c0 = c1
            t = end + GAP_S
        assert t - GAP_S <= s["end"] + 0.6, f"narration overruns scene {sc['id']}"
        if sc["badges"]:
            text = "\\N".join(ass_text(b) for b in sc["badges"])
            events.append(f"Dialogue: 1,{ts_ass(s['start'] + 0.4)},{ts_ass(s['end'] - 0.2)},Badge,,0,0,0,,"
                          f"{{\\fad(300,300)}}{text}")
    for (a0, a1), (b0, _) in zip(placed, placed[1:]):     # captions never overlap each other
        assert a1 <= b0 + 1e-6, "overlapping narration"
    sf.write(os.path.join(out, "narration.wav"), track[: int(dur * SR)], SR)
    open(os.path.join(out, "captions.ass"), "w").write(ASS_HEAD + "\n".join(events) + "\n")
    with open(os.path.join(out, "captions.srt"), "w") as fh:
        for k, (a, b, text) in enumerate(srt, 1):
            fh.write(f"{k}\n{ts_srt(a)} --> {ts_srt(b)}\n{text}\n\n")
    return rec["video"], dur


def render(out, video, final):
    ass = os.path.join(out, "captions.ass")
    vf = (f"fps=30,scale=1920:1080:flags=lanczos,format=yuv420p,"
          f"ass='{ass}':fontsdir='{FONTS}'")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", os.path.join(out, "narration.wav"),
           "-filter_complex", f"[0:v]{vf}[v];[1:a]aresample=48000,{VOICE_FX},aformat=channel_layouts=stereo[a]",
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "160k", "-shortest", final]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--final", default="/data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/InsightRx_demo.mp4")
    args = ap.parse_args()
    video, dur = build(args.out)
    render(args.out, video, args.final)
    subprocess.run(["cp", os.path.join(args.out, "captions.srt"), args.final.replace(".mp4", ".srt")], check=True)
    print(f"wrote {args.final} ({probe_duration(args.final):.1f} s from a {dur:.1f} s recording)")


if __name__ == "__main__":
    main()
