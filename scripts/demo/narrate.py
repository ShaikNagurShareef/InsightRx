"""Generate the voiceover with Kokoro-82M (local, GPU): one WAV per sentence plus timings.json.

  python scripts/demo/narrate.py [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from script import GAP_S, SCENE_PAD_S, SCENES, SPEED, VOICE, sentences  # noqa: E402

SR = 24_000
DEFAULT_OUT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    timings, total = {}, 0.0
    for sc in SCENES:
        rows = []
        for i, (caption, spoken) in enumerate(sentences(sc)):
            audio = np.concatenate([a for _, _, a in pipe(spoken, voice=VOICE, speed=SPEED)])
            path = os.path.join(args.out, "audio", f"{sc['id']}_{i}.wav")
            sf.write(path, audio, SR)
            rows.append({"caption": caption, "file": path, "dur": round(len(audio) / SR, 3)})
        speech = sum(r["dur"] for r in rows) + GAP_S * (len(rows) - 1)
        timings[sc["id"]] = {"sentences": rows, "speech": round(speech, 3), "min_len": round(speech + SCENE_PAD_S, 3)}
        total += speech + SCENE_PAD_S
        print(f"{sc['id']:11s} {speech:6.1f} s  ({len(rows)} sentences)")
    json.dump(timings, open(os.path.join(args.out, "timings.json"), "w"), indent=1)
    print(f"narration total with pads: {total:.1f} s ({int(total // 60)}:{int(total % 60):02d})")


if __name__ == "__main__":
    main()
