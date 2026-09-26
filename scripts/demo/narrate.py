"""Generate the voiceover: one WAV per sentence plus timings.json.

Engines
  vibevoice (default)   Microsoft VibeVoice-1.5B, MIT, local GPU: long-form, human-sounding narration. Each scene is read
                        as one continuous take and cut at sentence pauses (Whisper word timestamps). Run with vv-venv.
  chatterbox            Resemble AI Chatterbox, MIT, local GPU: expressive single sentences. Run with tts-venv.
  kokoro                Kokoro-82M, fast fallback. Run with the main venv.

Quality loop (chatterbox): every clip is transcribed with faster-whisper and compared with the script. A clip that drops,
garbles or hallucinates words (low word match, or an implausible speaking rate) is regenerated with a new seed and
the best take is kept. Clips are trimmed of edge silence and given short fades so sentences join naturally.

  vv-venv/bin/python scripts/demo/narrate.py [--engine vibevoice|chatterbox|kokoro] [--out DIR] [--only SCENE_ID ...]
"""
import argparse
import difflib
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from script import (CFG_WEIGHT, ENGINE, EXAGGERATION, GAP_S, SCENE_PAD_S, SCENES, SPEED, VOICE, VV_CFG,  # noqa: E402
                    VV_MODEL, VV_VOICE, sentences)

SR = 24_000
DEFAULT_OUT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/demo/out"
MAX_TRIES = 4
MIN_MATCH = 0.82               # word-sequence similarity between script and transcript
RATE_RANGE = (9.0, 20.0)       # characters per second: outside this, a take is cut off or rambling
FADE_S = 0.015


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))


def match(expected, heard):
    return difflib.SequenceMatcher(None, words(expected), words(heard)).ratio()


def tidy(audio, sr=SR, thresh=0.012, pad_s=0.05):
    """Trim leading/trailing silence and apply short fades (no clicks between sentences)."""
    idx = np.where(np.abs(audio) > thresh)[0]
    if len(idx):
        pad = int(pad_s * sr)
        audio = audio[max(0, idx[0] - pad): min(len(audio), idx[-1] + pad)]
    n = max(1, int(FADE_S * sr))
    if len(audio) > 2 * n:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        audio = audio.copy()
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio.astype(np.float32)


class Chatterbox:
    def __init__(self):
        import torch
        from chatterbox.tts import ChatterboxTTS
        from faster_whisper import WhisperModel
        self.torch = torch
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.asr = WhisperModel("medium.en", device="cuda", device_index=int(os.environ.get("ASR_GPU_INDEX", "0")), compute_type="float16")

    def say(self, spoken, caption, seed_base):
        best = None
        for attempt in range(MAX_TRIES):
            self.torch.manual_seed(seed_base + attempt * 7919)
            wav = self.model.generate(spoken, exaggeration=EXAGGERATION, cfg_weight=CFG_WEIGHT)
            audio = tidy(wav.squeeze(0).cpu().numpy())
            path = "/tmp/_narrate_take.wav"
            sf.write(path, audio, SR)
            heard = " ".join(s.text for s in self.asr.transcribe(path, beam_size=5)[0])
            score = max(match(spoken, heard), match(caption, heard))
            rate = len(spoken) / max(0.1, len(audio) / SR)
            ok = score >= MIN_MATCH and RATE_RANGE[0] <= rate <= RATE_RANGE[1]
            take = {"audio": audio, "score": score, "rate": rate, "heard": heard.strip(), "attempt": attempt}
            if best is None or (ok, score) > (best["ok"], best["score"]):
                best = {**take, "ok": ok}
            if ok:
                break
        return best


class VibeVoice:
    """Microsoft VibeVoice-1.5B: reads a whole scene as one continuous take (natural breathing, pauses and intonation),
    then Whisper word timestamps locate each sentence so the take can be cut at the pauses between sentences."""

    def __init__(self):
        import torch
        from faster_whisper import WhisperModel
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        self.torch = torch
        self.proc = VibeVoiceProcessor.from_pretrained(VV_MODEL)
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            VV_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
        self.model.set_ddpm_inference_steps(num_steps=10)
        self.asr = WhisperModel("medium.en", device="cuda", device_index=int(os.environ.get("ASR_GPU_INDEX", "0")), compute_type="float16")

    def _generate(self, text, attempt):
        inputs = self.proc(text=[f"Speaker 1: {text}"], voice_samples=[[VV_VOICE]], padding=True, return_tensors="pt",
                           return_attention_mask=True)
        inputs = {k: (x.to("cuda") if self.torch.is_tensor(x) else x) for k, x in inputs.items()}
        self.torch.manual_seed(1234 + attempt)
        cfg = {"do_sample": False} if attempt == 0 else {"do_sample": True, "temperature": 0.9, "top_p": 0.95}
        out = self.model.generate(**inputs, max_new_tokens=None, cfg_scale=VV_CFG, tokenizer=self.proc.tokenizer,
                                  generation_config=cfg, verbose=False)
        return tidy(out.speech_outputs[0].float().squeeze().cpu().numpy())

    def scene(self, spoken_list, captions):
        """-> (list of sentence clips, best ASR match, transcript)."""
        text = " ".join(spoken_list)
        best = None
        for attempt in range(3):
            audio = self._generate(text, attempt)
            path = "/tmp/_narrate_scene.wav"
            sf.write(path, audio, SR)
            segs, _ = self.asr.transcribe(path, beam_size=5, word_timestamps=True)
            wl = [(w.word, w.start, w.end) for s in segs for w in s.words]
            heard = "".join(w for w, _, _ in wl)
            score = max(match(text, heard), match(" ".join(captions), heard))
            if best is None or score > best[1]:
                best = (audio, score, heard, wl)
            if score >= MIN_MATCH:
                break
        audio, score, heard, wl = best
        return split_at_sentences(audio, spoken_list, wl), score, heard.strip()


# VibeVoice is language-model based and reads natural written text best: undo the letter-by-letter spellings that
# help sentence-level engines, but keep spoken-out numbers and plain-English expansions.
VV_FORMS = [
    ("I'm Nuh-goor Sha-reef Shaik, with Saa-hith Reddy Thoo-mala, Pra-nuv Naa-go-thoo, and Geeth-aanjali Naa-ga-boyna",
     "I'm Nagur Shareef Shaik, with Sahith Reddy Thummala, Pranav Nagothu, and Geethanjali Nagaboina"),
    ("Insight R-X", "Insight Rx"), ("H-C-Ps", "HCPs"), ("H-C-P", "HCP"), ("Dino-V-two", "DINO v2"), ("V-E-G-F A", "VEGF-A"),
    ("the Kem-B-L database", "the ChEMBL database"), ("S-G-L-T-two", "SGLT2"), ("C-M-S N-P-I registry", "CMS NPI Registry"),
    ("C-P-T ninety-two, two twenty-eight", "CPT ninety-two, two twenty-eight"), ("Fast-A-P-I", "FastAPI"), ("G-P-U", "GPU"),
]


def vv_text(spoken):
    for said, natural in VV_FORMS:
        spoken = spoken.replace(said, natural)
    return spoken


def split_at_sentences(audio, spoken_list, wl):
    """Cut a scene take into per-sentence clips at the pause between sentences, using Whisper word timings."""
    heard = [(re.sub(r"[^a-z0-9]", "", w.lower().replace("-", "")), s, e) for w, s, e in wl]
    heard = [h for h in heard if h[0]]
    script_words, owner = [], []
    for i, sent in enumerate(spoken_list):
        for w in words(sent):
            script_words.append(w)
            owner.append(i)
    sm = difflib.SequenceMatcher(None, script_words, [h[0] for h in heard], autojunk=False)
    first, last = {}, {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            i = owner[a + k]
            first.setdefault(i, heard[b + k][1])
            last[i] = heard[b + k][2]
    total = len(audio) / SR
    n = len(spoken_list)
    chars = [len(s) for s in spoken_list]
    cum = np.cumsum([0] + chars) / sum(chars) * total          # proportional fallback
    bounds = [0.0]
    for i in range(n - 1):
        end_i = last.get(i, cum[i + 1])
        start_next = first.get(i + 1, cum[i + 1])
        cut = (end_i + start_next) / 2 if start_next >= end_i else end_i
        bounds.append(min(max(cut, bounds[-1] + 0.2), total))
    bounds.append(total)
    return [audio[int(a * SR): int(b * SR)] for a, b in zip(bounds, bounds[1:])]


class Kokoro:
    def __init__(self):
        from kokoro import KPipeline
        self.pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    def say(self, spoken, caption, seed_base):
        audio = np.concatenate([a for _, _, a in self.pipe(spoken, voice=VOICE, speed=SPEED)])
        return {"audio": tidy(audio), "score": None, "rate": None, "heard": "", "attempt": 0, "ok": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=ENGINE, choices=["vibevoice", "chatterbox", "kokoro"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", help="regenerate only these scene ids (others keep their clips)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
    tpath = os.path.join(args.out, "timings.json")
    old = json.load(open(tpath)) if args.only and os.path.exists(tpath) else {}
    tts = {"chatterbox": Chatterbox, "kokoro": Kokoro, "vibevoice": VibeVoice}[args.engine]()
    timings, total, flagged = {}, 0.0, []
    for k, sc in enumerate(SCENES):
        if args.only and sc["id"] not in args.only and sc["id"] in old:
            timings[sc["id"]] = old[sc["id"]]
            total += old[sc["id"]]["min_len"]
            continue
        rows = []
        if args.engine == "vibevoice":
            pairs = sentences(sc)
            clips, score, heard = tts.scene([vv_text(sp) for _, sp in pairs], [c for c, _ in pairs])
            for i, ((caption, _), clip) in enumerate(zip(pairs, clips)):
                path = os.path.join(args.out, "audio", f"{sc['id']}_{i}.wav")
                sf.write(path, clip, SR)
                rows.append({"caption": caption, "file": path, "dur": round(len(clip) / SR, 3), "engine": "vibevoice",
                             "asr_match": round(score, 3), "heard": heard if i == 0 else ""})
            if score < MIN_MATCH:
                flagged.append((sc["id"], 0, round(score, 2), heard))
        for i, (caption, spoken) in enumerate([] if args.engine == "vibevoice" else sentences(sc)):
            take = tts.say(spoken, caption, seed_base=1000 * k + i)
            path = os.path.join(args.out, "audio", f"{sc['id']}_{i}.wav")
            sf.write(path, take["audio"], SR)
            dur = round(len(take["audio"]) / SR, 3)
            rows.append({"caption": caption, "file": path, "dur": dur, "engine": args.engine,
                         "asr_match": None if take["score"] is None else round(take["score"], 3), "heard": take["heard"]})
            if not take["ok"]:
                flagged.append((sc["id"], i, round(take["score"], 2), take["heard"]))
        speech = sum(r["dur"] for r in rows) + GAP_S * (len(rows) - 1)
        timings[sc["id"]] = {"sentences": rows, "speech": round(speech, 3), "min_len": round(speech + SCENE_PAD_S, 3)}
        total += speech + SCENE_PAD_S
        scores = [r["asr_match"] for r in rows if r["asr_match"] is not None]
        print(f"{sc['id']:11s} {speech:6.1f} s  ({len(rows)} sentences"
              + (f", ASR match min {min(scores):.2f})" if scores else ")"), flush=True)
    json.dump(timings, open(tpath, "w"), indent=1)
    print(f"narration total with pads: {total:.1f} s ({int(total // 60)}:{int(total % 60):02d})")
    for sid, i, score, heard in flagged:
        print(f"CHECK {sid}_{i}: best ASR match {score} -> {heard!r}")


if __name__ == "__main__":
    main()
