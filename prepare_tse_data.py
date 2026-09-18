#!/usr/bin/env python3
"""
Build a PS4 / REAL-PS4-style target speaker extraction (TSE) training set from a
single-speaker ASR manifest by simulating multi-speaker mixtures.

Input: one or more JSONL manifests, one utterance per line:
    {"audio_path": "/path/utt.wav", "speaker_id": "spk", "text": "transcript"}

Output (<out_root>/<name>/):
    mapping.csv                          mixture id -> wav path
    mixtures/<name>_mix_0000000.wav      16 kHz mono PCM16
    enrolment_speakers/<enroll_id>.wav   16 kHz mono PCM16 (a random crop, see --enroll_crop)
    TRAIN/<name>_meta.csv                one row per (mixture, target speaker)
    TRAIN/target_activity_segments.jsonl target speaker activity (VAD labels)
    prep_stats.json                      counts, drop reasons, settings

Each mixture holds 2-3 utterances from different speakers, placed with random
offsets (so they usually overlap) at random relative levels. For every speaker in
a mixture, the enrollment is a random crop (--enroll_crop, default 3-8 s) of a
*different* utterance of the same speaker, and the transcript is that speaker's
original utterance text.

Use in a PS4 training config:
    data:
      train_roots: [<out_root>]
      datasets: [<name>]
      dataset_langs: {<name>: th}   # optional: the meta CSV has a language column

Example (remap paths exported from another machine):
    python prepare_tse_data.py \\
        --manifest formatted_cleaned_1.7B.fixed.jsonl \\
        --out_root /data/tse --name ThaiSim --lang th \\
        --path_map /lustrefs/disk/project=/data/project \\
        --num_workers 32

Requirements: python >= 3.8, numpy, scipy, soundfile.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TAG_RE = re.compile(r"\[[^\]]*\]|<[^>]*>")   # [laughing], [emphasis], <noise>, ...
SPACE_RE = re.compile(r"\s+")
SAFE_RE = re.compile(r"[^0-9A-Za-z_.-]+")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def clean_text(text):
    return SPACE_RE.sub(" ", TAG_RE.sub(" ", text or "")).strip()


def safe(s):
    return SAFE_RE.sub("_", str(s)).strip("_") or "x"


def remap(path, path_maps):
    for old, new in path_maps:          # longest prefix first
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_audio(path, sr):
    """Read a file as float32 mono at sr Hz."""
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = math.gcd(int(file_sr), int(sr))
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def active_region(x, sr, floor_db=35.0, frame_ms=20.0):
    """(start, end) sample indices of the region above (peak frame level - floor_db)."""
    hop = max(1, int(sr * frame_ms / 1000))
    n = len(x) // hop
    if n == 0:
        return 0, len(x)
    power = (x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12
    db = 10 * np.log10(power)
    idx = np.flatnonzero(db > db.max() - floor_db)
    if idx.size == 0:
        return 0, len(x)
    return int(idx[0] * hop), int(min(len(x), (idx[-1] + 1) * hop))


def write_wav(path, x, sr):
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0.99:                      # avoid clipping in PCM16
        x = x * (0.99 / peak)
    sf.write(path, x, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# 1. scan the manifests
# ---------------------------------------------------------------------------

def probe(path):
    try:
        info = sf.info(path)
        return info.frames / info.samplerate, None
    except Exception as e:              # missing or unreadable
        return None, "missing_audio" if not os.path.exists(path) else f"unreadable: {type(e).__name__}"


def scan(args, path_maps, drops):
    rows = []
    excluded = set(args.exclude_speakers)
    for manifest in args.manifest:
        with open(manifest, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    drops["bad_json"] += 1
                    continue
                if not r.get("audio_path") or r.get("speaker_id") in (None, ""):
                    drops["missing_field"] += 1
                    continue
                if str(r["speaker_id"]) in excluded:
                    drops["excluded_speaker"] += 1
                    continue
                text = clean_text(r.get("text", ""))
                if len(text) < args.min_text_len:
                    drops["short_or_empty_text"] += 1
                    continue
                rows.append({"path": remap(r["audio_path"], path_maps),
                             "speaker": str(r["speaker_id"]), "text": text})
                if args.max_rows and len(rows) >= args.max_rows:
                    break
        if args.max_rows and len(rows) >= args.max_rows:
            break

    print(f"[scan] {len(rows)} rows with text; probing audio with {args.num_workers} threads ...")
    with ThreadPoolExecutor(max(1, args.num_workers)) as ex:
        results = list(ex.map(probe, [r["path"] for r in rows], chunksize=256))
    utts = []
    for r, (dur, err) in zip(rows, results):
        if err:
            drops[err] += 1
            continue
        r["dur"] = dur
        utts.append(r)
    print(f"[scan] {len(utts)} usable utterances, drops: {dict(drops)}")
    return utts


# ---------------------------------------------------------------------------
# 2. plan the mixtures (all random choices happen here, deterministically)
# ---------------------------------------------------------------------------

def enroll_id(name, utt):
    h = hashlib.md5(utt["path"].encode("utf-8")).hexdigest()[:8]
    return f"{name}_{safe(utt['speaker'])}_{safe(Path(utt['path']).stem)}_{h}"


def plan(args, utts):
    rng = np.random.default_rng(args.seed)
    by_spk = defaultdict(list)
    for u in utts:
        by_spk[u["speaker"]].append(u)

    sources, enrolls = {}, {}
    for spk, us in by_spk.items():
        src = [u for u in us if args.min_src_dur <= u["dur"] <= args.max_src_dur]
        enr = [u for u in us if u["dur"] >= args.min_enroll_dur]
        # need a source utterance plus a *different* enrollment utterance
        if src and enr and len({id(u) for u in src} | {id(u) for u in enr}) >= 2:
            sources[spk], enrolls[spk] = src, enr
    speakers = sorted(sources)
    skipped = sorted(set(by_spk) - set(speakers))
    print(f"[plan] {len(speakers)} eligible speakers; not eligible (need a {args.min_src_dur}-"
          f"{args.max_src_dur}s utterance and another >= {args.min_enroll_dur}s one): {skipped}")
    if len(speakers) < args.min_speakers:
        sys.exit(f"[plan] need at least {args.min_speakers} eligible speakers, found {len(speakers)}")

    n_src = sum(len(v) for v in sources.values())
    n_mix = args.num_mixtures or max(1, n_src // 2)
    if args.speaker_sampling == "uniform":
        weights = np.ones(len(speakers))
    else:                                # proportional to the number of source utterances
        weights = np.array([len(sources[s]) for s in speakers], dtype=float)
    weights /= weights.sum()

    mixtures = []
    for i in range(n_mix):
        k = int(rng.integers(args.min_speakers, min(args.max_speakers, len(speakers)) + 1))
        spks = rng.choice(speakers, size=k, replace=False, p=weights)
        parts, length = [], 0.0
        for spk in spks:
            src = sources[spk][rng.integers(len(sources[spk]))]
            cands = [u for u in enrolls[spk] if u is not src]
            if not cands:                # this speaker's only enrollment clip is src
                others = [u for u in sources[spk] if any(e is not u for e in enrolls[spk])]
                src = others[rng.integers(len(others))]
                cands = [u for u in enrolls[spk] if u is not src]
            enr = cands[rng.integers(len(cands))]
            crop = None
            if args.enroll_crop[1] > 0:  # (start fraction within the speech region, length in s)
                crop = (round(float(rng.random()), 4),
                        round(float(rng.uniform(args.enroll_crop[0], args.enroll_crop[1])), 2))
            eid = enroll_id(args.name, enr)
            if crop:
                eid += f"_c{int(crop[0] * 10000):04d}_{int(crop[1] * 100):04d}"
            if not parts:
                offset = 0.0
            else:                        # start anywhere inside the current mixture
                offset = float(rng.uniform(0.0, max(0.0, min(length, args.max_mix_dur - src["dur"]))))
            length = max(length, offset + src["dur"])
            parts.append({
                "speaker": spk, "path": src["path"], "text": src["text"], "offset": offset,
                "gain_db": float(rng.uniform(-args.gain_db, args.gain_db)),
                "enroll_id": eid, "enroll_path": enr["path"], "enroll_crop": crop,
            })
        mixtures.append({"id": f"{args.name}_mix_{i:07d}", "parts": parts})
    return mixtures, speakers


# ---------------------------------------------------------------------------
# 3. render audio (worker processes)
# ---------------------------------------------------------------------------

_CFG = {}


def _init_worker(cfg):
    _CFG.update(cfg)


def render_mixture(mix):
    sr = _CFG["sr"]
    try:
        placed, vad = [], {}
        for p in mix["parts"]:
            x = load_audio(p["path"], sr)
            s, e = active_region(x, sr)
            rms = float(np.sqrt(np.mean(x[s:e] ** 2))) + 1e-8
            x = x * (10 ** ((_CFG["level_dbfs"] + p["gain_db"]) / 20) / rms)
            start = int(round(p["offset"] * sr))
            placed.append((start, x))
            vad[p["speaker"]] = [[round((start + s) / sr, 3), round((start + e) / sr, 3)]]
        n = min(max(st + len(x) for st, x in placed), int(round(_CFG["max_mix_dur"] * sr)))
        mixture = np.zeros(n, dtype=np.float32)
        for st, x in placed:
            seg = x[: max(0, n - st)]
            mixture[st: st + len(seg)] += seg
        out = os.path.join(_CFG["mix_dir"], mix["id"] + ".wav")
        write_wav(out, mixture, sr)
        return {"id": mix["id"], "path": out, "dur": round(n / sr, 3), "vad": vad, "error": None}
    except Exception as e:
        return {"id": mix["id"], "error": f"{type(e).__name__}: {e}"}


def render_enrollment(item):
    eid, path, crop = item
    sr = _CFG["sr"]
    try:
        x = load_audio(path, sr)
        s, e = active_region(x, sr)          # speech region (skips leading/trailing silence)
        if crop:                             # random window inside the speech region
            frac, length = crop
            n = int(length * sr)
            if e - s > n:
                s += int(frac * (e - s - n))
            x = x[s: min(e, s + n)]
        else:                                # from the first speech, cut to max_enroll_dur
            x = x[s: s + int(_CFG["max_enroll_dur"] * sr)]
        write_wav(os.path.join(_CFG["enroll_dir"], eid + ".wav"), x, sr)
        return eid, None
    except Exception as e:
        return eid, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, help="input JSONL manifest(s)")
    ap.add_argument("--out_root", required=True, help="output root (use as data.train_roots)")
    ap.add_argument("--name", default="ThaiSim", help="dataset name (use in data.datasets)")
    ap.add_argument("--lang", default="th", help="language code written to the meta CSV (Whisper code)")
    ap.add_argument("--path_map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite an audio_path prefix; repeatable")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--min_src_dur", type=float, default=1.0, help="min length of a mixed utterance (s)")
    ap.add_argument("--max_src_dur", type=float, default=8.0, help="max length of a mixed utterance (s)")
    ap.add_argument("--min_enroll_dur", type=float, default=2.0, help="min enrollment length (s)")
    ap.add_argument("--max_enroll_dur", type=float, default=10.0,
                    help="without cropping, enrollments are cut to this (s)")
    ap.add_argument("--enroll_crop", type=float, nargs=2, default=[3.0, 8.0], metavar=("MIN", "MAX"),
                    help="random enrollment crop length range (s), taken from a random position in the "
                         "clip's speech region, one crop per meta row; '0 0' = no cropping")
    ap.add_argument("--max_mix_dur", type=float, default=10.0,
                    help="max mixture length (s); keep <= data.max_mix_len of the training config")
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=3)
    ap.add_argument("--num_mixtures", type=int, default=0,
                    help="number of mixtures (default: half the number of usable source utterances)")
    ap.add_argument("--gain_db", type=float, default=5.0, help="relative speaker levels drawn from +-gain_db")
    ap.add_argument("--level_dbfs", type=float, default=-25.0, help="reference speech level before gains")
    ap.add_argument("--speaker_sampling", choices=["uniform", "proportional"], default="uniform",
                    help="pick speakers uniformly (balances large/small speakers) or by utterance count")
    ap.add_argument("--exclude_speakers", nargs="*", default=[],
                    help="speaker_ids to skip (e.g. ids that cover several people)")
    ap.add_argument("--min_text_len", type=int, default=2, help="drop transcripts shorter than this (chars)")
    ap.add_argument("--max_rows", type=int, default=0, help="only read this many manifest rows (testing)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="scan and plan only; write nothing")
    ap.add_argument("--overwrite", action="store_true", help="allow writing into an existing non-empty dataset dir")
    args = ap.parse_args()

    if args.max_src_dur > args.max_mix_dur:
        ap.error("--max_src_dur must be <= --max_mix_dur")
    lo, hi = args.enroll_crop
    if hi > 0 and not 0 < lo <= hi <= args.max_enroll_dur:
        ap.error("--enroll_crop needs 0 < MIN <= MAX <= --max_enroll_dur (or '0 0' to disable)")
    if not 1 <= args.min_speakers <= args.max_speakers:
        ap.error("need 1 <= --min_speakers <= --max_speakers")
    path_maps = []
    for m in args.path_map:
        if "=" not in m:
            ap.error(f"--path_map must be OLD=NEW, got {m!r}")
        old, new = m.split("=", 1)
        path_maps.append((old, new))
    path_maps.sort(key=lambda p: len(p[0]), reverse=True)

    ds_dir = Path(args.out_root).resolve() / args.name
    if not args.dry_run and ds_dir.exists() and any(ds_dir.iterdir()) and not args.overwrite:
        sys.exit(f"{ds_dir} is not empty; pass --overwrite to write into it")

    t0 = time.time()
    drops = Counter()
    utts = scan(args, path_maps, drops)
    mixtures, speakers = plan(args, utts)
    n_rows = sum(len(m["parts"]) for m in mixtures)
    print(f"[plan] {len(mixtures)} mixtures, {n_rows} meta rows")
    if args.dry_run:
        return

    mix_dir, enroll_dir, train_dir = ds_dir / "mixtures", ds_dir / "enrolment_speakers", ds_dir / "TRAIN"
    for d in (mix_dir, enroll_dir, train_dir):
        d.mkdir(parents=True, exist_ok=True)
    cfg = {"sr": args.sample_rate, "level_dbfs": args.level_dbfs, "max_mix_dur": args.max_mix_dur,
           "max_enroll_dur": args.max_enroll_dur, "mix_dir": str(mix_dir), "enroll_dir": str(enroll_dir)}

    enroll_items = sorted({(p["enroll_id"], p["enroll_path"], p["enroll_crop"]) for m in mixtures for p in m["parts"]},
                          key=lambda t: t[0])
    with Pool(max(1, args.num_workers), initializer=_init_worker, initargs=(cfg,)) as pool:
        print(f"[write] {len(enroll_items)} enrollment clips ...")
        bad_enroll = {eid: err for eid, err in pool.imap_unordered(render_enrollment, enroll_items, chunksize=16) if err}
        print(f"[write] {len(mixtures)} mixtures ...")
        rendered = list(pool.imap(render_mixture, mixtures, chunksize=8))

    mix_ok = 0
    total_sec = 0.0
    with open(ds_dir / "mapping.csv", "w", newline="", encoding="utf-8") as fm, \
         open(train_dir / f"{args.name}_meta.csv", "w", newline="", encoding="utf-8") as fc, \
         open(train_dir / "target_activity_segments.jsonl", "w", encoding="utf-8") as fv:
        mapping = csv.writer(fm)
        mapping.writerow(["utterance", "path"])
        meta = csv.writer(fc)
        meta.writerow(["mixture_utterance", "enrolment_speakers_utterance", "source", "language",
                       "total_number_of_speaker", "speaker", "mixture_duration",
                       "ground_truth_transcript", "target_utterance_path"])
        for mix, res in zip(mixtures, rendered):
            if res["error"]:
                drops["mixture_render_failed"] += 1
                print(f"[warn] {mix['id']}: {res['error']}")
                continue
            parts = [p for p in mix["parts"] if p["enroll_id"] not in bad_enroll]
            if not parts:
                drops["mixture_without_enrollment"] += 1
                continue
            mapping.writerow([mix["id"], res["path"]])
            mix_ok += 1
            total_sec += res["dur"]
            for p in parts:
                meta.writerow([mix["id"], p["enroll_id"], args.name, args.lang, len(mix["parts"]),
                               p["speaker"], res["dur"], p["text"], p["path"]])
                fv.write(json.dumps({"mixture_utterance": mix["id"], "speaker": p["speaker"],
                                     "segments": res["vad"][p["speaker"]],
                                     "mixture_duration": res["dur"], "source": args.name},
                                    ensure_ascii=False) + "\n")
            drops["meta_row_without_enrollment"] += len(mix["parts"]) - len(parts)

    stats = {
        "mixtures": mix_ok, "hours": round(total_sec / 3600, 3), "speakers": len(speakers),
        "enrollment_clips": len(enroll_items) - len(bad_enroll), "drops": dict(drops),
        "settings": vars(args), "seconds": round(time.time() - t0, 1),
    }
    (ds_dir / "prep_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {mix_ok} mixtures ({stats['hours']} h), {stats['enrollment_clips']} enrollments -> {ds_dir}")
    print(f"[done] drops: {dict(drops)}")


if __name__ == "__main__":
    main()
