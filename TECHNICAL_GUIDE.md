# 🛠️ Tagify — Technical Guide (Manual Edition)

This is the deep-dive doc. The main [README](./README.md) tells people what Tagify does; this file explains **how**, **why it's built this way**, and **what to do when it breaks**.

---
📁 Project Stucture

```Tagify [MANUAL]/
├── scripts/
│   ├── bpm_mood_tagger.py
│   ├── local_genre_tagger.py
│   ├── local_lyrics_tagger.py
│   ├── online_genre_tagger.py
│   ├── online_lyrics_tagger.py
│   └── spotify_tagger.py
├── venvs/
│   ├── bpm_mood_tagger/
│   ├── local_genre_tagger/
│   ├── local_lyrics_tagger/
│   ├── online_genre_tagger/
│   ├── online_lyrics_tagger/
│   └── spotify_tagger/
└── setup_venvs.bat
```

---

## 1. Core Design Philosophy

Every module in Tagify — regardless of what it tags — follows the same four rules:

1. **Originals are sacred.** `INPUT_DIR` is opened read-only. Every file is `shutil.copy2`'d into `temp/` before anything touches it. Nothing is ever written back to the input folder.
2. **Checkpoint after every single file, not every batch.** Progress JSON is rewritten to disk after each file completes — not buffered, not saved every N files. If the process dies mid-run (crash, power cut, Ctrl+C), at most one file's work is lost.
3. **Soft failures flow through; hard failures halt.** A "no match found" or "unparseable filename" is not treated as broken — the file still gets copied to `outputs/` (just without that tag) and the run continues. A *real* error — network down, disk full, can't write a tag — stops the whole script immediately so you can fix the actual problem instead of the script burning through the rest of the run half-broken.
4. **Config is remembered, not hardcoded.** Nothing requires editing the script before first run. Input/output folders (and credentials, for Spotify and last.fm) are asked for interactively once and cached to `configs/<script_name>.json`; pressing Enter on future runs reuses the saved value.

### Shared folder layout

```
Tagify/
├── scripts/            → the 6 tagger .py files
├── venvs/               → one conda env per module (created by setup_venvs.bat)
├── configs/              → saved input/output paths + credentials, per script
├── logs/                 → DEBUG-level .log, checkpoint .json, and any caches, per script
├── temp/                 → working copies while a file is actively being tagged
├── failed/<reason>/      → catalogued copies of anything that hit a soft failure
└── outputs/               → the finished, tagged files (default output folder)
```

`temp/`, `failed/`, and `outputs/` are shared conceptually across modules but not physically namespaced — if you run multiple modules against the same input folder, each pass is additive (a file tagged by Spotify Tagger, then run through Genre Tagger, ends up with both sets of tags, since each module reads/writes different Vorbis fields).

![Folder and data flow diagram](./images/folder-data-flow.png)
*Placeholder — arrows showing input → temp → (tag) → output/failed.*

---

## 2. Environment Architecture

Each module gets its own isolated conda environment because their dependency trees actively conflict with each other (different TensorFlow/PyTorch/CUDA stacks). `setup_venvs.bat` builds all of them from one entry point.

| Module | Python | Device | Key deps | Notes |
|---|---|---|---|---|
| `spotify_tagger` | 3.11 | CPU | spotipy, requests, mutagen | Lightest env, no ML |
| `online_genre_tagger` | 3.11 | CPU | requests, mutagen, pylast, musicbrainzngs | No GPU needed |
| `online_lyrics_tagger` | 3.11 | CPU | requests, mutagen, syncedlyrics | No GPU needed |
| `bpm_mood_tagger` | 3.10 | CPU | aubio, librosa, numpy (conda-forge) | aubio has no Windows pip wheel — must come from conda-forge |
| `local_genre_tagger` | 3.10 | GPU | musicnn on TensorFlow 2.10, conda-forge cudatoolkit 11.2 + cudnn 8.1 | TF 2.10 is the *last* TF release with native Windows GPU support — this env is pinned there on purpose |
| `local_lyrics_tagger` | 3.11 | GPU | Demucs, DeepFilterNet3, faster-whisper on official PyTorch cu121 wheels | Heaviest env, ~3 GB Whisper model downloaded on first run |

### Gotchas baked into `setup_venvs.bat` (worth knowing before you debug them yourself)

- **`musicnn` is installed with `--no-deps`.** Its published metadata pins `numpy<1.17`, which conflicts with the numpy version TensorFlow 2.10 actually needs. Skipping dependency resolution here is intentional, not an oversight.
- **Both GPU-based offline modules use *system* ffmpeg, not conda-forge's.** conda-forge's ffmpeg build was seen crashing with `STATUS_DLL_NOT_FOUND` / `0xC0000135` on at least one real machine. System ffmpeg on PATH is required for `local_genre_tagger` and `local_lyrics_tagger` specifically.
- **`local_genre_tagger`'s GPU check commonly fails on a missing `zlibwapi.dll`** — that DLL is a cuDNN 8.1-on-Windows requirement, not a Tagify bug. MusicNN still runs on CPU if it's missing, just slower.
- **There's a known, deliberately-unfixed comment/code mismatch.** An old comment block in `setup_venvs.bat` references a filename override for `bpm_mood_tagger` (something about a `+` in the filename). That override doesn't exist anywhere in the actual script — every module resolves its script file the same generic way. It's left alone because neither side (comment vs. code) was worth guessing at changing without knowing which one was "correct" — flagged in the batch file's own header comment instead.
- **`conda run` is deliberately avoided for the live-progress bar.** There's a documented Windows bug (`conda/conda#9700`) where `conda run` clears terminal output mid-command. The setup script invokes conda's base `python.exe` directly instead, with a small embedded Python script driving a `tqdm` progress bar.

![Environment setup flow](./images/setup-venvs-flow.png)
*Placeholder — flowchart of setup_venvs.bat: check conda → per-module create env → install deps → verify imports → GPU check.*

---

## 3. Module-by-Module Internals

### 🟢 Spotify Tagger

- **Filename contract:** expects `Artist - Title.ogg`. Files without a `" - "` separator are logged and routed to `failed/bad_filename/` — this is a soft failure, not a halt.
- **Search strategy:** for each file, tries three query shapes in order — a strict `artist:"X" track:"Y"` query, an unquoted field query, then a plain free-text query — stopping at the first hit.
- **Title cleaning:** strips `(feat. X)` / `[ft. X]` / `(with X)` annotations from the matched title via regex before writing the `title` tag.
- **Rate limiting:** tracks a soft daily budget of 700 calls in `logs/spotify_daily_stats.json`, reset at local midnight. Going over the budget prints a warning but does **not** stop the run — you may start seeing HTTP 429s from Spotify if you push past it.
- **429 handling:** on an actual 429 response, it respects Spotify's `Retry-After` header (plus a few seconds of random jitter) before retrying that query.
- **Server errors (5xx)** and **connection-level failures** are both treated as network errors, which halts the whole pipeline for manual review rather than silently continuing.
- **Album art:** downloads the largest available cover image and embeds it as a 640×640 JPEG in the `metadata_block_picture` tag (base64-encoded FLAC Picture block, the standard Vorbis Comment convention).
- **Politeness delay:** ~1.8s ± 0.6s jitter between files, skipped only for filename-skip cases (nothing was actually requested from Spotify for those).

### 🌐📝 Lyrics Tagger — Online

- Tries **LRCLib → NetEase → Musixmatch**, in that order, stopping at the first source that returns lyrics.
- LRCLib is queried directly via its REST API; NetEase and Musixmatch go through the `syncedlyrics` library in plain (non-synced) mode.
- All returned lyrics are stripped of timestamp markup (e.g. `[00:12.34]`) and metadata lines (e.g. `作词 :`, `作曲 :`, `Lyrics:`) before being written, so only clean lyric text lands in the `LYRICS` tag.
- **Instrumental detection** triggers on a title containing a known instrumental keyword (e.g. "bgm", "(inst.)") or LRCLib explicitly flagging the track as instrumental — either short-circuits straight to "instrumental" without trying the remaining sources.
- If literally every source fails due to connectivity issues, the file goes to `failed/network_error/` and is retried on the next run — it's never marked "done" in that case.

### 📝 Lyrics Tagger — Offline

The most involved module in the suite. Six-stage pipeline per file:

1. **Demucs (`htdemucs_6s`, 2 shifts)** — 6-stem source separation to pull out an isolated vocal stem with minimal instrument bleed.
2. **ffmpeg preprocessing** — `highpass=100Hz` (removes sub-bass rumble) + `loudnorm=I=-16` (normalizes level) on the vocal stem.
3. **Resample to 48kHz → DeepFilterNet3** — neural denoising pass, run at the sample rate DeepFilterNet3 expects.
4. **Faster-Whisper (`large-v3`)** — `word_timestamps=True`, VAD disabled (Demucs already isolated the vocals), `beam_size=20`, `patience=3.0`, six-temperature fallback ladder (`0.0` → `1.0`) for difficult segments.
5. **Raw-audio fallback** — if the Demucs-stem transcription's average log-probability is below `FALLBACK_LOGPROB` (`-0.40`), the *original* (non-separated) audio is transcribed too, and whichever result scored a better log-probability wins.
6. **Multi-gate instrumental filtering** — a track is declared instrumental (no error, just "no lyrics found") if *any* of: vocal RMS energy is near-silent (< `0.004`), average log-probability is below the hard-abort threshold (`-0.60`), average no-speech probability exceeds `0.75`, or the total kept word count is under `15`.

Segment-level filtering also happens during transcription: any segment with `no_speech_prob > 0.85` or `avg_logprob < -2.00` is dropped before the instrumental gates even run.

![Offline lyrics pipeline stages](./images/lyrics-offline-pipeline.png)
*Placeholder — Demucs → ffmpeg → DeepFilterNet3 → Whisper → gates, as a horizontal pipeline diagram.*

**Note:** on Windows, this script does a one-time `os.add_dll_directory()` fix pointing at PyTorch's bundled DLL folder — a workaround for a common cuDNN-loading issue with PyTorch on Windows, applied before `torch` is imported.

### 🌐 Genre Tagger — Online

- Queries **Last.fm, iTunes, and MusicBrainz** for every track (all three are "on" for every song, not tried-in-sequence like the lyrics modules) and merges/scores their results into up to `TOP_N_GENRES` (default 3) tags.
- Last.fm contributes both track-level (`track.getTopTags`) and artist-level (`artist.getTopTags`) tags. iTunes brings Apple's own fixed taxonomy (K-Pop, J-Pop, Anime, World, Latino, Reggaeton y Flow, etc.) as a clean cross-check against Last.fm's messier folksonomy tags. MusicBrainz is the fallback most useful for small/independent artists with no Last.fm scrobble history.
- **MusicBrainz enforces 1 request/second** on its own end — it's by far the slowest of the three sources and the main per-file time cost in this module.
- **Artist-level lookups are cached** to `logs/genre_online_artist_cache.json`, so re-running the script (or processing a library with many tracks per artist) never re-spends a network call on an artist you've already looked up. This is the single biggest speed lever in this module — unique-artist count is almost always far smaller than track count.

### ♪ Genre Tagger — Offline

- Uses **MusicNN** (TensorFlow) to classify up to 3 genres per track, fully offline.
- No Last.fm/iTunes/MusicBrainz querying at all — purely audio-feature-based inference from the trained model.
- Same checkpoint/retry semantics as the other offline modules: `error` entries are retried automatically, `done` entries are always skipped.

### 🎧 BPM + Mood Tagger

- **BPM** is cross-validated using **both aubio and librosa** independently for extra confidence before a final value is written.
- **Mood** uses a two-pass, library-relative classifier:
  1. **Pass 1** scans the entire input folder once to compute the *global* mean and standard deviation for tempo, RMS loudness, spectral centroid, and zero-crossing rate across your whole library. This baseline is cached to `logs/feature_stats.json`.
  2. **Pass 2** Z-scores each individual song's features against that library-wide baseline (not fixed absolute thresholds) — so a 140 BPM track only reads as "fast" relative to *your* collection's own average, not some universal cutoff.
  3. Each song is then matched to whichever of **23 mood prototypes** sits closest to its normalized feature profile, using a weighted Euclidean distance.
- Because pass 1 requires scanning the whole folder up front, this module's startup is slower than the others on a first run — but the cached `feature_stats.json` is reused on subsequent runs unless the input folder changes significantly.

![Mood classification: Z-score + prototype matching](./images/mood-classifier-diagram.png)
*Placeholder — scatter plot or radar chart illustrating the 23 mood prototypes and Z-score normalization.*

---

## 4. Checkpointing & Error Semantics (all modules)

Every module's checkpoint JSON uses the same four buckets: `done`, `miss`, `skip`, `error`. The exact meaning of each varies slightly by module (see each script's own docstring header for specifics), but the pattern is consistent:

- **`done`** — always skipped on the next run, no matter what.
- **`miss` / `skip`** — informational for most modules; **never** cause a file to be skipped on re-run (the notable exception is Spotify Tagger, where "no match" and "bad filename" outcomes *are* recorded but still get retried since they're not in `done`).
- **`error`** — retried automatically on the next run for the offline modules; also **halts the entire pipeline immediately** the moment it happens, rather than continuing on to the next file.

This asymmetry is deliberate: "Spotify has no match for this song" is expected and shouldn't stop 8,000 other files from processing. "The network just went down" or "disk write failed" means every subsequent file is *also* going to fail the same way, so continuing would just burn time and API budget for nothing — better to stop, fix the real problem, and re-run (already-done files are skipped automatically either way).

A file that hits a soft failure (miss/skip) is still copied through to `outputs/` — you get every file back in the output folder either way, just with fewer tags on the ones that couldn't be matched.

---

## 5. Configuration Reference

Each module's `CONFIG` block sits near the top of its script. The values worth knowing about:

| Setting | Module | Default | Effect |
|---|---|---|---|
| `DAILY_CALL_LIMIT` | Spotify Tagger | `700` | Soft warning threshold for Spotify API calls/day |
| `BASE_DELAY` / `JITTER` | Spotify Tagger | `1.8` / `0.6` | Seconds between requests (politeness delay) |
| `TOP_N_GENRES` | Both Genre Taggers | `3` | Max genre tags written per track |
| `DEMUCS_MODEL` / `DEMUCS_SHIFTS` | Lyrics Offline | `htdemucs_6s` / `2` | Separation model + shift-averaging passes |
| `WHISPER_MODEL` / `WHISPER_BEAM` | Lyrics Offline | `large-v3` / `20` | Transcription model + beam search width |
| `HARD_ABORT_LOGPROB` | Lyrics Offline | `-0.60` | Below this, a track is declared instrumental outright |
| `MIN_TOTAL_WORDS` | Lyrics Offline | `15` | Minimum kept-segment word count to avoid the instrumental gate |

Input/output folders and (for Spotify Tagger) credentials are **not** in the `CONFIG` block — they're prompted for interactively on first run and saved to `configs/<script_name>.json`. Delete that file to be re-prompted from scratch.

---

## 6. Further Troubleshooting

Beyond what's in the main README:

| Symptom | Likely cause | Fix |
|---|---|---|
| `local_genre_tagger` reports `GPU_NO` during setup | Missing `zlibwapi.dll`, a cuDNN 8.1-on-Windows dependency | Install per [NVIDIA's cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html), or accept CPU fallback (slower, still works) |
| `local_lyrics_tagger`/`local_genre_tagger` crash with `STATUS_DLL_NOT_FOUND` around ffmpeg calls | conda-forge's ffmpeg build, not system ffmpeg | Confirm `where ffmpeg` resolves to a system install, not a conda one, and that it's earlier on PATH |
| Whisper output looks like nonsense/hallucinated text | Very sparse or heavily processed vocals | This is what `is_hallucination()` and the segment-level `no_speech_prob`/`avg_logprob` gates exist to catch — check `logs/lyrics_offline.log` for `[DROP-halluc-seg]` lines to confirm it was actually filtered |
| A whole run halts on file #1 of a large batch | An `error`-tier failure (not miss/skip) | Check the relevant module's `.log` in `logs/` for the actual exception — this is intentional pipeline-halt behavior, not a hang |
| `local_genre_tagger` env creation fails resolving numpy conflicts | musicnn's stale `numpy<1.17` pin | Already handled via `--no-deps` in `setup_venvs.bat` — if you're installing manually, replicate that flag |
| MusicBrainz-sourced genres take much longer than Last.fm/iTunes | Expected — MusicBrainz enforces 1 req/sec on their end | Not fixable client-side without violating their rate limit; the artist-level cache in `logs/genre_online_artist_cache.json` is the main mitigation for repeat runs |
| A `bpm_mood_tagger` run seems to "hang" at the very start | Pass 1 is scanning the whole library to build `feature_stats.json` | Expected on first run against a large folder — subsequent runs reuse the cached stats file |

---

## 7. Screenshots / Visuals

![Spotify Tagger console output example](./images/spotify-tagger-console.png)
*Placeholder — a real terminal capture of a Spotify Tagger run, showing the banner + per-file lines.*

![Offline Lyrics Tagger console output example](./images/lyrics-offline-console.png)
*Placeholder — a real terminal capture showing the pipeline stage log lines ([1/6]…[6/6]).*

![Example tagged file metadata](./images/example-tagged-metadata.png)
*Placeholder — a screenshot of a finished .ogg file's tags (e.g. in MusicBee, foobar2000, or Mp3tag) showing all the fields Tagify wrote.*

---

*This guide covers the Manual Edition only. Once the automated/pipeline edition exists, its own internals will get a separate write-up rather than folding into this one.*
