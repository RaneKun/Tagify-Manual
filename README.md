# 🎵 Tagify (Manual Edition)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A suite of standalone Python scripts that tag `.ogg` music libraries with rich metadata — Spotify info, lyrics, genre, and BPM/mood — written straight into the file's Vorbis comment tags. Each script is its own mini-tool with its own conda environment, so you only set up (and run) the ones you actually need.

> This is the **Manual Edition** — each tagger is run by hand, one at a time. An automated/pipeline version that chains all of them together is in [here]().

## ✨ Modules

- **🟢 Spotify Tagger** — Looks up each track on Spotify and writes title, artist, album, release year, and embedded album art (640×640 JPEG). Tracks Spotify's ~700 requests/day soft limit in a local JSON file and warns (but doesn't stop) if you go over.
- **🌐📝 Lyrics Tagger (Online)** — Pulls plain lyrics from LRCLib, NetEase, and Musixmatch (in that order) and writes them to the `LYRICS` tag, stripped of timestamps and metadata lines.
- **🅾️📝 Lyrics Tagger (Offline)** — Transcribes lyrics straight from the audio using a local pipeline (Demucs vocal separation → DeepFilterNet3 denoising → Faster-Whisper large-v3). No internet needed, but it does need a decent Nvidia GPU.
- **🌐♪ Genre Tagger (Online)** — Queries Last.fm, iTunes, and MusicBrainz, merges and scores their tags, and writes up to 3 genres per track. Caches artist-level lookups so repeat artists don't cost extra API calls.
- **🅾️♪ Genre Tagger (Offline)** — Uses MusicNN (TensorFlow) to sniff out up to 3 genres per track without any internet connection.
- **🎧 BPM + Mood Tagger** — Cross-validates BPM using both aubio and librosa, then assigns one of 23 mood prototypes based on tempo, loudness, and spectral features normalized against your whole library.

## 📋 Requirements

- Python 3.10 or 3.11 (a separate version is used per module — see below)
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) — every module runs in its own isolated conda environment -> ensure "Add Anaconda to my PATH environment variable" is checked ✓
- System-wide [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) on PATH (required by the offline Lyrics and Genre taggers) -> click [here](https://phoenixnap.com/kb/ffmpeg-windows) to know how to do that 
- An NVIDIA GPU + driver (optional, but strongly recommended for the offline Lyrics and Genre taggers — both fall back to CPU automatically if no GPU is found)
- A free [Spotify Developer](https://developer.spotify.com/) client ID/secret (for the Spotify Tagger only)
- A free [last.fm](https://www.last.fm/api/accounts) api key (for the Online Genre Tagger only)

No dependencies need to be installed by hand — `setup_venvs.bat` handles all of that per module.

## 🎯 How It Works

### 1. Run the setup script
Double-click `setup_venvs.bat`. It checks for conda, ffmpeg, and a GPU, then builds one conda environment per module inside `venvs/` (skipping any module whose script isn't present in `scripts/`, and skipping environments that are already set up). This step can take a while — the offline modules pull in TensorFlow or PyTorch with CUDA support.

### 2. Run a module
Each module is launched from its own environment:
```
conda run -p "venvs\spotify_tagger" python "scripts\spotify_tagger.py"
```
or activate the environment first:
```
conda activate "venvs\spotify_tagger"
python "scripts\spotify_tagger.py"
```

### 3. Point it at your library
On first run, each script asks for:
- An **input folder** of `.ogg` files (read-only — your originals are never touched)
- An **output folder** for the finished, tagged copies (defaults to `outputs/`)
- Spotify credentials, if running the Spotify Tagger
- last.fm credentials, if running the Online Genre Tagger

These choices are remembered in `configs/` for next time — just press Enter to reuse them.

### 4. Let it run
Every file is copied to `temp/`, tagged there, then moved to `outputs/`. Progress is saved after every single file to a checkpoint JSON in `logs/`, so you can safely stop and resume — already-tagged files are always skipped on the next run.

## ⚙️ Configuration

Each script has a small `CONFIG` block near the top for tweaking behavior, for example:

```python
# online_genre_tagger.py / local_genre_tagger.py
TOP_N_GENRES = 3   # max genre tags written per track
```

```python
# spotify_tagger.py
DAILY_CALL_LIMIT = 700   # soft warning limit, not a hard stop
```

Input/output folders and credentials are set interactively on first run and saved to `configs/<script_name>.json` — delete that file to be prompted again.

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **`setup_venvs.bat` says conda not found** | Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then open a new terminal so PATH refreshes |
| **Offline Lyrics/Genre tagger fails at runtime** | Make sure system-wide `ffmpeg` is on PATH — these two modules use the system copy, not a conda-installed one |
| **No GPU detected** | The offline modules still work on CPU, just slower — check your NVIDIA driver is current if you expected GPU support |
| **Spotify Tagger shows a rate-limit warning** | You've passed the ~700 calls/day soft budget — it's safe to keep going, but you may start seeing 429 errors; re-running later in the day resumes from your checkpoint |
| **A file keeps showing up as "missed"** | It has no match on the given source (Spotify/lyrics/genre) — it's still copied to `outputs/`, just without that tag |
| **Want to re-tag everything from scratch** | Delete the relevant checkpoint JSON in `logs/` |

## 📝 Notes

- Every module treats your input folder as **read-only** — files are copied to `temp/` before anything is written, and only the finished copy lands in `outputs/`.
- Filenames for the Spotify Tagger must follow `Artist - Title.ogg`; anything without a `" - "` separator is logged and skipped.
- All modules log everything (DEBUG level) to their own file under `logs/`, even when the console only shows a summary.

## 🙏 Credits

**Made with ♥ by Rane Kun**
