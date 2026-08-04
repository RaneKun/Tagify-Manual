"""
🎧 BPM + Mood Tagger — Offline Edition
─────────────────────────────────────────────────────────────────────────────
Extracts BPM (beats per minute) and a single mood tag using pure audio
analysis, then writes the "bpm" and "mood" Vorbis comment tags into your
.ogg files. Fully offline — no internet connection needed.

Mood detection uses a soft-distance classifier with global normalisation:
  • First pass  — scans your entire library once to compute the global
    mean & std for tempo, RMS, spectral centroid, and zero-crossing rate.
  • Normalisation — each song's features are Z-scored against your
    *entire* collection, so a 140 BPM track only counts as "fast" relative
    to your library's own average.
  • Prototype matching — 23 mood prototypes are defined; each song is
    assigned whichever mood's normalised profile sits closest to it
    (weighted Euclidean distance).

BPM uses dual-engine cross-validation (aubio + librosa) for extra confidence.

Checkpoint behaviour:
  ▸ Files listed under "done" are ALWAYS skipped on the next run.
  ▸ Files under "error" are NOT skipped — they'll be retried automatically.
  ▸ "miss" and "skip" are reserved for future use (kept for consistency).
  ▸ Delete or clear the checkpoint file to reprocess everything from scratch.

Output locations:
  ▸ Log file   → logs/bpm_mood.log                (everything, DEBUG level)
  ▸ Checkpoint → logs/bpm_mood_checkpoint.json
  ▸ Stats cache → logs/feature_stats.json          (library-wide Z-score baseline)

Before first run: edit the CONFIG block below to point at your music folder.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import sys
import json
import tempfile
import shutil
import subprocess
import warnings
import logging
import statistics
import time
import unicodedata
import textwrap
from typing import Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── TensorFlow not needed here, but silence warnings for a clean console ────
warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOOK & FEEL  ─  colors + tiny console helpers                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

THEME        = YELLOW   # ← this script's signature color
SCRIPT_EMOJI = "🎧"      # BPM + mood tagger's signature icon

def c_green(text):  return f"{GREEN}{text}{RESET}"
def c_yellow(text): return f"{YELLOW}{text}{RESET}"
def c_red(text):    return f"{RED}{text}{RESET}"
def c_theme(text):  return f"{THEME}{text}{RESET}"
def c_dim(text):    return f"{DIM}{text}{RESET}"
def c_bold(text):   return f"{BOLD}{text}{RESET}"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def _visible_width(text: str) -> int:
    """
    Approximate on-screen column width of a string, ignoring ANSI color
    codes. Emoji from the default-emoji-presentation blocks (🎧, 🌐, 📝, ...)
    always render as 2 columns. Text-presentation symbols (✓, ✗, arrows, the
    misc-symbols block, etc.) render as 1 column UNLESS immediately followed
    by U+FE0F (the emoji variation selector), which forces 2-column emoji
    rendering. Getting this distinction right is what keeps box borders
    closing cleanly regardless of which symbols/emoji end up in a line.
    """
    text = _ANSI_RE.sub("", text)
    chars = list(text)
    n = len(chars)
    width = 0
    i = 0
    while i < n:
        ch = chars[i]
        cp = ord(ch)

        if cp == 0x200D:            # zero-width joiner
            i += 1
            continue
        if cp == 0xFE0F:            # stray variation selector (no base found)
            i += 1
            continue
        if unicodedata.combining(ch):
            i += 1
            continue

        next_is_vs16 = (i + 1 < n) and ord(chars[i + 1]) == 0xFE0F

        # Default-emoji-presentation blocks are always 2 columns wide.
        if (0x1F300 <= cp <= 0x1FAFF) or (0x1F1E6 <= cp <= 0x1F1FF):
            width += 2
            i += 2 if next_is_vs16 else 1
            continue

        # Text-presentation symbols (✓ ✗ arrows ★ ☆ etc.) are narrow UNLESS
        # explicitly forced into emoji presentation by a following U+FE0F.
        if next_is_vs16:
            width += 2
            i += 2
            continue

        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        i += 1

    return width

# ── Helper to split ANSI codes from text ──────────────────────────────
def _split_ansi_codes(text: str):
    """Return (clean_text, list_of_ansi_sequences_in_order)."""
    ansi_re = re.compile(r'\033\[[0-9;]*[a-zA-Z]')
    parts = []
    last_end = 0
    for m in ansi_re.finditer(text):
        start, end = m.span()
        if start > last_end:
            parts.append(('text', text[last_end:start]))
        parts.append(('ansi', m.group()))
        last_end = end
    if last_end < len(text):
        parts.append(('text', text[last_end:]))
    return parts

def _wrap_line_with_ansi(line: str, max_width: int) -> list[str]:
    """
    Wrap a single line into multiple lines that each fit within max_width,
    preserving ANSI color codes and re‑applying them to each wrapped line.
    """
    parts = _split_ansi_codes(line)
    ansi_sequences = [p[1] for p in parts if p[0] == 'ansi']
    clean_text = ''.join(p[1] for p in parts if p[0] == 'text')
    if not clean_text:
        return [line]

    # Collect all ANSI codes that appear before any text (leading prefix)
    leading_ansi = []
    for part in parts:
        if part[0] == 'ansi':
            leading_ansi.append(part[1])
        else:
            break
    prefix = ''.join(leading_ansi)

    wrapped_texts = textwrap.wrap(clean_text, width=max_width, break_long_words=True)
    if not wrapped_texts:
        return [line]
    # Prepend the prefix to each wrapped line
    return [prefix + w for w in wrapped_texts]

# ── Dynamic banner function ─────────────────────────────────────────────────
def banner(lines: list[str]) -> None:
    """Print a boxed banner that wraps long lines, fitting terminal width."""
    try:
        term_width = shutil.get_terminal_size().columns
    except Exception:
        term_width = 80
    box_width = min(term_width - 2, 100)
    box_width = max(box_width, 40)
    inner_width = box_width - 2

    print(f"\n{c_theme('╔' + '═'*box_width + '╗')}")
    for line in lines:
        wrapped = _wrap_line_with_ansi(line, inner_width)
        for wline in wrapped:
            visible_len = _visible_width(wline)
            pad = max(inner_width - visible_len, 0)
            print(f"{c_theme('║')} {wline}{' '*pad} {c_theme('║')}")
    print(f"{c_theme('╚' + '═'*box_width + '╝')}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG  ─  input/output dirs entered interactively at startup           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── BPM settings ─────────────────────────────────────────────────────────────
BPM_ANALYSIS_DURATION  = 120             # Seconds of audio to analyse for BPM
BPM_MIN                = 60              # Tempo range for half/double correction
BPM_MAX                = 220

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DIRECTORY & LOGGING SETUP                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SCRIPT_DIR    = Path(__file__).parent.resolve()
PARENT_DIR    = SCRIPT_DIR.parent
LOGS_DIR      = PARENT_DIR / "logs"
CONFIGS_DIR   = PARENT_DIR / "configs"
TEMP_DIR      = PARENT_DIR / "temp"
FAILED_DIR    = PARENT_DIR / "failed"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_for_directory(purpose: str, must_exist: bool = True, default: Optional[Path] = None) -> Path:
    """Ask for a directory path; rejects empty input unless a default is given."""
    while True:
        prompt = f"[?] Enter the {purpose}"
        if default is not None:
            prompt += f" (default: {default})"
        prompt += ": "
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw:
            if default is not None:
                candidate = default
            else:
                print("[!] This field cannot be empty. Try again.")
                continue
        else:
            candidate = Path(raw).expanduser().resolve()
        if must_exist and not candidate.is_dir():
            print(f"[!] Not a valid directory: {candidate}")
            continue
        return candidate


# Save this run's chosen paths under configs/ for the record.
CONFIG_FILE = CONFIGS_DIR / "bpm_mood_tagger.json"

# ── Input / output folders — chosen manually at startup, both mandatory ──────
print()

# Load saved config (if any)
default_input = default_output = None
if CONFIG_FILE.exists():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        p = Path(data.get("input_directory", ""))
        if p.is_dir():
            default_input = p
        p = Path(data.get("output_directory", ""))
        # output dir need not exist; we'll create it anyway
        default_output = p
    except Exception:
        pass

INPUT_DIR  = _prompt_for_directory("input folder to read from (read-only, never modified)", must_exist=True, default=default_input)
default_output = default_output if default_output is not None else PARENT_DIR / "outputs"
OUTPUT_DIR = _prompt_for_directory("output folder for finished, tagged files", must_exist=False, default=default_output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print()

# Save config for next run
CONFIG_FILE.write_text(
    json.dumps({
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
    }, indent=2),
    encoding="utf-8"
)

# The source in INPUT_DIR is read-only and never modified. Every file gets
# copied into temp/ first, tagged there, then the finished copy is moved
# into OUTPUT_DIR.
MUSIC_FOLDER = str(INPUT_DIR)
TEMP_OUTPUT_DIR = TEMP_DIR

# NOTE: this filename intentionally includes a trailing space — it must
# match exactly, since renaming it would silently orphan progress on disk.
PROGRESS_FILE = LOGS_DIR / "bpm_mood_checkpoint.json "
LOG_FILE      = LOGS_DIR / "bpm_mood.log"


class ImmediateFileHandler(logging.FileHandler):
    """A FileHandler that flushes after every single record — crash-safe logs."""
    def emit(self, record):
        super().emit(record)
        self.flush()


_fmt = "%(asctime)s │ %(levelname)-8s │ %(funcName)-22s │ %(message)s"

# ── Console filter: suppress noisy cataloguing messages ──
class _SuppressConsoleNoise(logging.Filter):
    _BLOCK_PHRASES = (
        "Catalogued failed file",   # file‑cataloguing log line
    )
    def filter(self, record):
        msg = record.getMessage()
        return not any(phrase in msg for phrase in self._BLOCK_PHRASES)

_file_h = ImmediateFileHandler(LOG_FILE, mode='w', encoding="utf-8")
_file_h.setLevel(logging.DEBUG)
_console_h = logging.StreamHandler(sys.stdout)
_console_h.setLevel(logging.INFO)
_console_h.addFilter(_SuppressConsoleNoise())

logging.basicConfig(
    level=logging.DEBUG,
    format=_fmt,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_file_h, _console_h],
)

for lib in ("requests", "urllib3", "librosa", "aubio", "requests.packages.urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  MOOD  ─  soft-distance classifier with global normalisation
# ──────────────────────────────────────────────────────────────────────────────

# Ideal normalised profile (Z-score) for each mood: [Tempo_Z, RMS_Z, Centroid_Z, ZCR_Z]
# These values define the "sweet spot" of each mood in a standard normal space.
MOOD_PROTOTYPES = {
    "Aggressive":   [ 2.0,  2.0,  1.8,  2.0],
    "Exciting":     [ 2.5,  1.6,  2.0,  1.2],
    "Euphoric":     [ 1.2,  1.2,  2.5,  0.8],
    "Tense":        [ 0.5,  1.5,  0.8,  1.5],
    "Intense":      [ 1.0,  2.2,  0.5,  1.0],
    "Energetic":    [ 1.5,  1.5,  1.0,  0.5],
    "Happy":        [ 1.0,  0.8,  1.8,  0.2],
    "Upbeat":       [ 1.2,  0.5,  1.2,  0.0],
    "Catchy":       [ 0.2,  0.4,  0.5,  0.2],
    "Groovy":       [-0.5,  0.3,  0.0, -0.2],
    "Romantic":     [-0.8,  0.2, -1.0, -1.0],
    "Dreamy":       [-0.5, -0.8,  1.2, -0.5],
    "Ethereal":     [-1.0, -1.2,  2.0, -1.0],
    "Chill":        [-0.2, -0.8,  0.0, -0.8],
    "Relaxed":      [-0.8, -1.0, -0.8, -0.8],
    "Calm":         [-1.5, -1.2, -1.0, -1.0],
    "Peaceful":     [-2.0, -1.5, -1.2, -1.5],
    "Bittersweet":  [-1.0, -0.2, -1.5, -0.5],
    "Melancholic":  [-1.2, -1.0, -1.8, -0.5],
    "Gloomy":       [-1.5, -1.2, -2.0, -0.2],
    "Dark":         [-0.5,  0.5, -2.5,  0.5],
    "Mysterious":   [-0.2, -0.5, -1.5,  1.0],
    "Neutral":      [ 0.0,  0.0,  0.0,  0.0],
}


def get_library_stats(music_folder: str) -> dict:
    """
    First pass: scan all .ogg files once to compute the global mean and std
    of tempo, RMS, spectral centroid, and zero-crossing rate. Caches the
    result in logs/feature_stats.json, but only re‑uses the cache if the
    set of files (sorted list) is exactly the same as last time.
    """
    cache_file = LOGS_DIR / "feature_stats.json"

    # Get current file list (sorted for stable comparison)
    files = sorted(f for f in os.listdir(music_folder) if f.lower().endswith(".ogg"))
    current_snapshot = files  # list of filenames

    # Try to load and validate cache
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            # A valid cache pairs its stats with a snapshot of the file list,
            # so a change in the library can be detected.
            if "stats" in data and "file_snapshot" in data:
                if data["file_snapshot"] == current_snapshot:
                    logger.debug("Loading cached library stats (library unchanged)")
                    print(f"\n{c_theme('◆')} Using cached library stats (library unchanged)")
                    return data["stats"]
                else:
                    logger.debug("Library changed; re‑scanning for stats")
            else:
                # No snapshot to compare against, so the cache can't be trusted.
                logger.debug("Old cache format; re‑scanning")
        except Exception:
            logger.warning("Stats cache corrupt; re‑scanning...")
            # fall through to re‑scan

    # --- Re‑scan the entire library ---
    import librosa
    import numpy as np

    total = len(files)
    logger.debug(f"First pass: scanning {total} files for global mood stats...")
    print(f"\n{c_theme('◆')} First pass — scanning {total} files for library‑wide stats (this may take a long while depending on the library size)")

    all_feats = []
    for i, fname in enumerate(files):
        if i % 500 == 0:
            print(f"    {c_dim(f'{i}/{total} scanned')}")
        try:
            filepath = os.path.join(music_folder, fname)
            y, sr = librosa.load(filepath, sr=44100, duration=BPM_ANALYSIS_DURATION)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.atleast_1d(tempo)[0])
            rms = float(librosa.feature.rms(y=y).mean())
            cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            zcr = float(librosa.feature.zero_crossing_rate(y).mean())
            all_feats.append([
                np.clip(tempo, 60, 220),
                np.clip(rms, 0, 0.3),
                np.clip(cent, 0, 6000),
                np.clip(zcr, 0, 0.25),
            ])
        except Exception as exc:
            logger.debug(f"Could not extract stats for {fname}: {exc}")
            continue

    if not all_feats:
        logger.warning("No features collected; using fallback neutral stats")
        stats = {
            "mean": [130.0, 0.10, 2500.0, 0.08],
            "std":  [30.0, 0.05, 800.0, 0.04],
        }
    else:
        arr = np.array(all_feats)
        stats = {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
        }
        # Avoid zero std
        for i, s in enumerate(stats["std"]):
            if s < 0.001:
                stats["std"][i] = 0.001

    # Save cache with snapshot
    cache_data = {
        "stats": stats,
        "file_snapshot": current_snapshot,
    }
    cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    logger.debug(f"Global stats cached: mean={stats['mean']}, std={stats['std']}")
    return stats


def detect_mood_soft(y, sr: int, global_stats: dict, debug: bool = False) -> str:
    """
    Extract audio features, Z-score normalise using global library stats,
    then assign the mood whose prototype (weighted Euclidean) is closest.
    """
    import librosa
    import numpy as np

    # 1. Extract features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    rms = float(librosa.feature.rms(y=y).mean())
    cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    raw_feats = np.array([tempo, rms, cent, zcr])

    # 2. Z-score normalise using global library stats
    mean = np.array(global_stats["mean"])
    std = np.array(global_stats["std"])
    normalized = (raw_feats - mean) / std
    # Clip to ±3 to prevent a single extreme outlier from dominating
    normalized = np.clip(normalized, -3.0, 3.0)

    if debug:
        logger.debug(f"Raw features: tempo={tempo:.1f}, RMS={rms:.3f}, cent={cent:.1f}, ZCR={zcr:.3f}")
        logger.debug(f"Normalised: tempo={normalized[0]:.2f}, RMS={normalized[1]:.2f}, cent={normalized[2]:.2f}, ZCR={normalized[3]:.2f}")

    # 3. Weighted Euclidean distance to each prototype
    weights = np.array([1.0, 1.2, 0.9, 0.8])

    best_mood = "Neutral"
    best_dist = float("inf")
    distances = {}

    for mood, proto in MOOD_PROTOTYPES.items():
        diff = (normalized - np.array(proto)) * weights
        dist = np.sum(diff ** 2)   # weighted squared Euclidean
        distances[mood] = dist
        if dist < best_dist:
            best_dist = dist
            best_mood = mood

    if debug:
        sorted_dists = sorted(distances.items(), key=lambda x: x[1])
        logger.debug("Mood distances (top 5):")
        for mood, d in sorted_dists[:5]:
            logger.debug(f"    {mood:15s} : {d:.3f}")

    logger.debug(f"Mood assigned: {best_mood} (distance={best_dist:.3f})")
    return best_mood


# ──────────────────────────────────────────────────────────────────────────────
#  BPM  ─  dual-engine detection with cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_bpm(bpm: float) -> float:
    """Apply half/double tempo correction to snap BPM into a realistic range."""
    if bpm <= 0:
        return bpm
    while bpm < BPM_MIN:
        bpm *= 2.0
    while bpm > BPM_MAX:
        bpm /= 2.0
    return bpm


def detect_bpm_aubio(wav_path: str) -> float | None:
    """
    Detect BPM using the aubio Python library.
    Uses onset-based beat tracking with a 1024-sample window for resolution.
    Returns None if aubio is not installed or detection fails.
    """
    try:
        import aubio
        import numpy as np

        samplerate  = 44100
        win_s       = 1024   # analysis window (larger = better freq resolution)
        hop_s       = 256    # hop size (smaller = better time resolution)
        max_samples = int(samplerate * BPM_ANALYSIS_DURATION)

        src           = aubio.source(wav_path, samplerate, hop_s)
        actual_sr     = src.samplerate
        beat_detector = aubio.tempo("specdiff", win_s, hop_s, actual_sr)

        bpm_readings = []
        frames_read  = 0

        while frames_read < max_samples:
            samples, read = src()
            if beat_detector(samples):
                current_bpm = beat_detector.get_bpm()
                if current_bpm > 0:
                    bpm_readings.append(float(current_bpm))
            frames_read += read
            if read < hop_s:
                break

        if not bpm_readings:
            logger.debug("aubio: no BPM readings collected")
            return None

        # Use median (more robust than mean against outliers)
        bpm = statistics.median(bpm_readings)
        bpm = normalize_bpm(bpm)
        logger.debug(f"aubio BPM = {bpm:.2f} (from {len(bpm_readings)} readings)")
        return bpm

    except ImportError:
        logger.debug("aubio not installed; skipping aubio BPM engine")
        return None
    except Exception as exc:
        logger.warning(f"aubio BPM detection failed: {exc}")
        return None


def detect_bpm_librosa(wav_path: str) -> float | None:
    """
    Detect BPM using librosa's beat tracker. Runs three independent methods
    (onset-strength, standard beat-track, and harmonic) and returns the
    median result for robustness.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(wav_path, sr=22050, duration=BPM_ANALYSIS_DURATION)

        # Method 1: onset-strength beat tracking
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        t1 = float(np.atleast_1d(
            librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        )[0])

        # Method 2: standard beat_track
        t2, _ = librosa.beat.beat_track(y=y, sr=sr)
        t2 = float(np.atleast_1d(t2)[0])

        # Method 3: harmonic component (more stable for melodic/tonal music)
        y_harm = librosa.effects.harmonic(y)
        t3, _ = librosa.beat.beat_track(y=y_harm, sr=sr)
        t3 = float(np.atleast_1d(t3)[0])

        readings = [normalize_bpm(t) for t in [t1, t2, t3] if t > 0]
        if not readings:
            return None

        bpm = statistics.median(readings)
        logger.debug(f"librosa BPM = {bpm:.2f} (methods: {[round(r, 1) for r in readings]})")
        return bpm

    except Exception as exc:
        logger.warning(f"librosa BPM detection failed: {exc}")
        return None


def detect_bpm(wav_path: str) -> float | None:
    """
    Cross-validate aubio and librosa BPM estimates.

    Rules:
      - If both agree within 5% of each other → use their average
      - If they disagree → prefer aubio (generally more accurate for beat-heavy music)
      - If only one succeeds → use that result
      - Returns None only if both engines fail
    """
    bpm_aubio   = detect_bpm_aubio(wav_path)
    bpm_librosa = detect_bpm_librosa(wav_path)

    if bpm_aubio is not None and bpm_librosa is not None:
        diff_pct = abs(bpm_aubio - bpm_librosa) / max(bpm_aubio, bpm_librosa) * 100
        if diff_pct <= 5.0:
            final = (bpm_aubio + bpm_librosa) / 2.0
            logger.debug(f"BPM engines agree (diff={diff_pct:.1f}%) → average = {final:.2f}")
        else:
            final = bpm_aubio   # prefer aubio on disagreement
            logger.debug(
                f"BPM engines disagree ({bpm_aubio:.1f} vs {bpm_librosa:.1f}, "
                f"diff={diff_pct:.1f}%) → using aubio"
            )
        return final

    elif bpm_aubio is not None:
        logger.debug(f"Only aubio succeeded: {bpm_aubio:.2f}")
        return bpm_aubio

    elif bpm_librosa is not None:
        logger.debug(f"Only librosa succeeded: {bpm_librosa:.2f}")
        return bpm_librosa

    logger.warning("Both BPM engines failed")
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  PROGRESS TRACKER  (checkpoint JSON)
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA = "bpm-mood-checkpoint"
CHECKPOINT_KEYS   = ("done", "miss", "skip", "error")


def load_progress() -> tuple[set, dict]:
    """
    Return (done_set, progress_dict).

    progress_dict always contains the four tracking lists — "done", "miss",
    "skip", "error" — plus a "_meta" block used only for human-readability.
    Only files listed under "done" are ever skipped.
    """
    progress = {key: [] for key in CHECKPOINT_KEYS}
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # A bare list means every entry is a completed filename.
                progress["done"] = [os.path.basename(f) for f in data]
            else:
                progress = data
                for key in CHECKPOINT_KEYS:
                    progress.setdefault(key, [])
                # Normalize all entries to basename
                for key in CHECKPOINT_KEYS:
                    progress[key] = [os.path.basename(f) for f in progress[key]]
        except Exception as exc:
            logger.warning(f"Could not load checkpoint ({exc}) — starting fresh")

    done_set = set(progress["done"])
    logger.debug(
        f"Checkpoint loaded — done={len(done_set)}, miss={len(progress['miss'])}, "
        f"skip={len(progress['skip'])}, error={len(progress['error'])}"
    )
    return done_set, progress


def save_progress(progress: dict) -> None:
    """
    Persist the progress dict to disk as clean, human-friendly JSON.

    A "_meta" block (schema name, last-updated time, quick counts) is kept
    up top purely for readability — the tagger itself only ever reads the
    four tracking lists, so this never affects behaviour.
    """
    ordered = {
        "_meta": {
            "schema": CHECKPOINT_SCHEMA,
            "description": (
                "Tracks per-file BPM/mood tagging progress. Only files under "
                "'done' are skipped on the next run; 'error' entries are "
                "retried automatically."
            ),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "counts": {key: len(progress.get(key, [])) for key in CHECKPOINT_KEYS},
        },
        **{key: progress.get(key, []) for key in CHECKPOINT_KEYS},
    }
    PROGRESS_FILE.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────────────────────
#  MOVE FAILED FILES
# ──────────────────────────────────────────────────────────────────────────────

def move_to_failed(filepath: str, error_type: str) -> None:
    """Catalog a problem file into <parent>/failed/<error_type>/.

    This COPIES (never moves) — the temp/ working copy is left in place so
    you can still inspect or reprocess it before choosing to delete temp/.
    """
    target_dir = FAILED_DIR / error_type
    target_dir.mkdir(parents=True, exist_ok=True)

    src = Path(filepath)
    dst = target_dir / src.name
    try:
        shutil.copy2(str(src), str(dst))
        logger.info(f"Catalogued failed file → {dst}")   # filtered from console now
        print(f"          {c_dim('↳ catalogued under failed/' + error_type + '/')}")
    except Exception as e:
        logger.error(f"Failed to catalog {src} to {dst}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  OGG → WAV CONVERSION  (for BPM and mood)
# ──────────────────────────────────────────────────────────────────────────────

def ogg_to_wav(ogg_path: str, wav_path: str, sample_rate: int = 44100) -> bool:
    """Convert OGG to WAV via ffmpeg. BPM/mood analysis works best at 44.1kHz."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", ogg_path,
                "-ar", str(sample_rate),
                "-ac", "1",          # mono
                "-acodec", "pcm_s16le",
                wav_path,
            ],
            capture_output=True,
            timeout=120,
        )
        return result.returncode == 0 and os.path.exists(wav_path)
    except Exception as exc:
        logger.warning(f"ffmpeg conversion failed: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
#  PER-FILE BPM + MOOD PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_file(filepath: str, global_stats: dict, debug: bool = False) -> str:
    """
    Copy one OGG file into temp/, analyse THAT copy, and write BPM
    and mood tags onto it. The source file at `filepath` is only ever read
    from — never modified.
    Returns: "ok" | "error:<msg>"
    """
    from mutagen.oggvorbis import OggVorbis
    import librosa
    import numpy as np

    fname = os.path.basename(filepath)
    logger.debug(f"Processing BPM/mood for: {fname}")

    dest_path = str(TEMP_OUTPUT_DIR / fname)

    tmp_dir = tempfile.mkdtemp(prefix="bpm_mood_tagger_")
    wav_44k = os.path.join(tmp_dir, "audio_44k.wav")

    try:
        shutil.copy2(filepath, dest_path)
        audio = OggVorbis(dest_path)

        # ── Convert audio to 44.1kHz mono ───────────────────────────────────
        if not ogg_to_wav(dest_path, wav_44k, sample_rate=44100):
            logger.warning(f"44kHz WAV conversion failed for {fname}; falling back to OGG direct")
            wav_44k = dest_path

        # ── Load audio ───────────────────────────────────────────────────────
        try:
            y, sr = librosa.load(wav_44k, sr=44100, duration=BPM_ANALYSIS_DURATION)
        except Exception as exc:
            logger.error(f"Could not load audio for {fname}: {exc}")
            return f"error:load_audio:{exc}"

        # ── BPM detection ────────────────────────────────────────────────────
        bpm_val = detect_bpm(wav_44k)   # uses librosa internally anyway
        logger.debug(f"BPM final: {bpm_val}")

        # ── Mood detection (soft classifier) ────────────────────────────────
        mood_str = detect_mood_soft(y, sr, global_stats, debug=debug)
        logger.debug(f"Mood final: {mood_str}")

        # ── Write tags (overwrite) ───────────────────────────────────────────
        audio["mood"] = [mood_str]
        if bpm_val is not None:
            audio["bpm"] = [str(int(round(bpm_val)))]

        audio.save()

        result_str = f"mood='{mood_str}'  bpm={int(round(bpm_val)) if bpm_val else 'N/A'}"
        logger.debug(f"OK BPM/mood: {fname} → {result_str}")
        return "ok"

    except Exception as exc:
        logger.error(f"process_file failed for '{fname}': {exc}", exc_info=True)
        # Don't leave a half-tagged copy sitting in temp/
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass
        return f"error:{exc}"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.debug("═" * 72)
    logger.debug(f"{SCRIPT_EMOJI} BPM + Mood Tagger (offline) — session starting")
    logger.debug(f"Music folder   : {MUSIC_FOLDER}")

    # ── Dependency checks ────────────────────────────────────────────────────
    missing = [pkg for pkg in ("librosa", "aubio", "mutagen", "numpy", "soundfile")
               if _pkg_missing(pkg)]
    if missing:
        sys.exit(f"Missing packages: {missing}\nRun: pip install -r requirements.txt")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit(
            "ERROR: ffmpeg not found in PATH.\n"
            "Download from https://www.gyan.dev/ffmpeg/builds/ "
            "and add it to PATH."
        )

    if not os.path.isdir(MUSIC_FOLDER):
        sys.exit(f"ERROR: MUSIC_FOLDER not found — {MUSIC_FOLDER}")

    # ── Collect files ────────────────────────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(MUSIC_FOLDER)
        if f.lower().endswith(".ogg")
    )
    total = len(all_files)
    logger.debug(f"Total OGG files found: {total}")

    # ── Global library stats for mood (first pass, cached) ──────────────────
    global_stats = get_library_stats(MUSIC_FOLDER)

    done_set, progress = load_progress()
    # Keep only done files that actually exist
    current_done = {f for f in all_files if f in done_set}

    # Tidy up: if a file somehow ended up "done" AND in error/miss/skip, drop it there.
    for key in ("error", "miss", "skip"):
        progress[key] = [f for f in progress[key] if f not in current_done]

    banner([
        f"{c_bold(SCRIPT_EMOJI + '  BPM + Mood Tagger')}  {c_dim('· offline')}",
        f"Mood classifier : soft distance (global normalisation)",
        f"Total files     : {total}",
        f"Already done    : {len(current_done)}",
        f"To process      : {total - len(current_done)}",
    ])
    print()

    # Stats counters
    counts     = {"ok": 0, "already_tagged": 0, "error": 0}
    mood_freq  = defaultdict(int)
    used_moods = set()
    all_moods  = set(MOOD_PROTOTYPES.keys())
    bpm_values = []
    debug_done = False

    for i, fname in enumerate(all_files, 1):
        filepath = os.path.join(MUSIC_FOLDER, fname)
        prefix   = c_dim(f"[{i:>{len(str(total))}}/{total}]")

        if fname in current_done:
            counts["already_tagged"] += 1
            print(f"{prefix} {c_dim('· already tagged, skipping —')} {fname}")
            continue

        print(f"{prefix} {fname}")

        result = process_file(filepath, global_stats, debug=(not debug_done))
        debug_done = True

        if result == "ok":
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)   # store just the filename

            # Read back tags for display and stats (from the temp/ working
            # copy — that's the file that actually got tagged, not the source)
            try:
                from mutagen.oggvorbis import OggVorbis
                a = OggVorbis(str(TEMP_OUTPUT_DIR / fname))
                m = a.get("mood", [""])[0]
                b = a.get("bpm", [""])[0]
                if m:
                    mood_freq[m] += 1
                    used_moods.add(m)
                if b and b.isdigit():
                    bpm_values.append(int(b))
                print(f"          {c_green('✓')} mood='{m}'  bpm={b}")
            except Exception:
                print(f"          {c_green('✓ BPM/mood written')}")

            # Move the finished, tagged copy out of temp/ and into OUTPUT_DIR.
            try:
                shutil.move(str(TEMP_OUTPUT_DIR / fname), str(OUTPUT_DIR / fname))
                print(f"          {c_dim('↳ moved to ' + str(OUTPUT_DIR))}")
            except Exception as e:
                logger.error(f"Could not move {fname} from temp/ to OUTPUT_DIR: {e}")
                print(f"          {c_red('✗ failed to move into output dir:')} {e}")

            counts["ok"] += 1
            current_done.add(fname)

        else:  # "error:<msg>" or anything unexpected — a real problem, not a soft skip
            msg = result[6:] if result.startswith("error:") else result
            print(f"          {c_red('✗ error:')} {msg}")
            move_to_failed(filepath, "error")
            counts["error"] += 1
            progress["error"].append(fname)  # will be retried next run
            save_progress(progress)

            print(f"\n{c_red('HALTED —')} a real error hit '{fname}'; stopping the whole pipeline for manual review.")
            print(f"  {c_dim('Fix the underlying issue, then re-run run_tagger — already-tagged files are skipped automatically.')}")
            logger.critical(f"HALTED — real error on {fname}: {msg}")
            sys.exit(1)

        save_progress(progress)   # checkpoint after every single file

    save_progress(progress)  # final save, just in case

    # ── Session summary ──────────────────────────────────────────────────────
    banner([
        f"{c_bold('✓ BPM + Mood Tagger — session complete')}",
        f"Tagged (new)    : {counts['ok']}",
        f"Already done    : {counts['already_tagged']}",
        f"Errors          : {counts['error']}",
        f"Total processed : {counts['ok'] + counts['error']}",
    ])

    if bpm_values:
        avg_bpm = sum(bpm_values) / len(bpm_values)
        print(f"\n  {c_bold('Average BPM (tagged this session):')} {avg_bpm:.1f}")

    # Only show mood summary if we actually tagged new files
    unused = []
    if mood_freq:
        print(f"\n  {c_bold('Mood frequency this session:')}")
        for mood, cnt in sorted(mood_freq.items(), key=lambda x: -x[1]):
            print(f"    {mood:<18} : {cnt}")

        unused = sorted(all_moods - used_moods)

    if counts["ok"] > 0:
        if unused:
            print(f"\n  {c_dim(f'Not seen this session ({len(unused)}/{len(all_moods)}):')}")
            line = ""
            for m in unused:
                if len(line) + len(m) + 2 > 70:
                    print(f"    {c_dim(line)}")
                    line = ""
                line += m + ", "
            if line:
                print(f"    {c_dim(line[:-2])}")
        else:
            print(f"\n  {c_green(f'All {len(all_moods)} moods showed up at least once!')}")

    sign_off = "smooth run, nice work (｡•̀ᴗ-)✧" if counts["error"] == 0 \
        else "a few hiccups — worth a peek at the log (._.)"
    print(f"\n  {c_dim(sign_off)}\n")

    logger.debug(f"Session complete — counts={counts}")
    logger.debug("═" * 72)


def _pkg_missing(pkg: str) -> bool:
    try:
        __import__(pkg)
        return False
    except ImportError:
        return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{c_yellow('Interrupted —')} progress saved up to the last completed file.")
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}", exc_info=True)
        print(f"\n{c_red('FATAL:')} {exc}")
        sys.exit(1)