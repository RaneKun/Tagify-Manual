"""
♪ Genre Tagger — Offline Edition ♪
─────────────────────────────────────────────────────────────────────────────
Sniffs out up to 3 comma-separated genre tags using MusicNN and writes the
"genre" Vorbis comment tag straight into your .ogg files. Fully offline —
no internet connection needed, no API keys, just your files and a model.

Checkpoint behaviour:
  ▸ Files listed under "done" are ALWAYS skipped on the next run.
  ▸ Files under "error" are NOT skipped — they'll be retried automatically.
  ▸ "miss" and "skip" are reserved for future use (kept for consistency).
  ▸ Delete or clear the checkpoint file to reprocess everything from scratch.

Output locations:
  ▸ Log file   → logs/genre__offline.log        (everything, DEBUG level)
  ▸ Checkpoint → logs/genre_offline_checkpoint.json

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
import unicodedata
import textwrap
from typing import Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── TensorFlow verbosity suppression (must happen before TF import) ─────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOOK & FEEL  ─  colors + tiny console helpers                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

THEME        = CYAN     # every banner/box in this script uses this color
SCRIPT_EMOJI = "🎼"     # offline genre tagger's signature icon

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
# ║  CONFIG  ─  fully local/offline: no API keys needed, just input/output   ║
# ║  dirs entered interactively at startup                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Genre settings ───────────────────────────────────────────────────────────
MUSICNN_MODEL          = "MSD_musicnn"   # MSD_musicnn or MSD_musicnn_big (slower)
TOP_N_GENRES           = 3               # Maximum genres to write per file
CONFIDENCE_THRESHOLD   = 0.04            # Minimum MusicNN score to consider a tag
SUB_GENRE_THRESHOLD    = 0.10            # Minimum score for additional sub-genres
MAX_PER_FAMILY         = 2               # Cap tags from same genre family

# ── Input window for MusicNN ─────────────────────────────────────────────────
MUSICNN_INPUT_LENGTH   = 3               # Seconds per analysis chunk (3 = standard)

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
CONFIG_FILE = CONFIGS_DIR / "local_genre_tagger.json"

# ── Input / output folders — chosen manually at startup, both mandatory ──────
print()

default_input = default_output = None
if CONFIG_FILE.exists():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        p = Path(data.get("input_directory", ""))
        if p.is_dir():
            default_input = p
        p = Path(data.get("output_directory", ""))
        default_output = p
    except Exception:
        pass

INPUT_DIR  = _prompt_for_directory("input folder of OGG files to read (read-only, never modified)", must_exist=True, default=default_input)
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

PROGRESS_FILE = LOGS_DIR / "genre_offline_checkpoint.json"
LOG_FILE      = LOGS_DIR / "genre__offline.log"


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

# Suppress noisy third-party loggers so they don't clutter our pretty output.
for lib in ("requests", "urllib3", "tensorflow", "requests.packages.urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  GENRE VOCABULARY
# ──────────────────────────────────────────────────────────────────────────────

# Tags MusicNN produces that aren't actually genres — filtered out on sight.
JUNK_TAGS = {
    "seen live", "favorites", "favourite", "love", "awesome", "good", "cool",
    "beautiful", "amazing", "best", "classic", "chill", "relax", "party",
    "workout", "running", "sleep", "study", "sad", "happy", "sexy",
    "under 2000 listeners", "all", "spotify", "youtube", "downloaded",
    "owned", "wishlist", "recommended", "favourite albums", "albums i own",
    "music", "songs", "tracks", "female vocalists", "male vocalists",
    "instrumental", "cover", "remix", "live", "acoustic", "demo", "edit",
}

# Maps raw lowercase tag → pretty display name.
TAG_DISPLAY = {
    "electronic": "Electronic", "electro": "Electro", "dance": "Dance",
    "electronica": "Electronica", "house": "House", "techno": "Techno",
    "trance": "Trance", "dubstep": "Dubstep", "edm": "EDM",
    "ambient": "Ambient", "chillout": "Chillout", "downtempo": "Downtempo",
    "drum and bass": "Drum and Bass", "dnb": "DnB", "future bass": "Future Bass",
    "trap": "Trap", "phonk": "Phonk", "lofi": "Lo-Fi",
    "hip-hop": "Hip Hop", "hip hop": "Hip Hop", "rap": "Rap", "rnb": "R&B",
    "r&b": "R&B", "gangsta": "Gangsta Rap", "soul": "Soul", "funk": "Funk",
    "rock": "Rock", "alternative": "Alternative", "indie": "Indie",
    "punk": "Punk", "metal": "Metal", "hard rock": "Hard Rock",
    "classic rock": "Classic Rock", "grunge": "Grunge",
    "pop": "Pop", "synth pop": "Synth Pop", "dance pop": "Dance Pop",
    "k-pop": "K-Pop", "j-pop": "J-Pop",
    "jazz": "Jazz", "blues": "Blues", "reggae": "Reggae", "reggaeton": "Reggaeton",
    "latin": "Latin", "country": "Country", "classical": "Classical",
    "folk": "Folk", "world": "World", "experimental": "Experimental",
    "oldies": "Oldies", "60s": "60s", "80s": "80s", "90s": "90s",
}

# Groups similar tags into families so one style doesn't hog all 3 slots.
FAMILY_MAP = {
    "Electronic": [
        "electronic", "electro", "dance", "electronica", "house", "techno",
        "trance", "dubstep", "edm", "ambient", "chillout", "downtempo",
        "drum and bass", "dnb", "future bass", "lofi", "phonk",
    ],
    "Hip-Hop":    ["hip-hop", "hip hop", "rap", "rnb", "r&b", "trap", "gangsta"],
    "Rock":       ["rock", "alternative", "indie", "punk", "metal", "hard rock",
                   "classic rock", "grunge"],
    "Pop":        ["pop", "synth pop", "dance pop", "k-pop", "j-pop"],
    "Soul-Funk":  ["soul", "funk"],
    "Jazz":       ["jazz"],
    "Blues":      ["blues"],
    "Reggae":     ["reggae", "reggaeton"],
    "Latin":      ["latin"],
    "Country":    ["country"],
    "Classical":  ["classical"],
    "Folk":       ["folk"],
    "World":      ["world"],
    "Experimental": ["experimental"],
    "Oldies":     ["oldies", "60s", "70s", "80s", "90s"],
}


def extract_genres(taggram, tag_names: list, debug: bool = False) -> str:
    """
    Convert MusicNN's taggram output into a comma-separated genre string.

    Algorithm:
      1. Average scores across all analysis windows
      2. Sort descending by score
      3. Filter out junk tags
      4. Pass 1 — one representative tag per family (most confident family first)
      5. Pass 2 — fill any remaining slots with strong sub-genre tags
      6. Cap — never more than MAX_PER_FAMILY tags from a single family
    """
    import numpy as np
    avg    = taggram.mean(axis=0)
    ranked = avg.argsort()[::-1]

    if debug:
        logger.debug("Top-10 MusicNN tags for this track:")
        for idx in ranked[:10]:
            logger.debug(f"    {tag_names[idx]:20s} : {avg[idx]:.4f}")

    # Build candidate list: (score, raw_lower, display, family)
    candidates = []
    for idx in ranked:
        score = float(avg[idx])
        if score < CONFIDENCE_THRESHOLD:
            break
        raw = tag_names[idx].lower().strip()
        if raw in JUNK_TAGS:
            continue
        display = TAG_DISPLAY.get(raw, tag_names[idx].title())
        family = next(
            (fam for fam, members in FAMILY_MAP.items() if raw in members),
            display,    # unknown tag → treated as its own family
        )
        candidates.append((score, raw, display, family))

    if not candidates:
        return ""

    chosen     = []
    fam_counts = defaultdict(int)

    # Pass 1: one tag per family, most confident families first
    for score, raw, display, family in candidates:
        if len(chosen) >= TOP_N_GENRES:
            break
        if fam_counts[family] > 0 or display in chosen:
            continue
        chosen.append(display)
        fam_counts[family] += 1

    # Pass 2: fill remaining slots with strong sub-genre tags
    if len(chosen) < TOP_N_GENRES:
        for score, raw, display, family in candidates:
            if len(chosen) >= TOP_N_GENRES:
                break
            if fam_counts[family] >= MAX_PER_FAMILY:
                continue
            if display in chosen:
                continue
            if score < SUB_GENRE_THRESHOLD:
                continue
            chosen.append(display)
            fam_counts[family] += 1

    return ", ".join(chosen)


# ──────────────────────────────────────────────────────────────────────────────
#  PROGRESS TRACKER  (checkpoint JSON)
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA = "genre-offline-checkpoint"
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
                "Tracks per-file genre-tagging progress. Only files under "
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

    Copies (never moves) — the temp/ working copy is left in place so you
    can still inspect or reprocess it before choosing to delete temp/.
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


def _halt_pipeline(fname: str, detail: str) -> None:
    """A real error (not a soft 'no genre' miss) happened — stop the WHOLE
    pipeline here for manual review instead of silently moving to the next file."""
    print(f"\n{c_red('HALTED —')} a real error hit '{fname}'; stopping the whole pipeline for manual review.")
    print(f"  {c_dim('(no-genre misses never halt — only real errors do)')}")
    print(f"  {c_dim('Fix the underlying issue, then re-run run_tagger — already-tagged files are skipped automatically.')}")
    logger.critical(f"HALTED — {fname}: {detail}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
#  OGG → WAV CONVERSION  (for MusicNN)
# ──────────────────────────────────────────────────────────────────────────────

def ogg_to_wav(ogg_path: str, wav_path: str, sample_rate: int = 16000) -> bool:
    """Convert OGG to WAV via ffmpeg. MusicNN expects 16kHz mono input."""
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
#  PER-FILE GENRE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def process_file(filepath: str, debug: bool = False) -> str:
    """
    Analyse one OGG file and write its genre tag (overwrites any existing tag).
    Returns: "ok" | "error:<msg>" | "no_genre"
    """
    from musicnn.extractor import extractor
    from mutagen.oggvorbis import OggVorbis
    import numpy as np

    fname = os.path.basename(filepath)
    logger.debug(f"Processing genre for: {fname}")

    tmp_dir = tempfile.mkdtemp(prefix="genre_tagger_")
    wav_16k = os.path.join(tmp_dir, "audio_16k.wav")

    try:
        audio = OggVorbis(filepath)

        # ── Convert audio to 16kHz mono ─────────────────────────────────────
        if not ogg_to_wav(filepath, wav_16k, sample_rate=16000):
            logger.warning(f"16kHz WAV conversion failed for {fname}; trying to use OGG directly")
            wav_16k = filepath   # musicnn can sometimes handle OGG directly

        # ── Genre via MusicNN ────────────────────────────────────────────────
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            taggram, tag_names = extractor(
                wav_16k,
                model=MUSICNN_MODEL,
                input_length=MUSICNN_INPUT_LENGTH,
                extract_features=False,
            )
        genre_str = extract_genres(np.array(taggram), tag_names, debug=debug)
        logger.debug(f"Genre extracted: {genre_str!r}")

        if not genre_str:
            logger.debug(f"No genre tags found for {fname}")
            return "no_genre"

        # ── Write tag (overwrite) ───────────────────────────────────────────
        audio["genre"] = [genre_str]
        audio.save()

        logger.debug(f"OK genre: {fname} → {genre_str!r}")
        return "ok"

    except Exception as exc:
        logger.error(f"process_file failed for '{fname}': {exc}", exc_info=True)
        return f"error:{exc}"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.debug("═" * 72)
    logger.debug(f"{SCRIPT_EMOJI} Genre Tagger (offline) — session starting")
    logger.debug(f"Music folder   : {MUSIC_FOLDER}")
    logger.debug(f"MusicNN model  : {MUSICNN_MODEL}")

    # ── Dependency checks ────────────────────────────────────────────────────
    missing = [pkg for pkg in ("musicnn", "mutagen", "numpy", "soundfile")
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

    # ── Collect candidate files ──────────────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(MUSIC_FOLDER)
        if f.lower().endswith(".ogg")
    )
    total = len(all_files)
    logger.debug(f"Total candidate OGG files in input folder: {total}")

    done_set, progress = load_progress()
    # Keep only those done files that actually exist in the current folder
    current_done = {f for f in all_files if f in done_set}

    # Tidy up: remove from error/miss/skip any files that are now in current_done
    for key in ("error", "miss", "skip"):
        progress[key] = [f for f in progress[key] if f not in current_done]

    if total == 0:
        banner([
            f"{c_bold(SCRIPT_EMOJI + '  Genre Tagger')}  {c_dim('· offline · MusicNN')}",
            f"Nothing to do — the input folder has no .ogg files.",
        ])
        print(f"\n  {c_dim('nothing here for me to do, moving on (｡•̀ᴗ-)✧')}\n")
        logger.debug("No candidate files in input folder — exiting cleanly.")
        return

    banner([
        f"{c_bold(SCRIPT_EMOJI + '  Genre Tagger')}  {c_dim('· offline · MusicNN')}",
        f"Model         : {MUSICNN_MODEL}",
        f"Total files   : {total}",
        f"Already done  : {len(current_done)}",
        f"To process    : {total - len(current_done)}",
    ])
    print()

    # Stats counters
    counts         = {"ok": 0, "already_tagged": 0, "error": 0, "no_genre": 0}
    genre_freq     = defaultdict(int)
    all_genres_set = set(TAG_DISPLAY.values())   # full known vocabulary
    used_genres    = set()
    debug_done     = False

    for i, fname in enumerate(all_files, 1):
        src_path = os.path.join(MUSIC_FOLDER, fname)
        filepath = str(TEMP_DIR / fname)   # never touch the input file directly
        prefix   = c_dim(f"[{i:>{len(str(total))}}/{total}]")

        if fname in current_done:
            counts["already_tagged"] += 1
            print(f"{prefix} {c_dim('· already tagged, skipping —')} {fname}")
            continue

        print(f"{prefix} {fname}")

        try:
            shutil.copy2(src_path, filepath)
        except Exception as e:
            print(f"          {c_red('✗ error copying to temp/:')} {e}")
            logger.error(f"Could not copy {src_path} to temp/: {e}")
            counts["error"] += 1
            progress["error"].append(fname)
            save_progress(progress)
            _halt_pipeline(fname, f"copy to temp/ failed: {e}")
            continue

        result = process_file(filepath, debug=(not debug_done))
        debug_done = True

        if result == "ok":
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)   # just filename

            # Read the tag back so we can report + tally it.
            try:
                from mutagen.oggvorbis import OggVorbis
                a = OggVorbis(filepath)
                g = a.get("genre", [""])[0]
                for tag in g.split(", "):
                    if tag.strip():
                        genre_freq[tag.strip()] += 1
                        used_genres.add(tag.strip())
                print(f"          {c_green('✓')} {g}")
            except Exception:
                print(f"          {c_green('✓ genre written')}")

            try:
                shutil.move(filepath, str(OUTPUT_DIR / fname))
                print(f"          {c_dim('↳ moved to ' + str(OUTPUT_DIR))}")
            except Exception as e:
                logger.error(f"Could not move {fname} from temp/ to OUTPUT_DIR: {e}")
                print(f"          {c_red('✗ failed to move into output dir:')} {e}")

            counts["ok"] += 1
            # Update current_done so that subsequent files know this one is done
            current_done.add(fname)

        elif result == "no_genre":
            print(f"          {c_yellow('⚠ no confident genre found')}")
            move_to_failed(filepath, "no_genre")
            counts["no_genre"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)   # final — no method could find a genre for this one
            current_done.add(fname)

        else:  # "error:<msg>" or anything unexpected — a real problem, not a soft skip
            msg = result[6:] if result.startswith("error:") else result
            print(f"          {c_red('✗ error:')} {msg}")
            move_to_failed(filepath, "error")
            counts["error"] += 1
            progress["error"].append(fname)  # will be retried next run
            save_progress(progress)
            _halt_pipeline(fname, msg)

        save_progress(progress)   # checkpoint after every single file

    save_progress(progress)  # final save, just in case

    # ── Session summary ──────────────────────────────────────────────────────
    banner([
        f"{c_bold('✓ Genre Tagger — session complete')}",
        f"Tagged (new)    : {counts['ok']}",
        f"Already done    : {counts['already_tagged']}",
        f"No genre found  : {counts['no_genre']}",
        f"Errors          : {counts['error']}",
        f"Total processed : {counts['ok'] + counts['no_genre'] + counts['error']}",
    ])

    unused = []   # initialise to avoid reference error
    if genre_freq:
        print(f"\n  {c_bold('Genre frequency this session:')}")
        for tag, cnt in sorted(genre_freq.items(), key=lambda x: -x[1]):
            print(f"    {tag:<22} : {cnt}")

        unused = sorted(all_genres_set - used_genres)

    # Only show "all genres seen" if we actually tagged new files
    if counts["ok"] > 0:
        if unused:
            total_possible = len(all_genres_set)
            print(f"\n  {c_dim(f'Not seen this session ({len(unused)}/{total_possible}):')}")
            line = ""
            for g in unused:
                if len(line) + len(g) + 2 > 70:
                    print(f"    {c_dim(line)}")
                    line = ""
                line += g + ", "
            if line:
                print(f"    {c_dim(line[:-2])}")
        else:
            print(f"\n  {c_green('Every predefined genre showed up at least once!')}")

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