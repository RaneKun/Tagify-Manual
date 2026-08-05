"""
📝 Plain Lyrics Tagger — Offline Edition
─────────────────────────────────────────────────────────────────────────────
Extracts plain (unsynced) lyrics from audio using a local, high-accuracy
pipeline (Demucs → DeepFilterNet → Whisper). No internet connection required.

AUDIO PIPELINE
───────────────────────────────────
  1. Demucs htdemucs_6s        (6-stem, minimal instrument bleed into vocal stem)
  2. ffmpeg highpass=100Hz + loudnorm=I=-16   (remove sub-bass, normalise level)
  3. ffmpeg resample to 48000 Hz              (for DeepFilterNet3)
  4. DeepFilterNet3                           (neural denoising on 48 kHz stem)
  5. Faster-Whisper large-v3                  (word_timestamps=True, VAD off, beam=20)
  6. Fallback to raw audio if Demucs stem avg_logprob < FALLBACK_LOGPROB
  7. Hallucination filtering, word count gates → plain text output

Checkpoint behaviour:
  ▸ Files listed under "done" are ALWAYS skipped on the next run.
  ▸ Files under "error" are NOT skipped — they'll be retried automatically.
  ▸ "miss" and "skip" are reserved for future use (kept for consistency).
  ▸ Delete or clear the checkpoint file to reprocess everything from scratch.

Instrumental detection triggers on any of:
  ▸ Vocal RMS energy below MIN_VOCAL_RMS
  ▸ Whisper avg_logprob below HARD_ABORT_LOGPROB
  ▸ Average no-speech probability above AVG_NO_SPEECH_THRESHOLD
  ▸ Total word count below MIN_TOTAL_WORDS

Output locations:
  ▸ Log file   → logs/lyrics_offline.log         (everything, DEBUG level)
  ▸ Checkpoint → logs/lyrics_offline_checkpoint.json

Before first run: edit the CONFIG block below to point at your music folder.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import re

# ─── Windows cuDNN DLL fix ──────────────────────────────────────────────────
_DLL_COOKIE = None
if os.name == "nt":
    try:
        import importlib.util as _ilu
        _ts = _ilu.find_spec("torch")
        if _ts and _ts.origin:
            _torch_lib = os.path.join(os.path.dirname(_ts.origin), "lib")
            if os.path.isdir(_torch_lib):
                _DLL_COOKIE = os.add_dll_directory(_torch_lib)
    except Exception:
        pass

import json
import shutil
import subprocess
import tempfile
import logging
import warnings
import unicodedata
import textwrap
from typing import Optional
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOOK & FEEL  ─  colors + tiny console helpers                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

THEME        = MAGENTA   # ← this script's signature color
SCRIPT_EMOJI = "📝"       # lyrics tagger's signature icon

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

DEMUCS_MODEL  = "htdemucs_6s"
DEMUCS_DEVICE = "cuda"
DEMUCS_SHIFTS = 2

USE_DEEPFILTER    = True
DEEPFILTER_DEVICE = "cuda"

WHISPER_MODEL       = "large-v3"
WHISPER_DEVICE      = "cuda"
WHISPER_DTYPE       = "float16"
WHISPER_BEAM        = 20
WHISPER_PATIENCE    = 3.0
WHISPER_TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

VAD_THRESHOLD      = 0.10
VAD_MIN_SPEECH_MS  = 80
VAD_MIN_SILENCE_MS = 400
VAD_SPEECH_PAD_MS  = 400

SEGMENT_MAX_NO_SPEECH = 0.85
SEGMENT_MIN_LOGPROB   = -2.00

MIN_LINE_DURATION    = 0.30
SPARSE_WORDS_PER_SEC = 0.15
SPARSE_MAX_WORDS     = 3
SPARSE_MIN_DURATION  = 3.0

HARD_ABORT_LOGPROB      = -0.60
FALLBACK_LOGPROB        = -0.40
MIN_VOCAL_RMS           = 0.004
AVG_NO_SPEECH_THRESHOLD = 0.75
MIN_TOTAL_WORDS         = 15

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
CONFIG_FILE = CONFIGS_DIR / "local_lyrics_tagger.json"

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

PROGRESS_FILE = LOGS_DIR / "lyrics_offline_checkpoint.json"
LOG_FILE      = LOGS_DIR / "lyrics_offline.log"


class ImmediateFileHandler(logging.FileHandler):
    """A FileHandler that flushes after every single record — crash-safe logs."""
    def emit(self, record):
        super().emit(record)
        self.flush()


logger = logging.getLogger("plain_lyrics_tagger")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(
    "%(asctime)s │ %(levelname)-8s │ %(funcName)-26s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_fh = ImmediateFileHandler(LOG_FILE, mode='w', encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)

# ── Console filter: block any stray cataloguing messages (belt‑and‑suspenders) ──
class _SuppressConsoleNoise(logging.Filter):
    _BLOCK_PHRASES = (
        "Catalogued failed file",
    )
    def filter(self, record):
        msg = record.getMessage()
        return not any(phrase in msg for phrase in self._BLOCK_PHRASES)

# ── Console handler stays completely silent — every user-facing line goes
#    through print() instead, so the console never gets flooded by the
#    pipeline's very chatty per-stage debug logging. ─────────────────────────
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.CRITICAL + 1)   # suppress everything on console
_ch.setFormatter(_fmt)
_ch.addFilter(_SuppressConsoleNoise())

logger.addHandler(_fh)
logger.addHandler(_ch)

# Suppress noisy third-party loggers.
for lib in ("tensorflow", "absl", "urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
#  PROGRESS TRACKER  (checkpoint JSON)
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA = "lyrics-offline-checkpoint"
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
                "Tracks per-file plain-lyrics tagging progress. Only files "
                "under 'done' are skipped on the next run; 'error' entries "
                "are retried automatically."
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
        logger.info(f"Catalogued failed file → {dst}")   # filtered from console
        print(f"          {c_dim('↳ catalogued under failed/' + error_type + '/')}")
    except Exception as e:
        logger.error(f"Failed to catalog {src} to {dst}: {e}")


def _halt_pipeline(fname: str, detail: str) -> None:
    """A real error (not a soft instrumental miss) happened — stop the WHOLE
    pipeline here for manual review instead of moving to the next file."""
    print(f"\n{c_red('HALTED —')} a real error hit '{fname}'; stopping the whole pipeline for manual review.")
    print(f"  {c_dim('(instrumental confirmations never halt — only real errors do)')}")
    print(f"  {c_dim('Fix the underlying issue, then re-run run_tagger — already-tagged files are skipped automatically.')}")
    logger.critical(f"HALTED — {fname}: {detail}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
#  HALLUCINATION DETECTION
# ──────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_RE = re.compile(
    r"thank\s+you"
    r"|thanks\s+for\s+(watching|listening|tuning|joining|having)"
    r"|i'?ll\s+see\s+you\s+(next\s+time|in\s+the\s+next)"
    r"|see\s+you\s+next\s+time"
    r"|for\s+more\s+information"
    r"|visit\s+(us\s+at\s+)?www\."
    r"|www\.\S+\.(com|org|net|gov|edu)"
    r"|fema\s*\.(org|gov)"
    r"|https?://"
    r"|\.(com|org|net|gov|edu)\b"
    r"|subtitles?\s+by"
    r"|transcribed\s+by"
    r"|auto.?generated"
    r"|copyright\s*[0-9©]"
    r"|all\s+rights\s+reserved"
    r"|like\s+and\s+subscribe"
    r"|don'?t\s+forget\s+to\s+subscribe"
    r"|hit\s+the\s+bell"
    r"|please\s+subscribe",
    re.IGNORECASE,
)

_HALLUCINATION_SHORT = {
    "thank you", "thanks", "cheers",
    "see you later", "see you next time", "see you soon",
    "goodbye", "good bye", "good night",
    "ciao", "adios", "au revoir", "arrivederci", "auf wiedersehen", "sayonara",
    "uh huh", "mm hmm",
    "end of video", "end of song", "the end",
}

def is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if _HALLUCINATION_RE.search(t):
        return True
    words = re.sub(r"[^a-zA-Z\s]", "", t).split()
    if len(words) <= 2:
        normalised = t.lower().strip(".,!?… ")
        if normalised in _HALLUCINATION_SHORT:
            return True
    return False

def is_sparse(text: str, duration: float) -> bool:
    words = text.strip().split()
    if len(words) > SPARSE_MAX_WORDS or duration <= SPARSE_MIN_DURATION:
        return False
    return (len(words) / duration) < SPARSE_WORDS_PER_SEC

# ──────────────────────────────────────────────────────────────────────────────
#  PLAIN LYRICS BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_plain_lyrics(segments: list) -> str:
    raw_lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        duration = seg.end - seg.start
        if duration < MIN_LINE_DURATION:
            logger.debug(f"  [DROP-short {duration:.2f}s] '{text}'")
            continue
        if is_hallucination(text):
            logger.debug(f"  [DROP-halluc] '{text}'")
            continue
        if is_sparse(text, duration):
            logger.debug(f"  [DROP-sparse {duration:.2f}s] '{text}'")
            continue
        raw_lines.append(text)

    logger.debug(f"  Plain lines after filter: {len(raw_lines)}/{len(segments)} segments")
    if not raw_lines:
        return ""

    full_text = "\n".join(raw_lines)
    lines = full_text.split("\n")
    deduped = []
    prev = None
    for line in lines:
        norm = re.sub(r"[^\w\s]", "", line.lower()).strip()
        if norm and norm != prev:
            deduped.append(line)
            prev = norm
    result = "\n".join(deduped)

    word_count = len(result.split())
    logger.debug(f"  Final plain lyrics: {word_count} words, {len(deduped)} lines")
    preview = result[:200].replace("\n", " ")
    logger.debug(f"  Preview: {preview}...")
    return result

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def ogg_to_wav(src: str, dst: str, sr: int = 16000) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", str(sr), "-ac", "1", dst],
            capture_output=True, timeout=120,
            encoding='utf-8', errors='replace'
        )
        ok = r.returncode == 0 and os.path.exists(dst)
        if not ok:
            logger.debug(f"  ogg_to_wav failed (rc={r.returncode})")
        return ok
    except Exception as exc:
        logger.warning(f"  ogg_to_wav exception: {exc}")
        return False

def extract_vocals_demucs(ogg_path: str, out_dir: str) -> str | None:
    # Create a safe temporary copy of the audio file with a sanitized name
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', os.path.basename(ogg_path))
    safe_path = os.path.join(out_dir, safe_name)
    try:
        shutil.copy2(ogg_path, safe_path)
        logger.debug(f"  Copied to safe path: {safe_path}")
    except Exception as e:
        logger.error(f"  Failed to copy file to safe path: {e}")
        return None

    cmd = [
        sys.executable, "-m", "demucs",
        "-d", DEMUCS_DEVICE,
        "--two-stems=vocals",
        "-n", DEMUCS_MODEL,
        f"--shifts={DEMUCS_SHIFTS}",
        "--out", out_dir,
        safe_path,
    ]
    logger.debug(f"  Demucs cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
    except subprocess.TimeoutExpired:
        logger.error("  Demucs timed out (> 900 s)")
        return None
    except Exception as exc:
        logger.error(f"  Demucs subprocess error: {exc}")
        return None

    if r.returncode != 0:
        logger.warning(f"  Demucs exit {r.returncode}: {r.stderr[-400:]}")
        return None

    # The output folder will use the safe name (without extension) as the track name
    track_base = os.path.splitext(safe_name)[0]
    vp = os.path.join(out_dir, DEMUCS_MODEL, track_base, "vocals.wav")

    if os.path.exists(vp):
        logger.debug(f"  Demucs vocal stem: {vp} (exists)")
        return vp

    # Fallback: scan the output directory for any .wav file
    base_dir = os.path.join(out_dir, DEMUCS_MODEL, track_base)
    if os.path.isdir(base_dir):
        wav_files = [f for f in os.listdir(base_dir) if f.lower().endswith('.wav')]
        if wav_files:
            fallback = os.path.join(base_dir, wav_files[0])
            logger.debug(f"  Demucs vocal stem found as fallback: {fallback}")
            return fallback

    logger.error(f"  Demucs vocal stem not found at {vp} and no .wav fallback in {base_dir}")
    return None

def preprocess_ffmpeg(inp: str, out: str) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", inp,
             "-af", "highpass=f=100,loudnorm=I=-16:TP=-1.5:LRA=11", out],
            capture_output=True, timeout=180,
            encoding='utf-8', errors='replace'
        )
        ok = r.returncode == 0 and os.path.exists(out)
        logger.debug(f"  ffmpeg preprocess → ok={ok}")
        return ok
    except Exception as exc:
        logger.warning(f"  ffmpeg preprocess exception: {exc}")
        return False

def resample_to_48k(inp: str, out: str) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", inp, "-ar", "48000", "-ac", "1", out],
            capture_output=True, timeout=120,
            encoding='utf-8', errors='replace'
        )
        ok = r.returncode == 0 and os.path.exists(out)
        logger.debug(f"  48k resample → ok={ok}")
        return ok
    except Exception as exc:
        logger.warning(f"  48k resample exception: {exc}")
        return False

def enhance_deepfilter(inp: str, out: str, df_model, df_state) -> bool:
    try:
        from df.enhance import enhance, load_audio, save_audio
        audio, _ = load_audio(inp, sr=df_state.sr())
        enhanced = enhance(df_model, df_state, audio)
        save_audio(out, enhanced, df_state.sr())
        ok = os.path.exists(out)
        logger.debug(f"  DeepFilterNet enhance → ok={ok}")
        return ok
    except Exception as exc:
        logger.warning(f"  DeepFilterNet error: {exc} — using preprocessed stem")
        return False

def vocal_rms(wav_path: str) -> float:
    try:
        import wave, struct, math
        with wave.open(wav_path, "rb") as wf:
            n_read = min(wf.getnframes(), wf.getframerate() * 30)
            raw    = wf.readframes(n_read)
            step   = wf.getnchannels()
            samps  = struct.unpack_from(f"{n_read}h", raw[:n_read * 2])[::step]
            rms    = math.sqrt(sum(s * s for s in samps) / len(samps)) / 32768.0
            logger.debug(f"  vocal_rms = {rms:.5f}")
            return rms
    except Exception as exc:
        logger.debug(f"  vocal_rms error: {exc} — returning 1.0")
        return 1.0

# ──────────────────────────────────────────────────────────────────────────────
#  WHISPER TRANSCRIPTION
# ──────────────────────────────────────────────────────────────────────────────

def transcribe(
    audio_path:   str,
    whisper_model,
    context_hint: str = "",
) -> tuple[list, float, float]:
    prompt = (
        f"Song lyrics: {context_hint}" if context_hint.strip()
        else "Song lyrics:"
    )
    logger.debug(f"  Whisper prompt: {prompt!r}")

    segments_gen, info = whisper_model.transcribe(
        audio_path,
        beam_size                   = WHISPER_BEAM,
        patience                    = WHISPER_PATIENCE,
        language                    = None,
        initial_prompt              = prompt,
        word_timestamps             = True,
        condition_on_previous_text  = False,
        vad_filter                  = False,
        no_speech_threshold         = 0.6,
        compression_ratio_threshold = 2.4,
        temperature                 = WHISPER_TEMPERATURE,
        log_prob_threshold          = -1.0,
    )

    raw = list(segments_gen)
    logger.debug(
        f"  Whisper raw: {len(raw)} segments | "
        f"lang={info.language} ({info.language_probability:.2f})"
    )

    kept = []
    for seg in raw:
        t = seg.text.strip()
        if not t or len(t) < 2:
            continue
        if is_hallucination(t):
            logger.debug(f"  [DROP-halluc-seg] '{t[:50]}'")
            continue
        if seg.no_speech_prob > SEGMENT_MAX_NO_SPEECH:
            logger.debug(f"  [DROP-nsp {seg.no_speech_prob:.2f}] '{t[:40]}'")
            continue
        if seg.avg_logprob < SEGMENT_MIN_LOGPROB:
            logger.debug(f"  [DROP-lp {seg.avg_logprob:.3f}] '{t[:40]}'")
            continue
        kept.append(seg)

    logger.debug(f"  After filter: {len(kept)}/{len(raw)} segments kept")

    if not kept:
        return [], 1.0, -99.0

    avg_nsp = sum(s.no_speech_prob for s in kept) / len(kept)
    avg_lp  = sum(s.avg_logprob    for s in kept) / len(kept)
    logger.debug(f"  avg_no_speech={avg_nsp:.3f}  avg_logprob={avg_lp:.3f}")
    return kept, avg_nsp, avg_lp

# ──────────────────────────────────────────────────────────────────────────────
#  AUDIO DURATION (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def get_audio_duration(filepath: str) -> float:
    try:
        from mutagen.oggvorbis import OggVorbis
        return OggVorbis(filepath).info.length
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=30,
            encoding='utf-8', errors='replace'
        )
        return float(r.stdout.strip())
    except Exception:
        return 240.0

# ──────────────────────────────────────────────────────────────────────────────
#  PER-FILE PROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

def process_file(filepath: str, whisper_model, df_model, df_state) -> str:
    from mutagen.oggvorbis import OggVorbis

    fname        = os.path.basename(filepath)
    base         = os.path.splitext(fname)[0]
    context_hint = base if " - " in base else ""

    logger.debug(f"  File: {fname}")
    tmp_dir = tempfile.mkdtemp(prefix="plain_lyrics_")

    try:
        # 1. Demucs
        logger.debug(f"  [1/6] Demucs {DEMUCS_MODEL} shifts={DEMUCS_SHIFTS}...")
        raw_vocals = extract_vocals_demucs(filepath, tmp_dir)
        if raw_vocals is None:
            return "error:demucs:vocal stem not created"

        # 2. ffmpeg preprocess
        logger.debug("  [2/6] ffmpeg preprocess (highpass + loudnorm)...")
        preprocessed = os.path.join(tmp_dir, "vocals_pre.wav")
        current = (
            preprocessed
            if preprocess_ffmpeg(raw_vocals, preprocessed)
            else (logger.warning("  ffmpeg failed — using raw stem"), raw_vocals)[1]
        )

        # 3. DeepFilterNet
        if USE_DEEPFILTER and df_model is not None:
            logger.debug("  [3/6] Resample 48 kHz → DeepFilterNet3...")
            resampled = os.path.join(tmp_dir, "vocals_48k.wav")
            enhanced  = os.path.join(tmp_dir, "vocals_enh.wav")
            if resample_to_48k(current, resampled):
                if enhance_deepfilter(resampled, enhanced, df_model, df_state):
                    current = enhanced
                    logger.debug("  [3/6] DeepFilterNet3 applied")
                else:
                    logger.warning("  [3/6] DeepFilterNet failed — using preprocessed")
            else:
                logger.warning("  [3/6] 48 kHz resample failed — skipping DeepFilterNet")
        else:
            logger.debug("  [3/6] DeepFilterNet disabled")

        # 3b Vocal RMS
        chk = os.path.join(tmp_dir, "chk16k.wav")
        rms = vocal_rms(chk if ogg_to_wav(current, chk, 16000) else current)
        logger.debug(f"  [3b] Vocal RMS = {rms:.5f}  (min={MIN_VOCAL_RMS})")
        if rms < MIN_VOCAL_RMS:
            logger.debug("  [3b] Near-silent stem — instrumental")
            return "instrumental"

        # 4. Whisper
        logger.debug(f"  [4/6] Whisper (beam={WHISPER_BEAM}, patience={WHISPER_PATIENCE})...")
        try:
            segs, avg_nsp, avg_lp = transcribe(current, whisper_model, context_hint)
        except Exception as exc:
            return f"error:whisper:{exc}"

        # 4b Fallback
        if avg_lp < FALLBACK_LOGPROB:
            logger.debug(
                f"  [4b] Low stem confidence (logprob={avg_lp:.3f}) "
                f"— trying raw audio..."
            )
            try:
                raw_wav = os.path.join(tmp_dir, "raw.wav")
                if ogg_to_wav(filepath, raw_wav):
                    fb_segs, fb_nsp, fb_lp = transcribe(
                        raw_wav, whisper_model, context_hint
                    )
                    if fb_lp > avg_lp:
                        segs, avg_nsp, avg_lp = fb_segs, fb_nsp, fb_lp
                        logger.debug(f"  [4b] Raw audio better (logprob={fb_lp:.3f})")
                    else:
                        logger.debug("  [4b] Demucs stem still better — keeping")
            except Exception as exc:
                logger.warning(f"  [4b] Fallback error: {exc}")

        # 5. Instrumental gates
        if avg_lp < HARD_ABORT_LOGPROB:
            logger.debug(f"  [5/6] Hard abort — logprob={avg_lp:.3f} < {HARD_ABORT_LOGPROB} — instrumental")
            return "instrumental"

        if avg_nsp > AVG_NO_SPEECH_THRESHOLD:
            logger.debug(f"  [5/6] High no-speech avg ({avg_nsp:.3f} > {AVG_NO_SPEECH_THRESHOLD}) — instrumental")
            return "instrumental"

        if not segs:
            logger.debug("  [5/6] No segments after filter — instrumental")
            return "instrumental"

        total_words = sum(len(s.text.split()) for s in segs)
        logger.debug(f"  [5/6] Total words in kept segs: {total_words} (min={MIN_TOTAL_WORDS})")
        if total_words < MIN_TOTAL_WORDS:
            logger.debug(f"  [5/6] Too few words ({total_words}) — instrumental")
            return "instrumental"

        # 6. Build and write
        logger.debug("  [6/6] Building plain lyrics...")
        plain_lyrics = build_plain_lyrics(segs)

        if not plain_lyrics.strip():
            logger.debug("  [6/6] Plain lyrics empty after build — instrumental")
            return "instrumental"

        audio_obj           = OggVorbis(filepath)
        audio_obj["lyrics"] = [plain_lyrics]
        audio_obj.save()

        line_count = len(plain_lyrics.split("\n"))
        word_count = len(plain_lyrics.split())
        detail = f"{line_count} lines, {word_count} words, logprob={avg_lp:.3f}"
        logger.debug(f"  OK: {fname} — {detail}")
        return f"ok:{detail}"

    except Exception as exc:
        logger.error(f"  process_file exception for '{fname}': {exc}", exc_info=True)
        return f"error:{exc}"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.debug("═" * 72)
    logger.debug(f"{SCRIPT_EMOJI} Plain Lyrics Tagger (offline, unsynced) — session starting")
    logger.debug(f"Music folder : {MUSIC_FOLDER}")
    logger.debug(f"Demucs shifts: {DEMUCS_SHIFTS}")
    logger.debug(f"Whisper beam : {WHISPER_BEAM}  patience={WHISPER_PATIENCE}")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name  = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.debug(f"CUDA: {name}  ({total:.1f} GB VRAM)")
        else:
            logger.warning("CUDA not available — processing will be very slow on CPU")
    except ImportError:
        sys.exit("ERROR: PyTorch not installed — see requirements.txt")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("ERROR: faster-whisper not installed — pip install faster-whisper")

    for pkg in ("demucs", "mutagen"):
        try:
            __import__(pkg)
        except ImportError:
            sys.exit(f"ERROR: {pkg} not installed — pip install {pkg}")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.debug("ffmpeg: OK")
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("ERROR: ffmpeg not found in PATH")

    # ── Collect candidate files ──────────────────────────────────────────────
    # Checked BEFORE loading Demucs/Whisper so an empty run doesn't pay for a
    # slow model load.
    all_files = sorted(
        f for f in os.listdir(MUSIC_FOLDER)
        if f.lower().endswith(".ogg")
    )
    total = len(all_files)
    logger.debug(f"Total candidate OGG files in input folder: {total}")

    done_set, progress = load_progress()
    current_done = {f for f in all_files if f in done_set}
    for key in ("error", "miss", "skip"):
        progress[key] = [f for f in progress[key] if f not in current_done]

    if total == 0:
        banner([
            f"{c_bold(SCRIPT_EMOJI + '  Lyrics Tagger')}  {c_dim('· offline · Demucs + Whisper')}",
            f"Nothing to do — the input folder has no .ogg files.",
        ])
        print(f"\n  {c_dim('nothing here for me to do, moving on (｡•̀ᴗ-)✧')}\n")
        logger.debug("No candidate files in input folder — exiting cleanly.")
        return

    print(f"{c_dim(f'Found {total} candidate file(s) in the input folder.')}")
    print()

    # DeepFilterNet
    df_model = df_state = None
    if USE_DEEPFILTER:
        print(f"\n{c_theme('◆')} Loading DeepFilterNet3...")
        try:
            import torch
            from df.enhance import init_df
            df_model, df_state, _ = init_df()
            if DEEPFILTER_DEVICE == "cuda" and torch.cuda.is_available():
                df_model = df_model.to("cuda")
                print(f"    {c_green('✓ ready on CUDA')}")
                logger.debug("DeepFilterNet3: CUDA")
            else:
                print(f"    {c_yellow('✓ ready on CPU')}")
                logger.debug("DeepFilterNet3: CPU")
        except ImportError:
            print(f"    {c_yellow('⚠ not installed — stage skipped')}")
            print(f"    {c_dim('Install: pip install deepfilternet')}")
            logger.warning("DeepFilterNet3 not installed")
        except Exception as exc:
            print(f"    {c_yellow('⚠ load failed — stage skipped:')} {exc}")
            logger.warning(f"DeepFilterNet3 load failed: {exc}")

    # Whisper
    print(f"\n{c_theme('◆')} Loading Whisper {WHISPER_MODEL} {c_dim('(may download ~3 GB on first run)')}...")
    try:
        whisper_model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_DTYPE
        )
        print(f"    {c_green('✓ ready')}")
        logger.debug(f"Whisper: {WHISPER_MODEL} on {WHISPER_DEVICE}/{WHISPER_DTYPE}")
    except Exception as exc:
        sys.exit(f"ERROR: Whisper load failed — {exc}")

    df_status = (
        f"ON ({DEEPFILTER_DEVICE.upper()})" if df_model is not None
        else "OFF (not installed)"
    )

    banner([
        f"{c_bold(SCRIPT_EMOJI + '  Plain Lyrics Tagger')}  {c_dim('· offline · unsynced')}",
        f"Demucs        : {DEMUCS_MODEL}  shifts={DEMUCS_SHIFTS}",
        f"DeepFilter    : {df_status}  (explicit 48 kHz resample)",
        f"Whisper       : {WHISPER_MODEL}  beam={WHISPER_BEAM}  patience={WHISPER_PATIENCE}",
        f"Seg filter    : no_speech<{SEGMENT_MAX_NO_SPEECH}  logprob>{SEGMENT_MIN_LOGPROB}",
        f"Hard abort    : avg_logprob < {HARD_ABORT_LOGPROB}",
        f"Min words     : {MIN_TOTAL_WORDS}",
        f"Total files   : {total}   Already done : {len(current_done)}",
    ])
    print()

    logger.debug(
        f"Starting run — {total} files, {len(current_done)} already done, "
        f"beam={WHISPER_BEAM}"
    )

    counts = {"ok": 0, "instrumental": 0, "error": 0, "skipped": 0}

    for i, fname in enumerate(all_files, 1):
        src_path = os.path.join(MUSIC_FOLDER, fname)   # original input, read-only
        filepath = str(TEMP_DIR / fname)                # working copy
        prefix   = c_dim(f"[{i:>{len(str(total))}}/{total}]")

        if fname in current_done:
            counts["skipped"] += 1
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

        try:
            result = process_file(filepath, whisper_model, df_model, df_state)
        except Exception as exc:
            logger.error(f"Unhandled exception for '{fname}': {exc}", exc_info=True)
            result = f"error:{exc}"

        if result == "instrumental":
            print(f"          {c_yellow('⚠ instrumental — confirmed, no lyrics')}")
            move_to_failed(filepath, "instrumental")
            counts["instrumental"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)   # final — no method could find lyrics for this one
            current_done.add(fname)

        elif result.startswith("ok:"):
            try:
                from mutagen.oggvorbis import OggVorbis
                audio = OggVorbis(filepath)
                lyrics = audio.get("lyrics", [""])[0]
                snippet = lyrics[:60].replace('\n', ' ') + "..." if len(lyrics) > 60 else lyrics.replace('\n', ' ')
                print(f"          {c_green('✓')} \"{snippet}\"")
            except Exception:
                print(f"          {c_green('✓ lyrics written')}")

            try:
                shutil.move(filepath, str(OUTPUT_DIR / fname))
                print(f"          {c_dim('↳ moved to ' + str(OUTPUT_DIR))}")
            except Exception as e:
                logger.error(f"Could not move {fname} from temp/ to OUTPUT_DIR: {e}")
                print(f"          {c_red('✗ failed to move into output dir:')} {e}")

            counts["ok"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)
            current_done.add(fname)

        else:  # a real error — not a soft instrumental confirmation
            print(f"          {c_red('✗ error:')} {result}")
            move_to_failed(filepath, "error")
            counts["error"] += 1
            progress["error"].append(fname)   # will be retried next run
            save_progress(progress)
            _halt_pipeline(fname, result)

        save_progress(progress)   # checkpoint after every single file

    save_progress(progress)  # final save, just in case

    banner([
        f"{c_bold('✓ Plain Lyrics Tagger — session complete')}",
        f"Lyrics written  : {counts['ok']}",
        f"Instrumental    : {counts['instrumental']}",
        f"Skipped (log)   : {counts['skipped']}",
        f"Errors          : {counts['error']}",
    ])

    sign_off = "smooth run, nice work (｡•̀ᴗ-)✧" if counts["error"] == 0 \
        else "a few hiccups — worth a peek at the log (._.)"
    print(f"\n  {c_dim(sign_off)}\n")

    logger.debug(f"Run complete: {counts}")
    logger.debug("═" * 72)

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