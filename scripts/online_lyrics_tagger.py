"""
🌐📝 Plain Lyrics Tagger — Online Edition
─────────────────────────────────────────────────────────────────────────────
Fetches plain (unsynced) lyrics from multiple internet sources and writes
them directly into the LYRICS Vorbis comment tag.

All lyrics are automatically stripped of timestamps (e.g. [00:12.34], [01:23:45])
and metadata lines (e.g. "作词 : ...", "作曲 : ...", "Lyrics: ...") so that only
the actual lyric text is stored.

Sources (tried in this order for every song):
  1. LRCLib     — Free REST API, no auth, good plain‑lyrics coverage
  2. NetEase    — Via syncedlyrics library (plain mode); best for JP/KR/ZH
  3. Musixmatch — Via syncedlyrics library (plain mode); strong western/English

Checkpoint behaviour:
  ▸ Files listed in the checkpoint JSON under "done" are ALWAYS skipped.
  ▸ Files NOT in "done" are processed (and the LYRICS tag overwritten)
    regardless of whether a tag already exists.
  ▸ "miss", "skip", "error" are NOT used for skipping — they are informational.
  ▸ Delete or clear the checkpoint file to reprocess everything.

Instrumental detection triggers on any of:
  ▸ Title contains a known instrumental keyword (e.g. "bgm", "(inst.)")
  ▸ LRCLib explicitly flags the track as instrumental → immediately treated as instrumental
  ▸ No lyrics found on any source → logged as "no_lyrics"

Network‑error handling:
  ▸ If every source fails because of connectivity problems (DNS, timeouts, etc.),
    the file is moved to failed/network_error/ and retried on the next run.
  ▸ Network‑error files are NOT added to "done", so they are always retried.

Output locations:
  ▸ Log file   → logs/lyrics_online.log          (everything, DEBUG level)
  ▸ Checkpoint → logs/lyrics_online_checkpoint.json

Before first run: edit the CONFIG block below to point at your music folder.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import sys
import time
import unicodedata
import textwrap
from typing import Optional
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from mutagen.oggvorbis import OggVorbis

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOOK & FEEL  ─  colors + tiny console helpers                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

THEME        = RED   # ← this script's signature color
SCRIPT_EMOJI = "🌐📝"     # online lyrics tagger's signature icon

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

# ══════════════════════════════════════════════════════════════════════════════
#  LOGS FOLDER — created BEFORE any logging.FileHandler call
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR    = Path(__file__).parent.resolve()
LOGS_DIR      = SCRIPT_DIR.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING  (both file and console; DEBUG to file, INFO to console)
# ══════════════════════════════════════════════════════════════════════════════
class ImmediateFileHandler(logging.FileHandler):
    """A FileHandler that flushes after every single record — crash-safe logs."""
    def emit(self, record):
        super().emit(record)
        self.flush()

_fmt    = "%(asctime)s │ %(levelname)-8s │ %(message)s"
_file_h = ImmediateFileHandler(LOGS_DIR / "lyrics_online.log", encoding="utf-8", mode="w")
_file_h.setLevel(logging.DEBUG)

# ── Console handler: block noisy library messages ──
class _SuppressLibraryNoise(logging.Filter):
    """Reject log records whose message contains known library spam."""
    _BLOCK_PHRASES = (
        "No suitable lyrics found for",        # syncedlyrics
        "Moved failed file to",                # file‑move logging
        "Catalogued failed file to",           # file‑cataloguing logging (added)
        "Lyrics found for",                    # syncedlyrics success
        "Got status code",                     # syncedlyrics HTTP errors
        "[Musixmatch]",                        # syncedlyrics direct prints
        "[NetEase]",                           # (if present)
        "Instrumental (LRCLib flag)",          # internal instrumental detection
        "An error occurred while searching",   # syncedlyrics network errors
        "HTTPSConnectionPool",                 # urllib3 error details
        "Max retries exceeded",                # urllib3 error details
        "NameResolutionError",                 # DNS failure details
        "Failed to resolve",                   # DNS failure message
    )
    def filter(self, record):
        msg = record.getMessage()
        return not any(phrase in msg for phrase in self._BLOCK_PHRASES)

_console_h = logging.StreamHandler(sys.stdout)
_console_h.setLevel(logging.INFO)
_console_h.addFilter(_SuppressLibraryNoise())

logging.basicConfig(level=logging.DEBUG, format=_fmt, handlers=[_file_h, _console_h])

# Suppress third-party library logs
for lib in ('requests', 'urllib3', 'requests.packages.urllib3',
            'syncedlyrics', 'Musixmatch', 'NetEase'):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← no API keys needed (LRCLib + syncedlyrics are both keyless);
#  input/output dirs are entered interactively at startup, below
# ══════════════════════════════════════════════════════════════════════════════

PARENT_DIR   = SCRIPT_DIR.parent
CONFIGS_DIR  = PARENT_DIR / "configs"
TEMP_DIR     = PARENT_DIR / "temp"
FAILED_DIR   = PARENT_DIR / "failed"
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
CONFIG_FILE = CONFIGS_DIR / "online_lyrics_tagger.json"

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
MUSIC_DIR: str = str(INPUT_DIR)

CHECKPOINT_FILE: Path = LOGS_DIR / "lyrics_online_checkpoint.json"

SYNCEDLYRICS_PROVIDERS: list[str] = ["NetEase", "Musixmatch"]

LRCLIB_BASE_URL: str = "https://lrclib.net/api"
LRCLIB_TIMEOUT: int  = 15

REQUEST_DELAY_MIN: float       = 0.6
REQUEST_DELAY_MAX: float       = 1.4
INTER_PROVIDER_DELAY: float    = 0.35

MIN_WORD_COUNT: int = 15

INSTRUMENTAL_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "instrumental", "inst.", " inst ", "(inst)",
    "bgm", "music box", "piano ver", "piano version",
    "acoustic instrumental", "karaoke", "off vocal", "(off)",
    "(ost)", "original soundtrack", "original score",
    "no vocal", "no vocals",
})

_CLEAN_PATTERNS: list[tuple[str, str]] = [
    ("strip_feat_paren",   r'\s*[\(\[]\s*feat(?:uring)?\.?\s+[^\)\]]+[\)\]]'),
    ("strip_with_paren",   r'\s*[\(\[]\s*with\s+[^\)\]]+[\)\]]'),
    ("strip_feat_hyphen",  r'\s*-\s*feat(?:uring)?\.?\s+.+$'),
    ("strip_remix_paren",  r'\s*[\(\[]\s*[^\)\]]*(?:remix|edit|vip|mix)[^\)\]]*[\)\]]'),
    ("strip_remix_hyphen", r'\s*-\s*\S+\s+(?:remix|edit|vip|mix|version)\s*$'),
    ("strip_speed",        r'\s*[\(\[-]?\s*(?:sped[\s-]up|slowed(?:\s+down)?|nightcore|speed\s*up)'
                           r'[^\)\]]*[\)\]]?'),
    ("strip_lang_ver",     r'\s*[-\(\[]\s*(?:japanese|english|korean|chinese|spanish|german|french)'
                           r'\s*(?:ver(?:sion)?\.?)?[\)\]]?'),
    ("strip_last_paren",   r'\s*\([^)]{1,40}\)\s*$'),
]

# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  LYRICS CLEANING – plain text only, no timestamps, no metadata headers
# ─────────────────────────────────────────────────────────────────────────────
# Matches [mm:ss.xx], [mm:ss], [hh:mm:ss.xx], [hh:mm:ss]
TIMESTAMP_RE = re.compile(r'\[\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\]')

# Known metadata prefixes (Chinese, English) – lines starting with these are removed
_METADATA_PREFIXES = (
    "作词", "作曲", "编曲", "制作", "混音", "母带", "录音", "合声",
    "和声", "吉他", "钢琴", "鼓", "贝斯", "弦乐", "监制",
    "歌手", "专辑", "发行", "厂牌",
    "lyrics", "lyricist", "composer", "composition", "arrangement",
    "arranger", "producer", "production", "mixing", "mastering",
    "recording", "engineer", "vocals", "backing vocals",
    "written by", "music by", "words by",
)

# Precompile a regex that matches a line starting (after optional whitespace) with any of these prefixes,
# followed by optional Chinese/English colon and the rest.
_META_RE = re.compile(
    r'^\s*(?:' + '|'.join(re.escape(p) for p in _METADATA_PREFIXES) + r')\s*[：:]\s*.*$',
    re.IGNORECASE
)

def clean_lyrics_text(text: str) -> str:
    """Remove timestamps and metadata lines, return plain lyrics."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Remove timestamps
        line = TIMESTAMP_RE.sub('', line).strip()
        if not line:
            continue
        # Remove metadata lines (e.g. 作词 : ..., Lyrics: ...)
        if _META_RE.match(line):
            continue
        # Also remove lines that consist only of non‑lyric indicators like "---" or "★"
        if re.fullmatch(r'[-=★☆♪♫•·]{2,}', line):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

# ─────────────────────────────────────────────────────────────────────────────
#  STREAM FILTER FOR DIRECT PRINTS (syncedlyrics uses print(), not logging)
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def _filter_stderr(block_phrases: tuple[str, ...]):
    """Temporarily replace sys.stderr, discarding lines that contain any phrase."""
    import io
    class _Filter(io.StringIO):
        def write(self, s):
            for phrase in block_phrases:
                if phrase in s:
                    return          # skip it
            super().write(s)
    old_stderr = sys.stderr
    sys.stderr = _Filter()
    try:
        yield
    finally:
        sys.stderr = old_stderr

SYNCEDLYRICS_STDERR_FILTER = ("[Musixmatch]", "[NetEase]", "Got status code")

# ─────────────────────────────────────────────────────────────────────────────
#  SENTINELS
# ─────────────────────────────────────────────────────────────────────────────
_INSTRUMENTAL  = object()
_NETWORK_ERROR = object()      # all sources unreachable due to network issues

# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT  (dict with "done", "miss", "skip", "error")
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA = "lyrics-online-checkpoint"
CHECKPOINT_KEYS   = ("done", "miss", "skip", "error")

def load_checkpoint() -> tuple[set[str], dict]:
    """
    Return (done_set, progress_dict).

    progress_dict always contains the four tracking lists — "done", "miss",
    "skip", "error" — plus a "_meta" block used only for human-readability.
    Only files listed under "done" are ever skipped.
    """
    progress = {key: [] for key in CHECKPOINT_KEYS}
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
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
            logger.warning(f"Checkpoint load failed ({exc}); starting fresh")
            progress = {key: [] for key in CHECKPOINT_KEYS}
    done_set = set(progress["done"])
    logger.debug(f"Checkpoint loaded — done={len(done_set)}, miss={len(progress['miss'])}, "
                 f"skip={len(progress['skip'])}, error={len(progress['error'])}")
    return done_set, progress

def save_checkpoint(progress: dict) -> None:
    """
    Persist the progress dict to disk as clean, human-friendly JSON.

    A "_meta" block (schema name, last-updated time, quick counts) is kept
    up top purely for readability — the tagger itself only ever reads the
    four tracking lists, so this never affects behaviour.
    """
    try:
        ordered = {
            "_meta": {
                "schema": CHECKPOINT_SCHEMA,
                "description": (
                    "Tracks per-file online plain-lyrics tagging progress. "
                    "Only files under 'done' are skipped on the next run."
                ),
                "last_updated": datetime.now().isoformat(timespec="seconds"),
                "counts": {key: len(progress.get(key, [])) for key in CHECKPOINT_KEYS},
            },
            **{key: progress.get(key, []) for key in CHECKPOINT_KEYS},
        }
        CHECKPOINT_FILE.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.error(f"Checkpoint save failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
#  MOVE FAILED FILES
# ─────────────────────────────────────────────────────────────────────────────

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
        logger.info(f"Catalogued failed file to {dst}")   # file log only, filtered from console
        print(f"          {c_dim('↳ catalogued under failed/' + error_type + '/')}")
    except Exception as e:
        logger.error(f"Failed to catalog {src} to {dst}: {e}")


def _halt_pipeline(fname: str, detail: str) -> None:
    """A real error (not a soft instrumental/no-lyrics miss) happened — stop
    the WHOLE pipeline here for manual review instead of moving to the next file."""
    print(f"\n{c_red('HALTED —')} a real error hit '{fname}'; stopping the whole pipeline for manual review.")
    print(f"  {c_dim('(instrumental / no-lyrics misses never halt — only network/technical errors do)')}")
    print(f"  {c_dim('Fix the underlying issue, then re-run run_tagger — already-tagged files are skipped automatically.')}")
    logger.critical(f"HALTED — {fname}: {detail}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  FILENAME PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_filename(filepath: str) -> tuple[str, str]:
    stem = Path(filepath).stem
    if " - " not in stem:
        logger.warning(f"Non-standard filename (missing ' - ' separator): '{stem}'")
        return "", stem
    artist, title = stem.split(" - ", 1)
    return artist.strip(), title.strip()

def get_search_variants(artist: str, title: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = [(artist, title)]
    current_title = title
    for _label, pattern in _CLEAN_PATTERNS:
        cleaned = re.sub(pattern, "", current_title, flags=re.IGNORECASE).strip()
        if cleaned and cleaned.lower() != current_title.lower():
            variants.append((artist, cleaned))
            current_title = cleaned
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for v in variants:
        key = (v[0].lower(), v[1].lower())
        if key not in seen:
            seen.add(key)
            unique.append(v)
    logger.debug(f"Search variants ({len(unique)}): {unique}")
    return unique

def is_instrumental_by_title(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in INSTRUMENTAL_TITLE_KEYWORDS)

# ─────────────────────────────────────────────────────────────────────────────
#  LYRICS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_lyrics(text: str) -> bool:
    if not text or not text.strip():
        return False
    # Text has already been cleaned of timestamps and metadata.
    word_count = len(text.split())
    logger.debug(f"  Word count: {word_count} (minimum: {MIN_WORD_COUNT})")
    return word_count >= MIN_WORD_COUNT

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 1: LRCLib
# ─────────────────────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent":    "MusicTagger/1.0 (personal library; not commercial)",
    "Lrclib-Client": "MusicTagger/1.0",
})

def _lrclib_get(endpoint: str, params: dict) -> Optional[dict | list]:
    url = f"{LRCLIB_BASE_URL}/{endpoint}"
    try:
        resp = _SESSION.get(url, params=params, timeout=LRCLIB_TIMEOUT)
        if resp.status_code == 404:
            logger.debug(f"LRCLib /{endpoint} → 404 not found")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        logger.debug(f"LRCLib /{endpoint} → request timed out")
        return _NETWORK_ERROR
    except requests.RequestException as exc:
        logger.debug(f"LRCLib /{endpoint} → network error: {exc}")
        return _NETWORK_ERROR

def fetch_lrclib(artist: str, title: str):
    """Returns lyrics (plain str), _INSTRUMENTAL, _NETWORK_ERROR, or None."""
    params: dict[str, str] = {"track_name": title}
    if artist:
        params["artist_name"] = artist

    data = _lrclib_get("search", params)
    if data is _NETWORK_ERROR:
        return _NETWORK_ERROR

    if isinstance(data, list) and data:
        # Check for instrumental flag first
        for item in data:
            if item.get("instrumental"):
                logger.debug("LRCLib: first result flagged as instrumental")
                return _INSTRUMENTAL
        # Then look for plain lyrics
        for item in data:
            plain = (item.get("plainLyrics") or "").strip()
            if plain:
                plain = clean_lyrics_text(plain)   # clean now
                if is_valid_lyrics(plain):
                    logger.debug(f"LRCLib /search (plain): '{item.get('artistName')}' – '{item.get('trackName')}'")
                    return plain

    item = _lrclib_get("get", params)
    if item is _NETWORK_ERROR:
        return _NETWORK_ERROR

    if isinstance(item, dict):
        if item.get("instrumental"):
            logger.debug("LRCLib: /get result flagged as instrumental")
            return _INSTRUMENTAL
        plain = (item.get("plainLyrics") or "").strip()
        if plain:
            plain = clean_lyrics_text(plain)
            if is_valid_lyrics(plain):
                logger.debug("LRCLib /get (plain)")
                return plain

    return None

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 2: syncedlyrics
# ─────────────────────────────────────────────────────────────────────────────

def fetch_syncedlyrics(artist: str, title: str):
    """Returns lyrics (plain str), _NETWORK_ERROR, or None."""
    try:
        import syncedlyrics  # type: ignore
    except ImportError:
        logger.debug("syncedlyrics library not installed — skipping this source")
        return None

    q_title_first  = f"{title} {artist}".strip() if artist else title.strip()
    q_artist_first = f"{artist} {title}".strip() if artist else title.strip()
    queries        = list(dict.fromkeys([q_title_first, q_artist_first]))

    network_error_occurred = False

    for provider in SYNCEDLYRICS_PROVIDERS:
        for query in queries:
            logger.debug(f"syncedlyrics [{provider}] query (plain): '{query}'")
            try:
                with _filter_stderr(SYNCEDLYRICS_STDERR_FILTER):
                    lyrics = syncedlyrics.search(
                        query,
                        providers=[provider],
                        plain_only=True,
                        enhanced=False,
                    )
            except Exception as exc:
                logger.debug(f"syncedlyrics [{provider}] exception: {exc}")
                if any(pat in str(exc).lower() for pat in (
                    "connection", "timeout", "resolve", "refused",
                    "network", "dns", "name resolution"
                )):
                    network_error_occurred = True
                lyrics = None

            if lyrics and isinstance(lyrics, str) and lyrics.strip():
                lyrics = clean_lyrics_text(lyrics)   # clean syncedlyrics output
                if is_valid_lyrics(lyrics):
                    logger.debug(f"syncedlyrics [{provider}]: plain lyrics found")
                    return lyrics

            time.sleep(INTER_PROVIDER_DELAY)

    if network_error_occurred:
        return _NETWORK_ERROR
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-SOURCE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def _polite_sleep() -> None:
    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

def fetch_lyrics_all_sources(
    artist: str,
    title: str,
):
    """
    Returns tuple: (lyrics_or_sentinel, source_name)
      lyrics_or_sentinel can be a plain lyrics string, _INSTRUMENTAL,
      _NETWORK_ERROR, or None.
    """
    variants = get_search_variants(artist, title)
    had_network_error = False

    for v_artist, v_title in variants:
        logger.debug(f"  → variant: '{v_artist}' – '{v_title}'")

        lrclib_result = fetch_lrclib(v_artist, v_title)
        if lrclib_result is _INSTRUMENTAL:
            return _INSTRUMENTAL, "LRCLib"
        if lrclib_result is _NETWORK_ERROR:
            had_network_error = True
        elif isinstance(lrclib_result, str):
            return lrclib_result, "LRCLib"

        _polite_sleep()

        sync_result = fetch_syncedlyrics(v_artist, v_title)
        if sync_result is _NETWORK_ERROR:
            had_network_error = True
        elif isinstance(sync_result, str):
            return sync_result, "syncedlyrics"

        _polite_sleep()

    if had_network_error:
        return _NETWORK_ERROR, "none"
    return None, "none"

# ─────────────────────────────────────────────────────────────────────────────
#  PER-FILE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_file(filepath: str) -> tuple[str, str]:
    artist, title = parse_filename(filepath)

    # 1) Title keyword check (fast, no API)
    if is_instrumental_by_title(title):
        detail = f"keyword match: '{title}'"
        logger.info(f"Instrumental (keyword): {Path(filepath).name}")
        return "instrumental", detail

    logger.debug(f"Searching lyrics: '{artist}' – '{title}'")
    try:
        lyrics_or_sentinel, source = fetch_lyrics_all_sources(artist, title)
    except Exception as exc:
        logger.exception(f"Unexpected error fetching lyrics for {filepath}")
        return "error", str(exc)

    # 2) LRCLib instrumental flag
    if lyrics_or_sentinel is _INSTRUMENTAL:
        logger.info(f"Instrumental (LRCLib flag): {Path(filepath).name}")
        return "instrumental", "LRCLib instrumental flag"

    # 3) Network error (all sources unreachable)
    if lyrics_or_sentinel is _NETWORK_ERROR:
        logger.info(f"Network error (all sources unreachable): {Path(filepath).name}")
        return "network_error", "all sources network error"

    if lyrics_or_sentinel is None:
        logger.debug(f"No lyrics found: '{artist}' – '{title}'")
        return "no_lyrics", "all sources exhausted"

    lyrics = lyrics_or_sentinel  # plain string already cleaned
    word_count = len(lyrics.split())
    snippet = lyrics[:100].replace('\n', ' ')
    logger.debug(f"Lyrics fetched from {source}: word count={word_count}, snippet='{snippet}...'")

    try:
        tags = OggVorbis(filepath)
        tags["lyrics"] = [lyrics]
        tags.save()
    except Exception as exc:
        logger.error(f"Tag write failed for {filepath}: {exc}")
        return "error", f"write_tag: {exc}"

    logger.debug(f"ok [{source}] — {Path(filepath).name} (plain)")
    return "ok", source

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.debug("═" * 72)
    logger.debug(f"{SCRIPT_EMOJI} Plain Lyrics Tagger (online) — session starting")
    # Dependency checks, etc.
    try:
        import mutagen  # noqa: F401
    except ImportError:
        sys.exit("[!] mutagen is not installed.\n    pip install mutagen")

    _has_syncedlyrics = True
    try:
        import syncedlyrics  # noqa: F401 — type: ignore
    except ImportError:
        _has_syncedlyrics = False
        logger.debug(
            "syncedlyrics not installed — NetEase/Musixmatch sources disabled.\n"
            "  Install: pip install syncedlyrics"
        )

    if not os.path.isdir(MUSIC_DIR):
        sys.exit(f"[!] MUSIC_DIR not found:\n    {MUSIC_DIR}")

    all_ogg = sorted(
        os.path.join(MUSIC_DIR, f)
        for f in os.listdir(MUSIC_DIR)
        if f.lower().endswith(".ogg")
        and os.path.isfile(os.path.join(MUSIC_DIR, f))
    )
    total = len(all_ogg)
    if total == 0:
        sys.exit(f"[!] No .ogg files found in:\n    {MUSIC_DIR}")

    done_set, progress = load_checkpoint()
    current_done = {os.path.basename(f) for f in all_ogg if os.path.basename(f) in done_set}
    for key in ("error", "miss", "skip"):
        progress[key] = [f for f in progress[key] if f not in current_done]

    resumed = len(current_done)

    active_sources: list[str] = ["LRCLib"]
    if _has_syncedlyrics:
        active_sources += [f"syncedlyrics/{p}" for p in SYNCEDLYRICS_PROVIDERS]

    banner([
        f"{c_bold(SCRIPT_EMOJI + '  Plain Lyrics Tagger')}  {c_dim('· online')}",
        f"Directory      : {MUSIC_DIR}",
        f"Files          : {total}",
        f"Already done   : {resumed}",
        f"Active sources : {' → '.join(active_sources)}",
        f"Min word count : {MIN_WORD_COUNT}",
    ])
    print()

    logger.debug(f"Run started — total={total}, resumed={resumed}, sources={active_sources}")

    cnt: dict[str, int] = {
        "ok":           0,
        "instrumental": 0,
        "no_lyrics":    0,
        "network_error": 0,
        "errors":       0,
        "skipped":      0,
    }

    width = len(str(total))

    for i, src_path in enumerate(all_ogg, 1):
        fname = os.path.basename(src_path)
        prefix = c_dim(f"[{i:>{width}}/{total}]")

        if fname in current_done:
            print(f"{prefix} {c_dim('· already tagged, skipping —')} {fname}")
            cnt["skipped"] += 1
            continue

        print(f"{prefix} {fname}")
        logger.debug(f"Processing [{i}/{total}]: {fname}")

        # Never touch the input file directly — work on a copy in temp/.
        filepath = str(TEMP_DIR / fname)
        try:
            shutil.copy2(src_path, filepath)
        except Exception as e:
            print(f"          {c_red('✗ error copying to temp/:')} {e}")
            logger.error(f"Could not copy {src_path} to temp/: {e}")
            cnt["errors"] += 1
            progress["error"].append(fname)
            save_checkpoint(progress)
            _halt_pipeline(fname, f"copy to temp/ failed: {e}")
            continue

        status, detail = process_file(filepath)

        if status == "ok":
            try:
                from mutagen.oggvorbis import OggVorbis
                audio = OggVorbis(filepath)
                lyrics = audio.get("lyrics", [""])[0]
                snippet = lyrics[:60].replace('\n', ' ') + "..." if len(lyrics) > 60 else lyrics.replace('\n', ' ')
                print(f"          {c_green('✓')} \"{snippet}\"  {c_dim(f'[{detail}]')}")
            except Exception:
                print(f"          {c_green('✓ lyrics written')}  {c_dim(f'[{detail}]')}")

            try:
                shutil.move(filepath, str(OUTPUT_DIR / fname))
                print(f"          {c_dim('↳ moved to ' + str(OUTPUT_DIR))}")
            except Exception as e:
                logger.error(f"Could not move {fname} from temp/ to OUTPUT_DIR: {e}")
                print(f"          {c_red('✗ failed to move into output dir:')} {e}")

            cnt["ok"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)
            current_done.add(fname)
        elif status == "instrumental":
            print(f"          {c_yellow('⚠ instrumental — skipped')}")
            move_to_failed(filepath, "instrumental")
            cnt["instrumental"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)    # final — no retry needed
            current_done.add(fname)
        elif status == "no_lyrics":
            print(f"          {c_yellow('⚠ no lyrics found')}")
            move_to_failed(filepath, "no_lyrics")
            cnt["no_lyrics"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)    # final for THIS script — local_lyrics_tagger gets the next try
            current_done.add(fname)
        elif status == "network_error":
            print(f"          {c_yellow('⚠ network error — all sources unreachable')}")
            move_to_failed(filepath, "network_error")
            cnt["network_error"] += 1
            progress["miss"].append(fname)    # retry once fixed
            save_checkpoint(progress)
            _halt_pipeline(fname, "network error — all sources unreachable")
        elif status.startswith("error"):
            print(f"          {c_red('✗ error:')} {detail}")
            move_to_failed(filepath, "error")
            cnt["errors"] += 1
            progress["error"].append(fname)   # retry once fixed
            save_checkpoint(progress)
            _halt_pipeline(fname, detail)
        else:
            print(f"          {c_red('✗ unknown status:')} {status} ({detail})")
            cnt["errors"] += 1
            progress["error"].append(fname)   # retry once fixed
            save_checkpoint(progress)
            _halt_pipeline(fname, f"unknown status: {status} ({detail})")

        # Save checkpoint after every file
        save_checkpoint(progress)

    save_checkpoint(progress)

    banner([
        f"{c_bold('✓ Plain Lyrics Tagger — session complete')}",
        f"Lyrics written   : {cnt['ok']}",
        f"Instrumental     : {cnt['instrumental']}",
        f"Network errors   : {cnt['network_error']}",
        f"No lyrics found  : {cnt['no_lyrics']}",
        f"Errors           : {cnt['errors']}",
        f"Skipped (done)   : {cnt['skipped']}",
    ])

    sign_off = "smooth run, nice work (｡•̀ᴗ-)✧" if cnt["errors"] == 0 and cnt["network_error"] == 0 \
        else "a few hiccups — worth a peek at the log (._.)"
    print(f"\n  {c_dim(sign_off)}\n")

    logger.debug(f"Run finished: {cnt}")

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