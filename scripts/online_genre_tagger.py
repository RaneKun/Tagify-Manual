"""
🌐 Genre Tagger — Online Edition (multi-source)
─────────────────────────────────────────────────────────────────────────────
Fetches genre tags from multiple internet sources, merges + scores them, and
writes up to TOP_N_GENRES values into the GENRE Vorbis comment tag.

Sources (every ENABLED source is queried for every song):
  1. Last.fm     — Free API, no app review needed. track.getTopTags first,
                   artist.getTopTags always added too (cached). Best
                   general-purpose folksonomy coverage for a library this
                   varied.
  2. iTunes      — Free REST search, zero auth. Apple's fixed taxonomy
                   (K-Pop, J-Pop, Anime, World, Latino, Reggaeton y Flow…)
                   is a clean cross-check against Last.fm's messier tags.
  3. MusicBrainz — Free, no key (just an honest User-Agent). Community tag
                   data; the best fallback for tiny/independent producers
                   that Last.fm has no scrobble history for. The service
                   itself enforces 1 request/second — this is the slowest
                   source, which is why it's the only one with a
                   meaningful per-file time cost. Still on by default
                   because you said accuracy beats time-taken.

Checkpoint behaviour:
  ▸ Files listed in the checkpoint JSON under "done" are ALWAYS skipped.
  ▸ Files NOT in "done" are processed (and the GENRES tag overwritten)
    regardless of whether a tag already exists.
  ▸ "miss", "skip", "error" are NOT used for skipping — they are informational.
  ▸ Delete or clear the checkpoint file to reprocess everything.

Caching: artist-level lookups (Last.fm artist tags, MusicBrainz) are cached
to logs/genre_online_artist_cache.json so re-running the script never
re-spends a network call on an artist already looked up. This matters a lot
with many repeat artists in a library — the number of *unique* artists is
far smaller than the sheer number of tracks, so the cache does most of the
work.

Output locations:
  ▸ Log file     → logs/genre_online.log                (everything, DEBUG level)
  ▸ Checkpoint   → logs/genre_online_checkpoint.json
  ▸ Artist cache → logs/genre_online_artist_cache.json

Before first run: edit the CONFIG block below to point at your music folder.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
import textwrap
from typing import Optional
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from mutagen.oggvorbis import OggVorbis

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOOK & FEEL  ─  colors + tiny console helpers                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

THEME        = BLUE   # ← this script's signature color
SCRIPT_EMOJI = "🌐"    # online (multi-source) genre tagger's signature icon

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
# ║  DIRECTORY & LOGGING SETUP                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SCRIPT_DIR    = Path(__file__).parent.resolve()
LOGS_DIR      = SCRIPT_DIR.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

class ImmediateFileHandler(logging.FileHandler):
    """A FileHandler that flushes after every single record — crash-safe logs."""
    def emit(self, record):
        super().emit(record)
        self.flush()

_fmt    = "%(asctime)s │ %(levelname)-8s │ %(message)s"
_file_h = ImmediateFileHandler(LOGS_DIR / "genre_online.log", encoding="utf-8", mode="w")
_file_h.setLevel(logging.DEBUG)

# ── Console handler: block HTTP Request:, Moved failed, and network error messages ──
class _SuppressNoise(logging.Filter):
    """Reject any log record whose message contains these phrases."""
    _BLOCK_PHRASES = (
        "HTTP Request:",
        "Moved failed file to",
        "Catalogued failed file to",   # added to hide the new file‑cataloguing log
        "Network error",
    )
    def filter(self, record):
        msg = record.getMessage()
        return not any(phrase in msg for phrase in self._BLOCK_PHRASES)

_console_h = logging.StreamHandler(sys.stdout)
_console_h.setLevel(logging.INFO)
_console_h.addFilter(_SuppressNoise())

# Apply both handlers to the root logger
logging.basicConfig(level=logging.DEBUG, format=_fmt, handlers=[_file_h, _console_h])

# Also crank down the noisy libraries as a belt-and-suspenders measure
for lib in ('requests', 'urllib3', 'pylast', 'musicbrainzngs',
            'urllib3.connectionpool', 'musicbrainzngs.mbxml'):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG  ─  input/output dirs + API key entered interactively at start   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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


def _prompt_for_credential(label: str, default: Optional[str] = None) -> str:
    """Ask the user to type a required credential value; pressing Enter reuses default if provided."""
    while True:
        prompt = f"[?] Enter {label}"
        if default is not None:
            prompt += f" (default: [saved])"
        prompt += ": "
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        if not raw:
            print("[!] This field cannot be empty. Try again.")
            continue
        return raw


# Save this run's chosen paths + credential under configs/ for the record.
CONFIG_FILE = CONFIGS_DIR / "online_genre_tagger.json"

# ── Input / output folders — chosen manually at startup, both mandatory ──────
print()

default_input = default_output = None
default_lastfm = None
if CONFIG_FILE.exists():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        p = Path(data.get("input_directory", ""))
        if p.is_dir():
            default_input = p
        p = Path(data.get("output_directory", ""))
        default_output = p
        default_lastfm = data.get("lastfm_api_key")
    except Exception:
        pass

INPUT_DIR  = _prompt_for_directory("input folder of OGG files to read (read-only, never modified)", must_exist=True, default=default_input)
default_output = default_output if default_output is not None else PARENT_DIR / "outputs"
OUTPUT_DIR = _prompt_for_directory("output folder for finished, tagged files", must_exist=False, default=default_output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print()

# ── Credentials — this script needs a Last.fm API key ────────────────────────
LASTFM_API_KEY: str = _prompt_for_credential("your Last.fm API key", default=default_lastfm)
print()

# Save config for next run
CONFIG_FILE.write_text(
    json.dumps({
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "lastfm_api_key": LASTFM_API_KEY,
    }, indent=2),
    encoding="utf-8"
)

# The source in INPUT_DIR is read-only and never modified. Every file gets
# copied into temp/ first, tagged there, then the finished copy is moved
# into OUTPUT_DIR.
MUSIC_DIR: str = str(INPUT_DIR)

CHECKPOINT_FILE:      Path = LOGS_DIR / "genre_online_checkpoint.json"
ARTIST_CACHE_FILE:    Path = LOGS_DIR / "genre_online_artist_cache.json"

ENABLE_LASTFM:        bool = True
ENABLE_ITUNES:        bool = True
ENABLE_MUSICBRAINZ:   bool = True

LASTFM_TRACK_TAG_LIMIT:  int   = 10
LASTFM_ARTIST_TAG_LIMIT: int   = 10
LASTFM_DELAY:            float = 0.25

ITUNES_SEARCH_URL:   str   = "https://itunes.apple.com/search"
ITUNES_DELAY:        float = 2.0
ITUNES_TIMEOUT:       int  = 15
ITUNES_RESULT_LIMIT:  int  = 5
ITUNES_RESULTS_USED:  int  = 3

MUSICBRAINZ_APP_NAME:    str = "RaneKunGenreTagger"
MUSICBRAINZ_APP_VERSION: str = "1.0"
MUSICBRAINZ_CONTACT:     str = "not-provided@example.com"

TOP_N_GENRES:          int   = 3
MAX_TAGS_PER_FAMILY:   int   = 2
GENERIC_TAG_PENALTY:   float = 0.55

WEIGHT_LASTFM_TRACK:  float = 2.0
WEIGHT_LASTFM_ARTIST: float = 1.4
WEIGHT_ITUNES:        float = 1.8
WEIGHT_MUSICBRAINZ:   float = 1.5

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

JUNK_TAGS: frozenset[str] = frozenset({
    "seen live", "favorites", "favourite", "love", "awesome", "good", "cool",
    "beautiful", "amazing", "best", "classic", "relax", "party", "workout",
    "running", "sleep", "study", "sad", "happy", "sexy", "chill",
    "under 2000 listeners", "all", "spotify", "youtube", "downloaded", "owned",
    "wishlist", "recommended", "favourite albums", "albums i own", "music",
    "songs", "tracks", "female vocalists", "male vocalists", "vocal",
    "vocalists", "singer", "instrumental", "cover", "remix", "live", "demo",
    "edit", "english", "japanese", "korean", "chinese", "spanish", "german",
    "french", "turkish", "portuguese", "brazilian", "comedy", "children's music",
})

GENERIC_TAGS: frozenset[str] = frozenset({
    "edm", "electronic", "electronica", "dance", "rock", "alternative",
    "alternative rock", "pop",
})

GENRE_VOCAB: dict[str, tuple[str, str]] = {
    "edm":         ("EDM", "Electronic-Generic"),
    "electronic":  ("Electronic", "Electronic-Generic"),
    "electronica": ("Electronica", "Electronic-Generic"),
    "dance":       ("Dance", "Electronic-Generic"),
    "house":              ("House", "House"),
    "deep house":         ("Deep House", "House"),
    "tropical house":     ("Tropical House", "House"),
    "future house":       ("Future House", "House"),
    "bass house":         ("Bass House", "House"),
    "electro house":      ("Electro House", "House"),
    "complextro":         ("Complextro", "House"),
    "big room":           ("Big Room", "Big Room"),
    "techno":             ("Techno", "Techno"),
    "trance":             ("Trance", "Trance"),
    "progressive trance": ("Progressive Trance", "Trance"),
    "psytrance":          ("Psytrance", "Trance"),
    "dubstep":            ("Dubstep", "Dubstep"),
    "riddim":             ("Riddim", "Dubstep"),
    "future bass":        ("Future Bass", "Future Bass"),
    "trap":               ("Trap", "Trap"),
    "trap rap":           ("Trap", "Trap"),
    "drum and bass":      ("Drum & Bass", "Drum & Bass"),
    "drum & bass":        ("Drum & Bass", "Drum & Bass"),
    "dnb":                ("Drum & Bass", "Drum & Bass"),
    "jungle":             ("Jungle", "Drum & Bass"),
    "hardstyle":          ("Hardstyle", "Hardstyle"),
    "hardcore":           ("Hardcore", "Hardstyle"),
    "speedcore":          ("Speedcore", "Hardstyle"),
    "gabber":             ("Gabber", "Hardstyle"),
    "moombahton":         ("Moombahton", "Moombahton"),
    "glitch":             ("Glitch", "Experimental"),
    "idm":                ("IDM", "Experimental"),
    "breakcore":          ("Breakcore", "Experimental"),
    "experimental":       ("Experimental", "Experimental"),
    "ambient":            ("Ambient", "Ambient"),
    "new age":            ("New Age", "Ambient"),
    "chillout":           ("Chillout", "Downtempo"),
    "downtempo":          ("Downtempo", "Downtempo"),
    "synthwave":          ("Synthwave", "Synthwave"),
    "vaporwave":          ("Vaporwave", "Synthwave"),
    "lo-fi":              ("Lo-Fi", "Lo-Fi"),
    "lofi":               ("Lo-Fi", "Lo-Fi"),
    "chillhop":           ("Lo-Fi", "Lo-Fi"),
    "nightcore": ("Nightcore", "Nightcore"),
    "speed up":  ("Sped Up", "Nightcore"),
    "sped up":   ("Sped Up", "Nightcore"),
    "phonk":            ("Phonk", "Phonk"),
    "drift phonk":       ("Drift Phonk", "Phonk"),
    "brazilian phonk":   ("Brazilian Phonk", "Phonk"),
    "phonk brasileiro":  ("Brazilian Phonk", "Phonk"),
    "brazilian funk":  ("Brazilian Funk", "Brazilian Funk"),
    "funk carioca":    ("Brazilian Funk", "Brazilian Funk"),
    "baile funk":      ("Brazilian Funk", "Brazilian Funk"),
    "funk brasileiro":  ("Brazilian Funk", "Brazilian Funk"),
    "montagem":        ("Brazilian Funk", "Brazilian Funk"),
    "hip-hop":           ("Hip-Hop", "Hip-Hop"),
    "hip hop":           ("Hip-Hop", "Hip-Hop"),
    "hiphop":            ("Hip-Hop", "Hip-Hop"),
    "hip-hop/rap":       ("Hip-Hop", "Hip-Hop"),
    "rap":               ("Rap", "Hip-Hop"),
    "drill":             ("Drill", "Hip-Hop"),
    "gangsta rap":       ("Gangsta Rap", "Hip-Hop"),
    "conscious hip hop": ("Conscious Hip-Hop", "Hip-Hop"),
    "boom bap":          ("Boom Bap", "Hip-Hop"),
    "r&b":              ("R&B", "R&B"),
    "rnb":              ("R&B", "R&B"),
    "r&b/soul":         ("R&B", "R&B"),
    "rhythm and blues": ("R&B", "R&B"),
    "soul":             ("Soul", "Soul"),
    "neo soul":         ("Neo-Soul", "Soul"),
    "funk":             ("Funk", "Soul"),
    "rock":             ("Rock", "Rock-Generic"),
    "alternative":      ("Alternative", "Rock-Generic"),
    "alternative rock": ("Alternative", "Rock-Generic"),
    "indie":            ("Indie", "Indie"),
    "indie rock":       ("Indie", "Indie"),
    "indie pop":        ("Indie Pop", "Indie"),
    "punk":             ("Punk", "Punk"),
    "pop punk":         ("Pop Punk", "Punk"),
    "emo":              ("Emo", "Punk"),
    "metal":            ("Metal", "Metal"),
    "heavy metal":      ("Metal", "Metal"),
    "metalcore":        ("Metalcore", "Metal"),
    "hard rock":        ("Hard Rock", "Hard Rock"),
    "classic rock":     ("Classic Rock", "Hard Rock"),
    "progressive rock": ("Progressive Rock", "Hard Rock"),
    "post-rock":        ("Post-Rock", "Post-Rock"),
    "pop":         ("Pop", "Pop-Generic"),
    "synth-pop":   ("Synth-Pop", "Pop-Specific"),
    "synth pop":   ("Synth-Pop", "Pop-Specific"),
    "synthpop":    ("Synth-Pop", "Pop-Specific"),
    "dance pop":   ("Dance Pop", "Pop-Specific"),
    "electropop":  ("Electropop", "Pop-Specific"),
    "power pop":   ("Power Pop", "Pop-Specific"),
    "bedroom pop": ("Bedroom Pop", "Pop-Specific"),
    "city pop":    ("City Pop", "City Pop"),
    "k-pop":            ("K-Pop", "K-Pop"),
    "kpop":             ("K-Pop", "K-Pop"),
    "korean pop":       ("K-Pop", "K-Pop"),
    "j-pop":            ("J-Pop", "J-Pop"),
    "jpop":             ("J-Pop", "J-Pop"),
    "japanese pop":     ("J-Pop", "J-Pop"),
    "j-rock":           ("J-Rock", "J-Rock"),
    "jrock":            ("J-Rock", "J-Rock"),
    "japanese rock":    ("J-Rock", "J-Rock"),
    "visual kei":       ("Visual Kei", "J-Rock"),
    "c-pop":            ("C-Pop", "C-Pop"),
    "mandopop":         ("C-Pop", "C-Pop"),
    "cantopop":         ("C-Pop", "C-Pop"),
    "vocaloid":         ("Vocaloid", "Vocaloid"),
    "utaite":           ("Vocaloid", "Vocaloid"),
    "denpa":            ("Vocaloid", "Vocaloid"),
    "anime":            ("Anime", "Anime"),
    "anisong":          ("Anime", "Anime"),
    "vtuber":           ("Vtuber", "Anime"),
    "video game music": ("Video Game Music", "Soundtrack"),
    "vgm":              ("Video Game Music", "Soundtrack"),
    "soundtrack":       ("Soundtrack", "Soundtrack"),
    "score":            ("Soundtrack", "Soundtrack"),
    "ost":              ("Soundtrack", "Soundtrack"),
    "bollywood":    ("Bollywood", "Bollywood"),
    "desi":         ("Desi Pop", "Bollywood"),
    "desi hip hop": ("Desi Hip-Hop", "Bollywood"),
    "desi pop":     ("Desi Pop", "Bollywood"),
    "punjabi":      ("Punjabi", "Bollywood"),
    "indian pop":   ("Indian Pop", "Bollywood"),
    "filmi":        ("Filmi", "Bollywood"),
    "bhangra":      ("Bhangra", "Bollywood"),
    "turkish pop": ("Turkish Pop", "Turkish"),
    "türkçe pop":  ("Turkish Pop", "Turkish"),
    "arabesk":     ("Arabesk", "Turkish"),
    "latin":            ("Latin", "Latin"),
    "latin pop":        ("Latin Pop", "Latin"),
    "latin trap":       ("Latin Trap", "Latin"),
    "reggaeton":        ("Reggaeton", "Latin"),
    "reggaeton y flow": ("Reggaeton", "Latin"),
    "reggae":           ("Reggae", "Reggae"),
    "dancehall":        ("Dancehall", "Reggae"),
    "acoustic":           ("Acoustic", "Folk"),
    "folk":               ("Folk", "Folk"),
    "singer-songwriter":  ("Singer-Songwriter", "Folk"),
    "singer/songwriter":  ("Singer-Songwriter", "Folk"),
    "easy listening":     ("Easy Listening", "Folk"),
    "world":              ("World", "World"),
    "classical":          ("Classical", "Classical"),
    "jazz":               ("Jazz", "Jazz"),
    "blues":              ("Blues", "Blues"),
    "country":            ("Country", "Country"),
}

# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  SENTINEL for network errors
# ─────────────────────────────────────────────────────────────────────────────
_NETWORK_ERROR = object()

def _is_network_error(exc: Exception) -> bool:
    """Return True if the exception indicates a network/connectivity problem."""
    keywords = (
        "connection", "timeout", "resolve", "refused",
        "network", "dns", "name resolution", "reset",
    )
    return any(kw in str(exc).lower() for kw in keywords)

# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT  (dict with "done", "miss", "skip", "error")
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA = "genre-online-checkpoint"
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
                    "Tracks per-file online genre-tagging progress. Only "
                    "files under 'done' are skipped on the next run."
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
#  PERSISTENT ARTIST CACHE
# ─────────────────────────────────────────────────────────────────────────────

ARTIST_CACHE_SCHEMA = "genre-online-artist-cache"

def load_artist_cache() -> dict:
    """
    Return the cached per-artist tag lookups, keyed by source
    ("lastfm", "musicbrainz"). A top-level "_meta" key (if present) is
    informational only and is skipped when rebuilding the tuple lists.
    """
    if ARTIST_CACHE_FILE.exists():
        try:
            data = json.loads(ARTIST_CACHE_FILE.read_text(encoding="utf-8"))
            data.pop("_meta", None)
            for source in data:
                for artist_key, tag_list in data[source].items():
                    data[source][artist_key] = [tuple(t) for t in tag_list]
            logger.debug(f"Artist cache loaded — sources: {list(data.keys())}")
            return data
        except Exception as exc:
            logger.warning(f"Artist cache load failed ({exc}); starting fresh")
    return {}

def save_artist_cache(cache: dict) -> None:
    """
    Persist the artist cache as clean JSON, with a small "_meta" header
    (schema + last-updated + per-source artist counts) for readability.
    load_artist_cache() strips "_meta" back out before use, so this never
    affects lookup behaviour.
    """
    try:
        meta = {
            "schema": ARTIST_CACHE_SCHEMA,
            "description": "Per-artist tag lookups cached across runs to save network calls.",
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "artists_cached": {source: len(artists) for source, artists in cache.items()},
        }
        ordered = {"_meta": meta, **cache}
        ARTIST_CACHE_FILE.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.error(f"Artist cache save failed: {exc}")

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

def get_genre_search_title(artist: str, title: str) -> str:
    variants = get_search_variants(artist, title)
    return variants[-1][1]

# ─────────────────────────────────────────────────────────────────────────────
#  SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────────────────────────────────────
#  GENRE NORMALISATION + FAMILY-AWARE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_genre_tag(raw: str, base_weight: float) -> Optional[tuple[str, str, float]]:
    if not raw:
        return None
    key = raw.strip().lower()
    if not key or len(key) <= 1:
        return None
    if key in JUNK_TAGS:
        return None
    if key.isdigit():
        return None
    if re.fullmatch(r"(19|20)\d{2}s?", key):
        return None

    if key in GENRE_VOCAB:
        display, family = GENRE_VOCAB[key]
    else:
        display = raw.strip().title()
        family = display

    score = base_weight
    if key in GENERIC_TAGS:
        score *= GENERIC_TAG_PENALTY
    return display, family, score

def aggregate_and_select(candidates: list[tuple[str, float]]) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    family_of: dict[str, str] = {}

    for raw, weight in candidates:
        normalized = normalize_genre_tag(raw, weight)
        if normalized is None:
            continue
        display, family, score = normalized
        scores[display] = scores.get(display, 0.0) + score
        family_of[display] = family

    if not scores:
        return "", {}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    chosen: list[str] = []
    family_count: dict[str, int] = {}

    for display, _score in ranked:
        fam = family_of[display]
        if family_count.get(fam, 0) > 0:
            continue
        chosen.append(display)
        family_count[fam] = family_count.get(fam, 0) + 1
        if len(chosen) >= TOP_N_GENRES:
            break

    if len(chosen) < TOP_N_GENRES:
        for display, _score in ranked:
            if len(chosen) >= TOP_N_GENRES:
                break
            if display in chosen:
                continue
            fam = family_of[display]
            if family_count.get(fam, 0) >= MAX_TAGS_PER_FAMILY:
                continue
            chosen.append(display)
            family_count[fam] = family_count.get(fam, 0) + 1

    return ", ".join(chosen), scores

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 1: Last.fm
# ─────────────────────────────────────────────────────────────────────────────

_lastfm_client: Optional[object] = None
_lastfm_init_failed: bool = False

def _get_lastfm() -> Optional[object]:
    global _lastfm_client, _lastfm_init_failed
    if _lastfm_client is not None or _lastfm_init_failed:
        return _lastfm_client
    try:
        import pylast  # type: ignore
        _lastfm_client = pylast.LastFMNetwork(api_key=LASTFM_API_KEY)
        logger.debug("Last.fm client initialised")
    except ImportError:
        logger.debug("pylast not installed — Last.fm source unavailable")
        _lastfm_init_failed = True
    except Exception as exc:
        logger.warning(f"Last.fm client init failed: {exc}")
        _lastfm_init_failed = True
    return _lastfm_client

def fetch_lastfm_tags(artist: str, title: str, artist_cache: dict) -> object:
    """Returns list of (tag, weight) tuples, or _NETWORK_ERROR if connectivity fails."""
    if not ENABLE_LASTFM or not artist.strip():
        return []
    network = _get_lastfm()
    if network is None:
        return []

    results: list[tuple[str, float]] = []
    had_network_error = False

    try:
        track_obj = network.get_track(artist, title)
        for item in track_obj.get_top_tags(limit=LASTFM_TRACK_TAG_LIMIT):
            w = _safe_float(item.weight, default=10.0) / 100.0
            results.append((item.item.name, w * WEIGHT_LASTFM_TRACK))
    except Exception as exc:
        logger.debug(f"Last.fm track.getTopTags failed for '{artist} - {title}': {exc}")
        if _is_network_error(exc):
            had_network_error = True
    time.sleep(LASTFM_DELAY)

    cache_key = artist.lower().strip()
    lastfm_cache = artist_cache.setdefault("lastfm", {})
    if cache_key in lastfm_cache:
        artist_tags = lastfm_cache[cache_key]
    else:
        artist_tags = []
        try:
            artist_obj = network.get_artist(artist)
            for item in artist_obj.get_top_tags(limit=LASTFM_ARTIST_TAG_LIMIT):
                w = _safe_float(item.weight, default=10.0) / 100.0
                artist_tags.append((item.item.name, w))
        except Exception as exc:
            logger.debug(f"Last.fm artist.getTopTags failed for '{artist}': {exc}")
            if _is_network_error(exc):
                had_network_error = True
        time.sleep(LASTFM_DELAY)
        lastfm_cache[cache_key] = artist_tags

    if had_network_error and not results and not artist_tags:
        return _NETWORK_ERROR

    results.extend((name, w * WEIGHT_LASTFM_ARTIST) for name, w in artist_tags)
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 2: iTunes
# ─────────────────────────────────────────────────────────────────────────────

_ITUNES_SESSION = requests.Session()
_ITUNES_SESSION.headers.update({
    "User-Agent": "MusicTagger/1.0 (personal library; not commercial)",
})
_ITUNES_RANK_DECAY: list[float] = [1.0, 0.55, 0.3]

def fetch_itunes_genres(artist: str, title: str) -> object:
    """Returns list of (genre, weight) tuples, or _NETWORK_ERROR."""
    if not ENABLE_ITUNES:
        return []
    term = f"{artist} {title}".strip()
    if not term:
        return []

    try:
        resp = _ITUNES_SESSION.get(
            ITUNES_SEARCH_URL,
            params={
                "term": term,
                "media": "music",
                "entity": "song",
                "limit": ITUNES_RESULT_LIMIT,
            },
            timeout=ITUNES_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug(f"iTunes search failed for '{term}': {exc}")
        time.sleep(ITUNES_DELAY)
        if _is_network_error(exc):
            return _NETWORK_ERROR
        return []

    items = data.get("results", [])
    artist_lower = artist.lower().strip()
    ranked_items = sorted(
        items,
        key=lambda it: 0 if artist_lower and artist_lower in str(it.get("artistName", "")).lower() else 1,
    )

    results: list[tuple[str, float]] = []
    for idx, item in enumerate(ranked_items[:ITUNES_RESULTS_USED]):
        genre = item.get("primaryGenreName")
        if genre:
            decay = _ITUNES_RANK_DECAY[idx] if idx < len(_ITUNES_RANK_DECAY) else 0.2
            results.append((genre, decay * WEIGHT_ITUNES))

    time.sleep(ITUNES_DELAY)
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 3: MusicBrainz
# ─────────────────────────────────────────────────────────────────────────────

_mb_initialised: bool = False
_mb_init_failed: bool = False

def _init_musicbrainz() -> bool:
    global _mb_initialised, _mb_init_failed
    if _mb_initialised or _mb_init_failed:
        return _mb_initialised
    try:
        import musicbrainzngs as mb  # type: ignore
        mb.set_useragent(MUSICBRAINZ_APP_NAME, MUSICBRAINZ_APP_VERSION, MUSICBRAINZ_CONTACT)
        _mb_initialised = True
        logger.debug("MusicBrainz client initialised")
    except ImportError:
        logger.debug("musicbrainzngs not installed — MusicBrainz source unavailable")
        _mb_init_failed = True
    except Exception as exc:
        logger.warning(f"MusicBrainz init failed: {exc}")
        _mb_init_failed = True
    return _mb_initialised

def fetch_musicbrainz_tags(artist: str, artist_cache: dict) -> object:
    """Returns list of (tag, weight) tuples, or _NETWORK_ERROR."""
    if not ENABLE_MUSICBRAINZ or not artist.strip():
        return []
    if not _init_musicbrainz():
        return []

    cache_key = artist.lower().strip()
    mb_cache = artist_cache.setdefault("musicbrainz", {})
    if cache_key in mb_cache:
        return [(name, w * WEIGHT_MUSICBRAINZ) for name, w in mb_cache[cache_key]]

    import musicbrainzngs as mb

    tag_results: list[tuple[str, float]] = []
    try:
        search = mb.search_artists(query=artist, limit=3)
        candidates = search.get("artist-list", [])

        best = None
        for cand in candidates:
            if cand.get("name", "").lower() == artist.lower():
                best = cand
                break
        if best is None and candidates:
            best = candidates[0]

        if best is not None:
            detail = mb.get_artist_by_id(best["id"], includes=["tags"])
            for t in detail.get("artist", {}).get("tag-list", []):
                name = t.get("name", "")
                if name:
                    count = _safe_float(t.get("count"), default=1.0)
                    tag_results.append((name, min(count, 10.0) / 10.0))
    except Exception as exc:
        logger.debug(f"MusicBrainz lookup failed for '{artist}': {exc}")
        if _is_network_error(exc):
            return _NETWORK_ERROR

    mb_cache[cache_key] = tag_results
    return [(name, w * WEIGHT_MUSICBRAINZ) for name, w in tag_results]

# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-SOURCE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def fetch_genre_all_sources(artist: str, title: str, artist_cache: dict) -> tuple[object, list[str]]:
    search_title = get_genre_search_title(artist, title)

    all_candidates: list[tuple[str, float]] = []
    sources_hit: list[str] = []
    had_network_error = False

    if ENABLE_LASTFM:
        lf = fetch_lastfm_tags(artist, search_title, artist_cache)
        if lf is _NETWORK_ERROR:
            had_network_error = True
        elif lf:
            sources_hit.append("Last.fm")
            all_candidates.extend(lf)

    if ENABLE_ITUNES:
        it = fetch_itunes_genres(artist, search_title)
        if it is _NETWORK_ERROR:
            had_network_error = True
        elif it:
            sources_hit.append("iTunes")
            all_candidates.extend(it)

    if ENABLE_MUSICBRAINZ:
        mbt = fetch_musicbrainz_tags(artist, artist_cache)
        if mbt is _NETWORK_ERROR:
            had_network_error = True
        elif mbt:
            sources_hit.append("MusicBrainz")
            all_candidates.extend(mbt)

    genre_string, scores = aggregate_and_select(all_candidates)
    if scores:
        logger.debug(f"Score breakdown for '{artist} - {title}': {scores}")

    if not genre_string and had_network_error:
        return _NETWORK_ERROR, sources_hit

    return genre_string, sources_hit

# ─────────────────────────────────────────────────────────────────────────────
#  PER-FILE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_file(filepath: str, artist_cache: dict) -> tuple[str, str]:
    artist, title = parse_filename(filepath)
    logger.debug(f"Looking up genre: '{artist}' - '{title}'")

    try:
        genre_string, sources_hit = fetch_genre_all_sources(artist, title, artist_cache)
    except Exception as exc:
        logger.exception(f"Unexpected error fetching genre for {filepath}")
        return "error", str(exc)

    if genre_string is _NETWORK_ERROR:
        logger.debug(f"Network error (all sources unreachable): '{artist}' - '{title}'")
        return "network_error", "all sources network error"

    if not genre_string:
        logger.debug(f"No genre found: '{artist}' - '{title}' (sources hit: {sources_hit or 'none'})")
        return "no_genre", "all sources returned nothing usable"

    try:
        tags = OggVorbis(filepath)
        tags["genre"] = [genre_string]
        tags.save()
    except Exception as exc:
        logger.error(f"Tag write failed for {filepath}: {exc}")
        return "error", f"write_tag: {exc}"

    logger.debug(f"ok [{'+'.join(sources_hit)}] — {Path(filepath).name}: {genre_string}")
    return "ok", genre_string

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
        logger.info(f"Catalogued failed file to {dst}")   # filtered from console
        print(f"          {c_dim('↳ catalogued under failed/' + error_type + '/')}")
    except Exception as e:
        logger.error(f"Failed to catalog {src} to {dst}: {e}")


def _halt_pipeline(fname: str, detail: str) -> None:
    """A real error (not a soft 'no genre' miss) happened — stop the WHOLE
    pipeline here for manual review instead of silently moving to the next file."""
    print(f"\n{c_red('HALTED —')} a real error hit '{fname}'; stopping the whole pipeline for manual review.")
    print(f"  {c_dim('(no-genre misses never halt — only network/technical errors do)')}")
    print(f"  {c_dim('Fix the underlying issue, then re-run run_tagger — already-tagged files are skipped automatically.')}")
    logger.critical(f"HALTED — {fname}: {detail}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.debug("═" * 72)
    logger.debug(f"{SCRIPT_EMOJI} Genre Tagger (online, multi-source) — session starting")
    global ENABLE_LASTFM, ENABLE_MUSICBRAINZ

    # Dependency checks
    try:
        import mutagen  # noqa: F401
    except ImportError:
        sys.exit("[!] mutagen is not installed.\n    pip install mutagen")
    try:
        import requests as _r  # noqa: F401
    except ImportError:
        sys.exit("[!] requests is not installed.\n    pip install requests")

    if ENABLE_LASTFM:
        try:
            import pylast  # noqa: F401
        except ImportError:
            logger.warning(
                "pylast not installed — disabling Last.fm source.\n"
                "  Install: pip install pylast"
            )
            ENABLE_LASTFM = False

    if ENABLE_LASTFM and not LASTFM_API_KEY:
        logger.warning("Last.fm API key is empty — disabling Last.fm source.")
        print(f"  {c_yellow('[WARN]')} Last.fm API key is empty — Last.fm source disabled for this run.")
        ENABLE_LASTFM = False

    if ENABLE_MUSICBRAINZ:
        try:
            import musicbrainzngs  # noqa: F401
        except ImportError:
            logger.warning(
                "musicbrainzngs not installed — disabling MusicBrainz source.\n"
                "  Install: pip install musicbrainzngs"
            )
            ENABLE_MUSICBRAINZ = False

    if not (ENABLE_LASTFM or ENABLE_ITUNES or ENABLE_MUSICBRAINZ):
        sys.exit("[!] No genre sources are enabled/available — nothing to do.")

    if not os.path.isdir(MUSIC_DIR):
        sys.exit(f"[!] MUSIC_DIR not found:\n    {MUSIC_DIR}")

    all_ogg = sorted(
        os.path.join(MUSIC_DIR, f)
        for f in os.listdir(MUSIC_DIR)
        if f.lower().endswith(".ogg") and os.path.isfile(os.path.join(MUSIC_DIR, f))
    )
    total = len(all_ogg)
    if total == 0:
        sys.exit(f"[!] No .ogg files found in:\n    {MUSIC_DIR}")

    done_set, progress = load_checkpoint()
    current_done = {os.path.basename(f) for f in all_ogg if os.path.basename(f) in done_set}
    for key in ("error", "miss", "skip"):
        progress[key] = [f for f in progress[key] if f not in current_done]

    resumed = len(current_done)
    artist_cache = load_artist_cache()

    active_sources: list[str] = []
    if ENABLE_LASTFM:
        active_sources.append("Last.fm")
    if ENABLE_ITUNES:
        active_sources.append("iTunes")
    if ENABLE_MUSICBRAINZ:
        active_sources.append("MusicBrainz")

    banner([
        f"{c_bold(SCRIPT_EMOJI + '  Genre Tagger')}  {c_dim('· online · multi-source')}",
        f"Directory       : {MUSIC_DIR}",
        f"Files           : {total}",
        f"Already done    : {resumed}",
        f"Active sources  : {' + '.join(active_sources)}",
        f"Max genres/song : {TOP_N_GENRES}  (max {MAX_TAGS_PER_FAMILY} per family)",
    ])
    print()

    logger.debug(f"Run started — total={total}, resumed={resumed}, sources={active_sources}")

    cnt: dict[str, int] = {"ok": 0, "no_genre": 0, "network_error": 0, "errors": 0, "skipped": 0}
    genre_counter: dict[str, int] = {}
    used_genres = set()
    all_genre_display = {display for display, _ in GENRE_VOCAB.values()}
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
        working_path = str(TEMP_DIR / fname)
        try:
            shutil.copy2(src_path, working_path)
        except Exception as e:
            print(f"          {c_red('✗ error copying to temp/:')} {e}")
            logger.error(f"Could not copy {src_path} to temp/: {e}")
            cnt["errors"] += 1
            progress["error"].append(fname)
            save_checkpoint(progress)
            save_artist_cache(artist_cache)
            _halt_pipeline(fname, f"copy to temp/ failed: {e}")
            continue

        status, detail = process_file(working_path, artist_cache)

        if status == "ok":
            print(f"          {c_green('✓')} {detail}")
            try:
                shutil.move(working_path, str(OUTPUT_DIR / fname))
                print(f"          {c_dim('↳ moved to ' + str(OUTPUT_DIR))}")
            except Exception as e:
                logger.error(f"Could not move {fname} from temp/ to OUTPUT_DIR: {e}")
                print(f"          {c_red('✗ failed to move into output dir:')} {e}")
            cnt["ok"] += 1
            for tag in detail.split(", "):
                genre_counter[tag] = genre_counter.get(tag, 0) + 1
                used_genres.add(tag)
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)
            current_done.add(fname)
        elif status == "no_genre":
            print(f"          {c_yellow('⚠ no genre found')}")
            move_to_failed(working_path, "no_genre")
            cnt["no_genre"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)   # final for THIS script — local_genre_tagger gets the next try
            current_done.add(fname)
        elif status == "network_error":
            print(f"          {c_yellow('⚠ network error — all sources unreachable')}")
            move_to_failed(working_path, "network_error")
            cnt["network_error"] += 1
            progress["miss"].append(fname)   # retry once fixed
            save_checkpoint(progress)
            save_artist_cache(artist_cache)
            _halt_pipeline(fname, "network error — all sources unreachable")
        elif status == "error":
            print(f"          {c_red('✗ error:')} {detail}")
            move_to_failed(working_path, "error")
            cnt["errors"] += 1
            progress["error"].append(fname)  # retry once fixed
            save_checkpoint(progress)
            save_artist_cache(artist_cache)
            _halt_pipeline(fname, detail)
        else:
            print(f"          {c_red('✗ unknown status:')} {status} ({detail})")
            cnt["errors"] += 1
            progress["error"].append(fname)  # retry once fixed
            save_checkpoint(progress)
            save_artist_cache(artist_cache)
            _halt_pipeline(fname, f"unknown status: {status} ({detail})")

        # Save checkpoint and artist cache after every file
        save_checkpoint(progress)
        save_artist_cache(artist_cache)

    # Final saves
    save_checkpoint(progress)
    save_artist_cache(artist_cache)

    banner([
        f"{c_bold('✓ Genre Tagger — session complete')}",
        f"Genres written  : {cnt['ok']}",
        f"Network errors  : {cnt['network_error']}",
        f"No genre found  : {cnt['no_genre']}",
        f"Errors          : {cnt['errors']}",
        f"Skipped (done)  : {cnt['skipped']}",
    ])

    unused = []   # initialise to avoid reference error
    if genre_counter:
        print(f"\n  {c_bold('Genre frequency this run:')}")
        for tag, n in sorted(genre_counter.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {tag:<22} : {n}")

        unused = sorted(all_genre_display - used_genres)

    # Only show "all predefined genres" if we actually wrote new genres
    if cnt["ok"] > 0:
        if unused:
            print(f"\n  {c_dim(f'Not seen this run ({len(unused)}/{len(all_genre_display)}):')}")
            line = ""
            for g in unused:
                if len(line) + len(g) + 2 > 70:
                    print(f"    {c_dim(line)}")
                    line = ""
                line += g + ", "
            if line:
                print(f"    {c_dim(line[:-2])}")
        else:
            print(f"\n  {c_green('All predefined genres were used at least once!')}")

    sign_off = "smooth run, nice work (｡•̀ᴗ-)✧" if cnt["errors"] == 0 and cnt["network_error"] == 0 \
        else "a few hiccups — worth a peek at the log (._.)"
    print(f"\n  {c_dim(sign_off)}\n")

    logger.debug(f"Run finished: {cnt}")
    logger.debug(f"Genre frequency: {genre_counter}")

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