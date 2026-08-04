"""
🟢 Spotify Tagger — Metadata Edition
─────────────────────────────────────────────────────────────────────────────
Fetches Spotify metadata (title, artist, albumartist, album, date, album art)
and writes them directly into the corresponding Vorbis comment tags.

Tags written to each file:
  ▸ title                  : Full track title with feat. annotations removed
  ▸ artist                 : All artists comma-separated (main + all featuring)
  ▸ albumartist            : Primary album artist
  ▸ album                  : Album name
  ▸ date                   : Release year (4 digits)
  ▸ metadata_block_picture : Album art embedded as JPEG (640×640 JPEG, base64)

Spotify daily limit:
  ▸ Spotify's Web API throttles search-heavy usage to ~600-700 requests/day
    per credential pair. This script tracks daily usage in a JSON file inside
    logs/ and shows a warning when the budget (700) is exceeded, but does NOT
    stop — you can continue at your own risk (Spotify may start returning 429s).
  ▸ With 8,203 files and ~700 calls/day budget ≈ 12 days to complete.
    Each day you just re-run the script; it picks up where it left off.

Filename format assumption:
  ▸ Files must be named "Artist - Title.ogg" (Spytify default output).
  ▸ Files with no " - " separator are logged and skipped automatically.

Checkpoint behaviour:
  ▸ Files listed in the checkpoint JSON under "done" are ALWAYS skipped.
  ▸ Files NOT in the "done" list are processed (and the tags overwritten)
    regardless of whether a tag already exists.
  ▸ "miss" and "skip" entries are NOT considered processed — they will be retried.
  ▸ Delete or clear the checkpoint file to reprocess everything.

Output locations:
  ▸ Log file     → logs/spotify_tagger.log        (everything, DEBUG level)
  ▸ Checkpoint   → logs/spotify_progress.json
  ▸ Daily stats  → logs/spotify_daily_stats.json

Before first run: edit the CONFIG block below with your own credentials and folder.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import re
import shutil
import time
import json
import random
import logging
import datetime
import unicodedata
import textwrap
import requests
import spotipy
import base64
from typing import Optional
from pathlib import Path
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.cache_handler import CacheHandler
from mutagen.oggvorbis import OggVorbis
from mutagen.flac import Picture

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOOK & FEEL  ─  colors + tiny console helpers                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

THEME        = GREEN   # ← this script's signature color (Spotify green!)
SCRIPT_EMOJI = "🟢"     # Spotify tagger's signature icon

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
# ║  CONFIG  ─  input/output dirs + Spotify creds entered interactively      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SCRIPT_DIR   = Path(__file__).parent.resolve()
PARENT_DIR   = SCRIPT_DIR.parent
LOGS_DIR     = PARENT_DIR / "logs"
CONFIGS_DIR  = PARENT_DIR / "configs"
TEMP_DIR     = PARENT_DIR / "temp"
FAILED_DIR   = PARENT_DIR / "failed"
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


# Save this run's chosen paths + credentials under configs/ for the record.
CONFIG_FILE = CONFIGS_DIR / "spotify_tagger.json"

# ── Input / output folders — chosen manually at startup, both mandatory ──────
print()

default_input = default_output = None
default_client_id = default_client_secret = None
if CONFIG_FILE.exists():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        p = Path(data.get("input_directory", ""))
        if p.is_dir():
            default_input = p
        p = Path(data.get("output_directory", ""))
        default_output = p
        default_client_id = data.get("spotify_client_id")
        default_client_secret = data.get("spotify_client_secret")
    except Exception:
        pass

INPUT_DIR  = _prompt_for_directory("input folder of OGG files to read (read-only, never modified)", must_exist=True, default=default_input)
default_output = default_output if default_output is not None else PARENT_DIR / "outputs"
OUTPUT_DIR = _prompt_for_directory("output folder for finished, tagged files", must_exist=False, default=default_output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print()

# ── Credentials — this script needs your Spotify app's client ID/secret ─────
SPOTIFY_CLIENT_ID:     str = _prompt_for_credential("your Spotify client ID", default=default_client_id)
SPOTIFY_CLIENT_SECRET: str = _prompt_for_credential("your Spotify client secret", default=default_client_secret)
print()

# Save config for next run
CONFIG_FILE.write_text(
    json.dumps({
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "spotify_client_id": SPOTIFY_CLIENT_ID,
        "spotify_client_secret": SPOTIFY_CLIENT_SECRET,
    }, indent=2),
    encoding="utf-8"
)

# The source in INPUT_DIR is read-only and never modified. Every file gets
# copied into temp/ first, tagged there, then the finished copy is moved
# into OUTPUT_DIR.
MUSIC_FOLDER = str(INPUT_DIR)

DAILY_CALL_LIMIT = 700
BASE_DELAY = 1.8
JITTER     = 0.6

PROGRESS_FILE    = LOGS_DIR / "spotify_progress.json"
DAILY_STATS_FILE = LOGS_DIR / "spotify_daily_stats.json"
LOG_FILE         = LOGS_DIR / "spotify_tagger.log"

class ImmediateFileHandler(logging.FileHandler):
    """A FileHandler that flushes after every single record — crash-safe logs."""
    def emit(self, record):
        super().emit(record)
        self.flush()

_fmt = "%(asctime)s │ %(levelname)-8s │ %(funcName)-22s │ %(message)s"

# ── Console filter: keep the terminal clean ──────────────────────────────
class _SuppressConsoleNoise(logging.Filter):
    """Reject log records whose message contains noisy boilerplate."""
    _BLOCK_PHRASES = (
        "Moved failed file to",
        "Catalogued failed file to",   # added to hide the file‑cataloguing log line
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

for lib in ['requests', 'urllib3', 'spotipy', 'requests.packages.urllib3']:
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Null cache handler ──
class NullCacheHandler(CacheHandler):
    def get_cached_token(self):            return None
    def save_token_to_cache(self, token):  pass

# ── Regex: remove (feat. X), (ft. X), (with X), [feat. X] from titles ──
FEAT_RE = re.compile(
    r'\s*[\(\[]\s*(?:feat(?:uring)?\.?|ft\.?|with)\s+[^\)\]]+[\)\]]',
    flags=re.IGNORECASE,
)

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

# ──────────────────────────────────────────────────────────────────────────────
#  DAILY QUOTA TRACKER
# ──────────────────────────────────────────────────────────────────────────────

def load_daily_stats() -> dict:
    today = datetime.date.today().isoformat()
    if DAILY_STATS_FILE.exists():
        try:
            data = json.loads(DAILY_STATS_FILE.read_text(encoding="utf-8"))
            if data.get("date") == today:
                logger.debug(f"Daily stats loaded — {data['calls']} calls used so far today")
                return data
        except Exception as exc:
            logger.warning(f"Could not read daily stats ({exc}); resetting to 0")
    stats = {"date": today, "calls": 0}
    _save_daily_stats(stats)
    logger.debug(f"Daily stats reset for {today}")
    return stats

def _save_daily_stats(stats: dict) -> None:
    """
    Persist today's call count. A "_meta" key is added purely for
    readability; load_daily_stats() only ever reads "date" and "calls",
    so this can't affect behaviour.
    """
    ordered = {
        "_meta": {
            "schema": "spotify-daily-stats",
            "description": "Tracks Spotify Web API calls used today against the soft daily budget.",
            "daily_soft_limit": DAILY_CALL_LIMIT,
        },
        "date": stats["date"],
        "calls": stats["calls"],
    }
    DAILY_STATS_FILE.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

def increment_calls(stats: dict, n: int = 1) -> None:
    stats["calls"] += n
    _save_daily_stats(stats)

def budget_remaining(stats: dict) -> int:
    return max(0, DAILY_CALL_LIMIT - stats["calls"])

# ──────────────────────────────────────────────────────────────────────────────
#  PROGRESS TRACKER  (resume across sessions)
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_SCHEMA = "spotify-checkpoint"
CHECKPOINT_KEYS   = ("done", "miss", "skip", "error")

def load_progress() -> dict:
    """
    Return the progress dict (keys: "done", "miss", "skip", "error").
    A "_meta" key, if present, is informational only and is stripped here.
    """
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            data.pop("_meta", None)
            for key in CHECKPOINT_KEYS:
                data.setdefault(key, [])
            # Normalize filenames to basename
            for key in CHECKPOINT_KEYS:
                data[key] = [os.path.basename(f) for f in data[key]]
            counts = {k: len(data[k]) for k in CHECKPOINT_KEYS}
            logger.debug(f"Progress loaded — {counts}")
            return data
        except Exception as exc:
            logger.error(f"Failed to load progress ({exc}); starting fresh")
    return {key: [] for key in CHECKPOINT_KEYS}

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
                "Tracks per-file Spotify metadata tagging progress. Only "
                "files under 'done' are skipped on the next run."
            ),
            "last_updated": datetime.datetime.now().isoformat(timespec="seconds"),
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

    Copies (never moves) — the temp/ working copy is what still gets
    carried into OUTPUT_DIR regardless, per the final-pass rule below.
    """
    target_dir = FAILED_DIR / error_type
    target_dir.mkdir(parents=True, exist_ok=True)

    src = Path(filepath)
    dst = target_dir / src.name
    try:
        shutil.copy2(str(src), str(dst))
        logger.info(f"Catalogued failed file to {dst}")   # hidden from console
        print(f"          {c_dim('↳ catalogued under failed/' + error_type + '/')}")
    except Exception as e:
        logger.error(f"Failed to catalog {src} to {dst}: {e}")


def _halt_pipeline(fname: str, detail: str) -> None:
    """A real error (network down, tag-write failure) happened — stop the
    WHOLE pipeline here for manual review. (Only real errors halt — 'no
    Spotify match' and bad filenames still get pushed through to output/.)"""
    print(f"\n{c_red('HALTED —')} a real error hit '{fname}'; stopping the whole pipeline for manual review.")
    print(f"  {c_dim('(no Spotify match / bad filenames never halt — only network/technical errors do)')}")
    print(f"  {c_dim('Fix the underlying issue, then re-run run_tagger — already-tagged files are skipped automatically.')}")
    logger.critical(f"HALTED — {fname}: {detail}")
    sys.exit(1)


def copy_to_output(filepath: str, fname: str) -> None:
    """Copy the fully-processed (or as-far-as-it-got) file into output/ — the
    true final destination. Called for every outcome except a hard halt.
    A copy failure here (disk full, permissions...) is itself a real error
    and halts the pipeline, same as any other stage."""
    dst = OUTPUT_DIR / fname
    try:
        shutil.copy2(filepath, str(dst))
        logger.info(f"Copied to output/: {dst}")
    except Exception as e:
        logger.error(f"Failed to copy {filepath} to output/: {e}")
        print(f"          {c_red('✗ could not copy to output/:')} {e}")
        _halt_pipeline(fname, f"could not copy to output/: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  METADATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def clean_title(raw: str) -> str:
    cleaned = FEAT_RE.sub("", raw).strip()
    cleaned = re.sub(r'[\s\-–—]+$', '', cleaned).strip()
    return cleaned if cleaned else raw

def build_artist_string(track: dict) -> str:
    names = [a["name"] for a in track.get("artists", [])]
    return ", ".join(names)

def embed_album_art(audio: OggVorbis, image_url: str) -> bool:
    try:
        logger.debug(f"Downloading album art: {image_url}")
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()

        pic        = Picture()
        pic.data   = resp.content
        pic.type   = 3
        pic.mime   = "image/jpeg"
        pic.width  = 640
        pic.height = 640
        pic.depth  = 24

        audio["metadata_block_picture"] = [
            base64.b64encode(pic.write()).decode("ascii")
        ]
        logger.debug(f"Art embedded OK — {len(resp.content):,} bytes")
        return True
    except Exception as exc:
        logger.warning(f"Album art embedding failed: {exc}")
        return False

# ──────────────────────────────────────────────────────────────────────────────
#  SPOTIFY SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def search_track(
    sp:          spotipy.Spotify,
    artist:      str,
    title:       str,
    daily_stats: dict,
    retries:     int = 3,
) -> object:
    """
    Returns the track dict on success, None if not found,
    or _NETWORK_ERROR if a network/connectivity error occurs.
    """
    clean_q = FEAT_RE.sub("", title).strip()
    queries = [
        f'artist:"{artist}" track:"{clean_q}"',
        f"artist:{artist} track:{clean_q}",
        f"{artist} {clean_q}",
    ]

    for attempt in range(retries):
        for query in queries:
            try:
                logger.debug(f"Spotify query (attempt {attempt+1}): {query}")
                results = sp.search(q=query, type="track", limit=1)
                increment_calls(daily_stats, 1)

                items = results["tracks"]["items"]
                if items:
                    match = items[0]
                    logger.debug(
                        f"  ✓ Match: '{match['name']}' by "
                        f"'{match['artists'][0]['name']}' "
                        f"(album: '{match['album']['name']}')"
                    )
                    return match

            except spotipy.exceptions.SpotifyException as exc:
                if exc.http_status == 429:
                    wait = int(getattr(exc, "headers", {}).get("Retry-After", 30))
                    wait += random.uniform(2, 8)
                    logger.warning(f"Spotify 429 — rate limited, waiting {wait:.0f}s")
                    print(f"          {c_yellow(f'⏳ rate limited — waiting {wait:.0f}s')}")
                    time.sleep(wait)
                    break
                elif exc.http_status and 500 <= exc.http_status < 600:
                    # Server error → treat as network error
                    logger.error(f"Spotify server error {exc.http_status}: {exc}")
                    return _NETWORK_ERROR
                else:
                    logger.error(f"Spotify API error {exc.http_status}: {exc}")
                    return None
            except Exception as exc:
                if _is_network_error(exc):
                    logger.warning(f"Network error during Spotify search: {exc}")
                    return _NETWORK_ERROR
                backoff = 2 ** attempt
                logger.warning(f"Search error ({exc}) — retry in {backoff}s")
                time.sleep(backoff)
                break

    logger.debug(f"  ✗ No Spotify match found: '{artist}' — '{title}'")
    return None

# ──────────────────────────────────────────────────────────────────────────────
#  PER-FILE PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def process_file(
    filepath:    str,
    sp:          spotipy.Spotify,
    daily_stats: dict,
) -> str:
    fname_base = os.path.splitext(os.path.basename(filepath))[0]

    if " - " not in fname_base:
        logger.warning(f"SKIP (unparseable filename — no ' - ' separator): {fname_base}")
        print(f"          {c_yellow('⚠ unparseable filename:')} {fname_base}")
        return "skip"

    artist, title = fname_base.split(" - ", 1)
    artist = artist.strip()
    title  = title.strip()
    logger.debug(f"Parsed → Artist='{artist}'  Title='{title}'")

    track = search_track(sp, artist, title, daily_stats)
    if track is _NETWORK_ERROR:
        print(f"          {c_yellow('⚠ network error (search failed)')}")
        return "network_error"
    if not track:
        print(f"          {c_yellow('⚠ no Spotify match')}")
        return "miss"

    album = track["album"]

    tag_title    = clean_title(track["name"])
    tag_artist   = build_artist_string(track)
    tag_alb_art  = album.get("artists", [{}])[0].get("name", artist)
    tag_album    = album.get("name", "")
    tag_date     = album.get("release_date", "")[:4]

    logger.debug(
        f"  Tags → title='{tag_title}' | artist='{tag_artist}' | "
        f"album='{tag_album}' ({tag_date})"
    )

    try:
        audio = OggVorbis(filepath)

        audio["title"]       = [tag_title]
        audio["artist"]      = [tag_artist]
        audio["albumartist"] = [tag_alb_art]
        audio["album"]       = [tag_album]
        audio["date"]        = [tag_date]

        images = sorted(
            album.get("images", []),
            key=lambda img: img.get("width", 0),
            reverse=True,
        )
        if images:
            embed_album_art(audio, images[0]["url"])

        audio.save()
        print(f"          {c_green('✓')} {tag_artist} — {tag_title}  {c_dim(f'→ ({tag_album}) [{tag_date}]')}")
        logger.debug(f"OK: {fname_base}")
        return "ok"

    except Exception as exc:
        logger.error(f"Tag write failed for '{fname_base}': {exc}", exc_info=True)
        print(f"          {c_red('✗ write error:')} {exc}")
        return "error"

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.debug("═" * 72)
    logger.debug(f"{SCRIPT_EMOJI} Spotify Tagger — session starting")
    logger.debug(f"Music folder : {MUSIC_FOLDER}")
    logger.debug(f"Daily budget (soft limit) : {DAILY_CALL_LIMIT} API calls/day (warns only)")

    if not os.path.isdir(MUSIC_FOLDER):
        logger.critical(f"MUSIC_FOLDER not found: {MUSIC_FOLDER}")
        sys.exit(f"ERROR: Folder not found — {MUSIC_FOLDER}")

    daily_stats = load_daily_stats()
    progress    = load_progress()

    # ── Optional cleanup: remove any file from error/miss/skip if it's already in done ──
    done_set = set(progress["done"])
    # Keep only done files that exist in current folder
    all_files = sorted(f for f in os.listdir(MUSIC_FOLDER) if f.lower().endswith(".ogg"))
    current_done = {f for f in all_files if f in done_set}
    for key in ("error", "miss", "skip"):
        progress[key] = [f for f in progress[key] if f not in current_done]

    remaining = budget_remaining(daily_stats)
    logger.debug(f"API calls remaining today before soft limit: {remaining}")

    limit_exceeded_warned = False
    if daily_stats["calls"] >= DAILY_CALL_LIMIT:
        msg = (
            f"Daily Spotify API soft limit already reached "
            f"({daily_stats['calls']}/{DAILY_CALL_LIMIT} calls used on {daily_stats['date']}). "
            f"Continuing anyway — you may encounter rate limiting (429 errors)."
        )
        logger.warning(msg)
        print(f"\n{c_yellow('⚠ ' + msg)}\n")
        limit_exceeded_warned = True

    logger.debug("Initialising Spotify client (no cache mode)...")
    try:
        sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                cache_handler=NullCacheHandler(),
            ),
            requests_timeout=15,
            retries=3,
        )
        sp.search(q="test", type="track", limit=1)
        increment_calls(daily_stats, 1)
        logger.debug("Spotify client ready and authenticated")
    except Exception as exc:
        logger.critical(f"Spotify init failed: {exc}", exc_info=True)
        sys.exit(f"ERROR: Could not connect to Spotify — {exc}")

    total = len(all_files)
    logger.debug(f"Total OGG files found: {total}")

    unprocessed = [f for f in all_files if f not in current_done]
    logger.debug(
        f"Unprocessed files: {len(unprocessed)} "
        f"(already done: {len(current_done)})"
    )

    banner([
        f"{c_bold(SCRIPT_EMOJI + '  Spotify Tagger')}  {c_dim('· overwrite mode')}",
        f"Folder           : {MUSIC_FOLDER}",
        f"Total files      : {total}",
        f"Already done     : {len(current_done)}",
        f"To process       : {len(unprocessed)}",
        f"Soft limit today : {DAILY_CALL_LIMIT} calls  (warning only)",
        f"Calls used today : {daily_stats['calls']}",
    ])
    print()

    stats                    = {"ok": 0, "miss": 0, "skip": 0, "network_error": 0, "error": 0}
    missed_files: list[str] = []   # explicit record of everything spotify_tagger couldn't tag
    calls_at_session_start   = daily_stats["calls"]

    for i, fname in enumerate(all_files, 1):
        src_path = os.path.join(MUSIC_FOLDER, fname)
        filepath = str(TEMP_DIR / fname)
        prefix   = c_dim(f"[{i:>{len(str(total))}}/{total}]")

        if fname in current_done:
            stats["skip"] += 1
            print(f"{prefix} {c_dim('· already tagged, skipping —')} {fname}")
            continue

        if not limit_exceeded_warned and budget_remaining(daily_stats) <= 0:
            msg = (
                f"Daily Spotify API soft limit ({DAILY_CALL_LIMIT}) reached. "
                f"Continuing anyway — you may get 429 rate limit errors."
            )
            logger.warning(msg)
            print(f"\n{c_yellow('⚠ ' + msg)}\n")
            limit_exceeded_warned = True

        print(f"{prefix} {fname}")

        # Never touch the input file directly — work on a copy in temp/.
        try:
            shutil.copy2(src_path, filepath)
        except Exception as e:
            logger.error(f"Could not copy {src_path} to temp/: {e}")
            print(f"          {c_red('✗ error copying to temp/:')} {e}")
            stats["error"] += 1
            progress["error"].append(fname)
            save_progress(progress)
            _halt_pipeline(fname, f"copy to temp/ failed: {e}")
            continue

        try:
            status = process_file(filepath, sp, daily_stats)
        except Exception as exc:
            logger.error(f"Unhandled exception on '{fname}': {exc}", exc_info=True)
            print(f"          {c_red('✗ unexpected error:')} {exc}")
            status = "error"

        # Update stats and progress
        if status == "ok":
            stats["ok"] += 1
            for key in ("error", "miss", "skip"):
                if fname in progress[key]:
                    progress[key].remove(fname)
            progress["done"].append(fname)
            current_done.add(fname)
            copy_to_output(filepath, fname)

        elif status == "miss":
            stats["miss"] += 1
            move_to_failed(filepath, "no_match")
            progress["miss"].append(fname)
            copy_to_output(filepath, fname)
            missed_files.append(fname)
            print(f"          {c_yellow('⚠ MISSED BY SPOTIFY —')} no match found; still carried through to output/.")
            logger.warning(f"MISSED BY SPOTIFY (no match found): {fname} — copied to output/ without Spotify metadata")

        elif status == "skip":
            stats["skip"] += 1
            move_to_failed(filepath, "bad_filename")
            progress["skip"].append(fname)
            copy_to_output(filepath, fname)
            missed_files.append(fname)
            print(f"          {c_yellow('⚠ MISSED BY SPOTIFY —')} unparseable filename; still carried through to output/.")
            logger.warning(f"MISSED BY SPOTIFY (bad filename): {fname} — copied to output/ without Spotify metadata")

        elif status == "network_error":
            stats["network_error"] += 1
            move_to_failed(filepath, "network_error")
            progress["miss"].append(fname)   # retry once fixed
            save_progress(progress)
            _halt_pipeline(fname, "network error")

        else:  # error — a real problem, not a soft miss
            stats["error"] += 1
            move_to_failed(filepath, "error")
            progress["error"].append(fname)  # retry once fixed
            save_progress(progress)
            _halt_pipeline(fname, str(status))

        # Save checkpoint after every file
        save_progress(progress)

        # Polite delay
        if status not in ("skip",):
            delay = BASE_DELAY + random.uniform(-JITTER, JITTER)
            time.sleep(max(0.8, delay))

    # Final save & summary
    save_progress(progress)
    session_calls = daily_stats["calls"] - calls_at_session_start

    banner([
        f"{c_bold('✓ Spotify Tagger — session complete')}",
        f"Tagged OK        : {stats['ok']}",
        f"Network errors   : {stats['network_error']}",
        f"No Spotify match : {stats['miss']}",
        f"Skipped          : {stats['skip']}",
        f"Errors           : {stats['error']}",
        f"API calls (session) : {session_calls}",
        f"API calls (today)   : {daily_stats['calls']}/{DAILY_CALL_LIMIT}  (soft limit)",
    ])

    if missed_files:
        print(f"\n{c_yellow('⚠ MISSED BY SPOTIFY')} {c_dim(f'({len(missed_files)} file(s) — still in output/, just without Spotify metadata)')}")
        for mf in missed_files:
            print(f"    {c_yellow('•')} {mf}")
        logger.warning(f"MISSED BY SPOTIFY this run ({len(missed_files)}): {missed_files}")

    sign_off = "smooth run, nice work (｡•̀ᴗ-)✧" if stats["error"] == 0 and stats["network_error"] == 0 \
        else "a few hiccups — worth a peek at the log (._.)"
    print(f"\n  {c_dim(sign_off)}\n")

    logger.debug(f"Session complete. stats={stats}. missed_by_spotify={missed_files}. api_today={daily_stats['calls']}")
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