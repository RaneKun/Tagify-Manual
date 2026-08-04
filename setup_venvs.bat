@echo off
REM ============================================================================
REM  ⚙️  setup_venvs.bat  —  Per-Script Conda Environment Setup
REM ============================================================================
REM  Creates one isolated Conda environment per tagger script, inside .\venvs
REM  Scripts are looked up inside .\scripts  (an env is only built if its .py
REM  file actually exists there).
REM  Skips environments that are already fully installed (marker-file based).
REM  Minimal console output; FULL verbose detail goes to logs\setup.log
REM
REM  Environments created (all under .\venvs\<name>), color-coded to match
REM  each script's own console theme:
REM    spotify_tagger        py3.11  CPU   (spotipy, requests, mutagen)               [green]
REM    online_genre_tagger   py3.11  CPU   (requests, mutagen, pylast, musicbrainzngs)[blue]
REM    online_lyrics_tagger  py3.11  CPU   (requests, mutagen, syncedlyrics)          [red]
REM    bpm_mood_tagger       py3.10  CPU   (aubio+librosa+numpy from conda-forge,     [yellow]
REM                                         aubio has NO Windows pip wheel).
REM                                         NOTE: on-disk file is bpm+mood_tagger.py
REM                                         (typo with a "+"), see SCRIPT_FILE below.
REM    local_genre_tagger    py3.10  GPU   (musicnn on TensorFlow 2.10 -- the LAST    [cyan]
REM                                         TF release with native-Windows GPU;
REM                                         needs conda-forge cudatoolkit=11.2 +
REM                                         cudnn=8.1, self-contained per-env).
REM                                         musicnn is installed with --no-deps
REM                                         because its stale metadata pins
REM                                         numpy<1.17, which conflicts with the
REM                                         numpy TensorFlow 2.10 actually needs.
REM                                         Uses SYSTEM ffmpeg, not conda-forge's
REM                                         (conda-forge's crashed with
REM                                         STATUS_DLL_NOT_FOUND / 0xC0000135 on
REM                                         at least one real-world machine).
REM    local_lyrics_tagger   py3.11  GPU   (Demucs + DeepFilterNet3 + faster-whisper  [magenta]
REM                                         on official PyTorch cu121 wheels).
REM                                         Also uses SYSTEM ffmpeg for the same
REM                                         reason as local_genre_tagger above.
REM
REM  NOTE ON THE "+" TYPO: the block comment above (inherited from an earlier
REM  version of this file) mentions a special-case filename override for
REM  bpm_mood_tagger. That override does NOT actually exist anywhere in this
REM  script — every section, including bpm_mood_tagger's, resolves
REM  SCRIPT_FILE the same generic way ("%NAME%.py"). Left exactly as-is
REM  since guessing at which side (comment vs. code) is "correct" risks
REM  changing real behaviour; flagging it here instead so it's easy to spot.
REM ============================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM ---------------------------------------------------------------------------
REM  ANSI colors (classic ESC-var trick -- works in cmd.exe on Win10/11 by
REM  default for an interactive console). Purely cosmetic: every color
REM  variable degrades to plain text if VT processing isn't available.
REM ---------------------------------------------------------------------------
for /f %%A in ('echo prompt $E^|cmd') do set "ESC=%%A"
set "C_RESET=%ESC%[0m"
set "C_BOLD=%ESC%[1m"
set "C_DIM=%ESC%[2m"
set "C_GREEN=%ESC%[92m"
set "C_YELLOW=%ESC%[93m"
set "C_RED=%ESC%[91m"
set "C_RED_DARK=%ESC%[31m"
set "C_CYAN=%ESC%[96m"
set "C_BLUE=%ESC%[94m"
set "C_MAGENTA=%ESC%[95m"

REM ---------------------------------------------------------------------------
REM  0. PATHS
REM ---------------------------------------------------------------------------
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VENVS_DIR=%ROOT%\venvs"
set "LOGS_DIR=%ROOT%\logs"
set "SCRIPTS_DIR=%ROOT%\scripts"
set "LOGFILE=%LOGS_DIR%\setup.log"

if not exist "%VENVS_DIR%" mkdir "%VENVS_DIR%" >nul 2>&1
if not exist "%LOGS_DIR%"  mkdir "%LOGS_DIR%"  >nul 2>&1

REM ---- CLEAR the log file at the start of every run (ONLY here) ----
> "%LOGFILE%" echo.   & REM truncates the file

set "GET_ROW_PS1=%TEMP%\get_row_%RANDOM%.ps1"
> "%GET_ROW_PS1%" echo $Host.UI.RawUI.CursorPosition.Y

set "CLEAR_BLOCK_PS1=%TEMP%\clear_block_%RANDOM%.ps1"
> "%CLEAR_BLOCK_PS1%" echo param($startRow, $endRow)
>>"%CLEAR_BLOCK_PS1%" echo $raw = $Host.UI.RawUI
>>"%CLEAR_BLOCK_PS1%" echo $buf = $raw.BufferSize
>>"%CLEAR_BLOCK_PS1%" echo $rect = New-Object System.Management.Automation.Host.Rectangle 0, [int]$startRow, ($buf.Width - 1), ([int]$endRow - 1)
>>"%CLEAR_BLOCK_PS1%" echo $blank = New-Object System.Management.Automation.Host.BufferCell(' ', $raw.ForegroundColor, $raw.BackgroundColor, [System.Management.Automation.Host.BufferCellType]::Complete)
>>"%CLEAR_BLOCK_PS1%" echo $raw.SetBufferContents($rect, $blank)
>>"%CLEAR_BLOCK_PS1%" echo $raw.CursorPosition = New-Object System.Management.Automation.Host.Coordinates 0, [int]$startRow

REM ---------------------------------------------------------------------------
REM  RUN_PROGRESS_PY  --  runs one conda/pip step as a subprocess (via conda's
REM  own base python, invoked directly - NOT through "conda run", which has a
REM  documented Windows bug (conda/conda#9700) that clears terminal output)
REM  while a tqdm progress bar renders on the console, colour-matched to the
REM  environment's header colour. The bar's percentage is eased toward
REM  completion over an estimated duration (SizeMB is only used to pace this
REM  - conda/pip don't expose real byte-level progress in a way that's worth
REM  parsing here) and jumps to 100% the moment the real process exits. Only
REM  the bar + percentage are shown - no size/speed/ETA text. Exits with the
REM  real process's exit code so the caller's "if errorlevel 1" checks work
REM  as expected. The real command's combined stdout/stderr is written to
REM  the log file once the step finishes.
REM ---------------------------------------------------------------------------
set "RUN_PROGRESS_PY=%TEMP%\run_progress_%RANDOM%.py"
> "%RUN_PROGRESS_PY%" echo import sys, subprocess, time, math, threading, random
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo mode, env_path, size_mb, color, log_file = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
>>"%RUN_PROGRESS_PY%" echo arg1 = sys.argv[6] if len(sys.argv^) ^> 6 else "none"
>>"%RUN_PROGRESS_PY%" echo arg2 = sys.argv[7] if len(sys.argv^) ^> 7 else "none"
>>"%RUN_PROGRESS_PY%" echo arg3 = sys.argv[8] if len(sys.argv^) ^> 8 else "none"
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo try:
>>"%RUN_PROGRESS_PY%" echo     from tqdm import tqdm
>>"%RUN_PROGRESS_PY%" echo except Exception:
>>"%RUN_PROGRESS_PY%" echo     tqdm = None
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo def qq(s^):
>>"%RUN_PROGRESS_PY%" echo     return '"' + s.replace('"', '') + '"'
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo def build_cmd(^):
>>"%RUN_PROGRESS_PY%" echo     if mode == "condacreate":
>>"%RUN_PROGRESS_PY%" echo         return ["conda", "create", "--prefix", qq(env_path^), "python=" + arg1, "-y"]
>>"%RUN_PROGRESS_PY%" echo     if mode == "pipupgrade":
>>"%RUN_PROGRESS_PY%" echo         return ["conda", "run", "-p", qq(env_path^), "python", "-m", "pip", "install", "--upgrade", "pip"]
>>"%RUN_PROGRESS_PY%" echo     if mode == "pipinstall":
>>"%RUN_PROGRESS_PY%" echo         pkgs = [qq(p^) for p in arg1.split(^)]
>>"%RUN_PROGRESS_PY%" echo         extra = [] if arg2 in ("none", ""^) else arg2.split(^)
>>"%RUN_PROGRESS_PY%" echo         return ["conda", "run", "-p", qq(env_path^), "python", "-m", "pip", "install"] + pkgs + extra
>>"%RUN_PROGRESS_PY%" echo     if mode == "condainstall":
>>"%RUN_PROGRESS_PY%" echo         pkgs = [qq(p^) for p in arg2.split(^)]
>>"%RUN_PROGRESS_PY%" echo         extra = [] if arg3 in ("none", ""^) else arg3.split(^)
>>"%RUN_PROGRESS_PY%" echo         return ["conda", "install", "-p", qq(env_path^), "-c", arg1] + pkgs + ["-y"] + extra
>>"%RUN_PROGRESS_PY%" echo     raise SystemExit("unknown mode: " + mode^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo cmd_str = " ".join(build_cmd(^)^)
>>"%RUN_PROGRESS_PY%" echo proc = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding="utf-8", errors="replace"^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo output_lines = []
>>"%RUN_PROGRESS_PY%" echo def reader(pipe^):
>>"%RUN_PROGRESS_PY%" echo     for line in iter(pipe.readline, ""^):
>>"%RUN_PROGRESS_PY%" echo         output_lines.append(line^)
>>"%RUN_PROGRESS_PY%" echo     pipe.close(^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo t = threading.Thread(target=reader, args=(proc.stdout,^)^)
>>"%RUN_PROGRESS_PY%" echo t.start(^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo bar = None
>>"%RUN_PROGRESS_PY%" echo if tqdm is not None:
>>"%RUN_PROGRESS_PY%" echo     bar = tqdm(total=100, bar_format="          {bar} {percentage:3.0f}%%", colour=color, ncols=56, leave=False^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo start = time.time(^)
>>"%RUN_PROGRESS_PY%" echo est_duration = max(2.5, size_mb / 6.0^)
>>"%RUN_PROGRESS_PY%" echo while proc.poll(^) is None:
>>"%RUN_PROGRESS_PY%" echo     elapsed = time.time(^) - start
>>"%RUN_PROGRESS_PY%" echo     pct = min(95.0, 100.0 * (1 - math.exp(-elapsed / est_duration^)^)^)
>>"%RUN_PROGRESS_PY%" echo     if bar is not None:
>>"%RUN_PROGRESS_PY%" echo         bar.n = round(pct, 1^)
>>"%RUN_PROGRESS_PY%" echo         bar.refresh(^)
>>"%RUN_PROGRESS_PY%" echo     time.sleep(0.15^)
>>"%RUN_PROGRESS_PY%" echo t.join(^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo if bar is not None:
>>"%RUN_PROGRESS_PY%" echo     steps_left = 100 - int(bar.n^)
>>"%RUN_PROGRESS_PY%" echo     step_delay = min(0.08, 0.6 / steps_left^) if steps_left ^> 0 else 0.08
>>"%RUN_PROGRESS_PY%" echo     for step in range(int(bar.n^) + 1, 101^):
>>"%RUN_PROGRESS_PY%" echo         bar.n = step
>>"%RUN_PROGRESS_PY%" echo         bar.refresh(^)
>>"%RUN_PROGRESS_PY%" echo         time.sleep(step_delay^)
>>"%RUN_PROGRESS_PY%" echo     bar.n = 100
>>"%RUN_PROGRESS_PY%" echo     bar.refresh(^)
>>"%RUN_PROGRESS_PY%" echo     time.sleep(random.uniform(1, 5^)^)
>>"%RUN_PROGRESS_PY%" echo     bar.close(^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo with open(log_file, "a", encoding="utf-8", errors="replace"^) as f:
>>"%RUN_PROGRESS_PY%" echo     f.writelines(output_lines^)
>>"%RUN_PROGRESS_PY%" echo.
>>"%RUN_PROGRESS_PY%" echo sys.exit(proc.returncode if proc.returncode is not None else 1^)

echo. >> "%LOGFILE%"
echo ================================================================================ >> "%LOGFILE%"
echo  SETUP RUN STARTED : %date% %time% >> "%LOGFILE%"
echo  Script root       : %ROOT% >> "%LOGFILE%"
echo  Scripts folder    : %SCRIPTS_DIR% >> "%LOGFILE%"
echo ================================================================================ >> "%LOGFILE%"

cls
echo %C_CYAN%================================================================%C_RESET%
echo %C_CYAN%  %C_BOLD%⚙️  Music Tagger Scripts%C_RESET%%C_CYAN% -- Per-Script Conda Environment Setup%C_RESET%
echo %C_CYAN%================================================================%C_RESET%
echo   Full verbose log : %LOGFILE%
echo   Environments dir : %VENVS_DIR%
echo   Scripts folder   : %SCRIPTS_DIR%
echo %C_CYAN%================================================================%C_RESET%
echo.

REM ---------------------------------------------------------------------------
REM  1. REQUIREMENT CHECKS
REM ---------------------------------------------------------------------------
echo %C_BOLD%[1/3] Checking requirements...%C_RESET%
echo.

REM ---- conda --------------------------------------------------------------
where conda >nul 2>>"%LOGFILE%"
if errorlevel 1 goto :NO_CONDA
set "TMPFILE=%TEMP%\conda_ver_%RANDOM%.txt"
call conda --version > "%TMPFILE%" 2>&1
set "CONDA_VER="
if exist "%TMPFILE%" set /p CONDA_VER=<"%TMPFILE%"
del "%TMPFILE%" >nul 2>&1
call :OK "conda found"
call :LOG "Found !CONDA_VER!"
call conda info >> "%LOGFILE%" 2>&1
goto :AFTER_CONDA_CHECK

:NO_CONDA
call :ERR "conda was NOT found in PATH."
echo.
echo    Miniconda/Anaconda is REQUIRED for this setup.
echo    Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
echo    After installing, open a NEW terminal so PATH refreshes, then re-run this script.
echo.
call :LOG "FATAL: conda not found. Aborting run."
pause
exit /b 1

:AFTER_CONDA_CHECK

REM ---- scripts folder -------------------------------------------------------
if exist "%SCRIPTS_DIR%\" goto :AFTER_SCRIPTS_DIR_CHECK
call :WARN "Scripts folder not found: %SCRIPTS_DIR%"
echo    Create a "scripts" folder next to this .bat and put the .py files in it.
call :LOG "Scripts folder missing: %SCRIPTS_DIR%"

:AFTER_SCRIPTS_DIR_CHECK

REM ---- NVIDIA GPU / driver --------------------------------------------------
set "GPU_AVAILABLE=0"
where nvidia-smi >nul 2>>"%LOGFILE%"
if errorlevel 1 goto :NO_GPU
call :OK "NVIDIA GPU / driver detected."
set "GPU_AVAILABLE=1"
nvidia-smi >> "%LOGFILE%" 2>&1
call :LOG "GPU_AVAILABLE=1"
goto :AFTER_GPU_CHECK

:NO_GPU
call :WARN "nvidia-smi not found - no NVIDIA driver detected."
echo    GPU environments (local_lyrics_tagger, local_genre_tagger) will install
echo    CPU-only fallback packages instead. To enable GPU acceleration, install
echo    or update the NVIDIA driver from https://www.nvidia.com/Download/index.aspx
call :LOG "GPU not detected -> GPU_AVAILABLE=0. CPU-only packages will be used for GPU envs."

:AFTER_GPU_CHECK

REM ---- system ffmpeg (local_genre_tagger and local_lyrics_tagger rely on it
REM      directly now - see 2e/2f below; bpm_mood_tagger still gets its own
REM      via conda-forge alongside aubio) ---------------------------------
where ffmpeg >nul 2>>"%LOGFILE%"
if errorlevel 1 goto :NO_SYS_FFMPEG
call :OK "System-wide ffmpeg found."
goto :AFTER_FFMPEG_CHECK

:NO_SYS_FFMPEG
call :WARN "System-wide ffmpeg NOT found on PATH."
echo    local_genre_tagger and local_lyrics_tagger both need a working system
echo    ffmpeg on PATH (they rely on the system copy rather than an installed one).
echo    Download a build from https://www.gyan.dev/ffmpeg/builds/ and add its
echo    bin folder to PATH, then re-run this script.
call :LOG "System ffmpeg missing - local_genre_tagger/local_lyrics_tagger will fail at runtime without it."

:AFTER_FFMPEG_CHECK

REM ---- internet connectivity -------------------------------------------------
ping -n 1 pypi.org >nul 2>>"%LOGFILE%"
if errorlevel 1 goto :NO_INTERNET
call :OK "Internet connectivity looks fine."
goto :AFTER_NET_CHECK

:NO_INTERNET
call :WARN "Could not reach pypi.org - check your internet connection before continuing."

:AFTER_NET_CHECK

REM ---- conda base python + tqdm (used to render progress bars below; invoked
REM      directly as python.exe rather than via "conda run", which has a
REM      documented Windows bug that clears terminal output - see notes above
REM      RUN_PROGRESS_PY) -----------------------------------------------------
set "CONDA_BASE="
for /f "usebackq delims=" %%B in (`conda info --base 2^>nul`) do set "CONDA_BASE=%%B"
set "BASE_PY=%CONDA_BASE%\python.exe"
if not exist "%BASE_PY%" (
    call :WARN "Could not locate conda base python.exe - progress bars will be disabled, installs still work"
    set "BASE_PY="
    goto :AFTER_TQDM_CHECK
)
"%BASE_PY%" -c "import tqdm" >nul 2>&1
if not errorlevel 1 (
    call :OK "tqdm already available"
    goto :AFTER_TQDM_CHECK
)
call :INFO "installing tqdm (used for progress bars)"
"%BASE_PY%" -m pip install --quiet tqdm >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    call :WARN "tqdm install failed - progress bars will be disabled, installs still work"
) else (
    call :OK "tqdm installed"
)

:AFTER_TQDM_CHECK

echo.
echo %C_BOLD%[2/3] Setting up environments%C_RESET%  %C_DIM%(will take a very long time, have your lunch/dinner)%C_RESET%
echo.

REM ===========================================================================
REM  2a. spotify_tagger        (pure API/metadata, CPU only)
REM ===========================================================================
set "NAME=spotify_tagger"
set "SCRIPT_FILE=%NAME%.py"
set "ENV_PATH=%VENVS_DIR%\%NAME%"
if not exist "%SCRIPTS_DIR%\%SCRIPT_FILE%" (
    echo   [SKIP]  %NAME% - "%SCRIPT_FILE%" not found in scripts folder
    call :LOG "[SKIP] %NAME% - %SCRIPT_FILE% not found in %SCRIPTS_DIR%, skipping environment"
    goto :AFTER_SPOTIFY
)
call :HEADER "%NAME%" "%C_GREEN%"
set "ENV_COLOR=green"
if exist "%ENV_PATH%\.setup_ok" goto :SKIP_SPOTIFY
call :BLOCK_START

call :RUN_PROGRESS "condacreate" "%ENV_PATH%" 35 "%NAME%: creating conda env (python=3.11)" "3.11" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: conda create FAILED - see log"
    goto :AFTER_SPOTIFY
)

call :RUN_PROGRESS "pipupgrade" "%ENV_PATH%" 2 "%NAME%: upgrading pip" "none" "none" "none"

call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 8 "%NAME%: installing spotipy, requests, mutagen" "spotipy requests mutagen" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: pip install FAILED - see log"
    goto :AFTER_SPOTIFY
)

call :VERIFY "%ENV_PATH%" "import spotipy, requests, mutagen; print('OK')" "%NAME%"
if errorlevel 1 goto :AFTER_SPOTIFY

call :BLOCK_END
echo done > "%ENV_PATH%\.setup_ok"
call :OK "%NAME% environment ready"
goto :AFTER_SPOTIFY

:SKIP_SPOTIFY
call :SKIP "%NAME%"

:AFTER_SPOTIFY

REM ===========================================================================
REM  2b. online_genre_tagger   (pure API/metadata, CPU only)
REM ===========================================================================
set "NAME=online_genre_tagger"
set "SCRIPT_FILE=%NAME%.py"
set "ENV_PATH=%VENVS_DIR%\%NAME%"
if not exist "%SCRIPTS_DIR%\%SCRIPT_FILE%" (
    echo   [SKIP]  %NAME% - "%SCRIPT_FILE%" not found in scripts folder
    call :LOG "[SKIP] %NAME% - %SCRIPT_FILE% not found in %SCRIPTS_DIR%, skipping environment"
    goto :AFTER_ONLINE_GENRE
)
call :HEADER "%NAME%" "%C_BLUE%"
set "ENV_COLOR=blue"
if exist "%ENV_PATH%\.setup_ok" goto :SKIP_ONLINE_GENRE
call :BLOCK_START

call :RUN_PROGRESS "condacreate" "%ENV_PATH%" 35 "%NAME%: creating conda env (python=3.11)" "3.11" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: conda create FAILED - see log"
    goto :AFTER_ONLINE_GENRE
)

call :RUN_PROGRESS "pipupgrade" "%ENV_PATH%" 2 "%NAME%: upgrading pip" "none" "none" "none"

call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 7 "%NAME%: installing requests, mutagen, pylast, musicbrainzngs" "requests mutagen pylast musicbrainzngs" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: pip install FAILED - see log"
    goto :AFTER_ONLINE_GENRE
)

call :VERIFY "%ENV_PATH%" "import requests, mutagen, pylast, musicbrainzngs; print('OK')" "%NAME%"
if errorlevel 1 goto :AFTER_ONLINE_GENRE

call :BLOCK_END
echo done > "%ENV_PATH%\.setup_ok"
call :OK "%NAME% environment ready"
goto :AFTER_ONLINE_GENRE

:SKIP_ONLINE_GENRE
call :SKIP "%NAME%"

:AFTER_ONLINE_GENRE

REM ===========================================================================
REM  2c. online_lyrics_tagger  (pure API/metadata, CPU only)
REM ===========================================================================
set "NAME=online_lyrics_tagger"
set "SCRIPT_FILE=%NAME%.py"
set "ENV_PATH=%VENVS_DIR%\%NAME%"
if not exist "%SCRIPTS_DIR%\%SCRIPT_FILE%" (
    echo   [SKIP]  %NAME% - "%SCRIPT_FILE%" not found in scripts folder
    call :LOG "[SKIP] %NAME% - %SCRIPT_FILE% not found in %SCRIPTS_DIR%, skipping environment"
    goto :AFTER_ONLINE_LYRICS
)
call :HEADER "%NAME%" "%C_RED_DARK%"
set "ENV_COLOR=red"
if exist "%ENV_PATH%\.setup_ok" goto :SKIP_ONLINE_LYRICS
call :BLOCK_START

call :RUN_PROGRESS "condacreate" "%ENV_PATH%" 35 "%NAME%: creating conda env (python=3.11)" "3.11" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: conda create FAILED - see log"
    goto :AFTER_ONLINE_LYRICS
)

call :RUN_PROGRESS "pipupgrade" "%ENV_PATH%" 2 "%NAME%: upgrading pip" "none" "none" "none"

call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 6 "%NAME%: installing requests, mutagen, syncedlyrics" "requests mutagen syncedlyrics" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: pip install FAILED - see log"
    goto :AFTER_ONLINE_LYRICS
)

call :VERIFY "%ENV_PATH%" "import requests, mutagen, syncedlyrics; print('OK')" "%NAME%"
if errorlevel 1 goto :AFTER_ONLINE_LYRICS

call :BLOCK_END
echo done > "%ENV_PATH%\.setup_ok"
call :OK "%NAME% environment ready"
goto :AFTER_ONLINE_LYRICS

:SKIP_ONLINE_LYRICS
call :SKIP "%NAME%"

:AFTER_ONLINE_LYRICS

REM ===========================================================================
REM  2d. bpm_mood_tagger       (librosa + aubio, CPU only - aubio has NO pip
REM                             wheels on Windows, MUST come from conda-forge)
REM ===========================================================================
set "NAME=bpm_mood_tagger"
set "SCRIPT_FILE=%NAME%.py"
set "ENV_PATH=%VENVS_DIR%\%NAME%"
if not exist "%SCRIPTS_DIR%\%SCRIPT_FILE%" (
    echo   [SKIP]  %NAME% - "%SCRIPT_FILE%" not found in scripts folder
    call :LOG "[SKIP] %NAME% - %SCRIPT_FILE% not found in %SCRIPTS_DIR%, skipping environment"
    goto :AFTER_BPM
)
call :HEADER "%NAME%" "%C_YELLOW%"
set "ENV_COLOR=yellow"
if exist "%ENV_PATH%\.setup_ok" goto :SKIP_BPM
call :BLOCK_START

call :RUN_PROGRESS "condacreate" "%ENV_PATH%" 35 "%NAME%: creating conda env (python=3.10)" "3.10" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: conda create FAILED - see log"
    goto :AFTER_BPM
)

call :RUN_PROGRESS "condainstall" "%ENV_PATH%" 180 "%NAME%: installing ffmpeg, aubio, librosa, numpy via conda-forge" "conda-forge" "ffmpeg aubio librosa numpy" "none"
if errorlevel 1 (
    call :ERR "%NAME%: conda-forge install FAILED - see log"
    goto :AFTER_BPM
)

call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 2 "%NAME%: installing mutagen via pip" "mutagen" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: pip install FAILED - see log"
    goto :AFTER_BPM
)

call :VERIFY "%ENV_PATH%" "import aubio, librosa, numpy, mutagen; print('OK')" "%NAME%"
if errorlevel 1 goto :AFTER_BPM

call :BLOCK_END
echo done > "%ENV_PATH%\.setup_ok"
call :OK "%NAME% environment ready"
goto :AFTER_BPM

:SKIP_BPM
call :SKIP "%NAME%"

:AFTER_BPM

REM ===========================================================================
REM  2e. local_genre_tagger    (MusicNN on TensorFlow - GPU via TF 2.10, the
REM                             LAST TF release with native-Windows GPU support.
REM                             Needs conda-forge cudatoolkit=11.2 + cudnn=8.1,
REM                             self-contained inside this env only.)
REM
REM  musicnn's published metadata hard-pins numpy<1.17, which is incompatible
REM  with the modern numpy TensorFlow 2.10 needs. musicnn's actual code does
REM  not require that ancient numpy to run, so it is installed with --no-deps
REM  AFTER tensorflow/numpy/librosa are already in place, skipping pip's
REM  dependency check entirely for that one package.
REM ===========================================================================
set "NAME=local_genre_tagger"
set "SCRIPT_FILE=%NAME%.py"
set "ENV_PATH=%VENVS_DIR%\%NAME%"
if not exist "%SCRIPTS_DIR%\%SCRIPT_FILE%" (
    echo   [SKIP]  %NAME% - "%SCRIPT_FILE%" not found in scripts folder
    call :LOG "[SKIP] %NAME% - %SCRIPT_FILE% not found in %SCRIPTS_DIR%, skipping environment"
    goto :AFTER_LOCAL_GENRE
)
call :HEADER "%NAME%" "%C_CYAN%"
set "ENV_COLOR=cyan"
if exist "%ENV_PATH%\.setup_ok" goto :SKIP_LOCAL_GENRE
call :BLOCK_START

call :RUN_PROGRESS "condacreate" "%ENV_PATH%" 35 "%NAME%: creating conda env (python=3.10)" "3.10" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: conda create FAILED - see log"
    goto :AFTER_LOCAL_GENRE
)

call :LOG "%NAME%: skipping conda-forge ffmpeg (using system ffmpeg instead - see :CHECK_SYS_FFMPEG)"
call :CHECK_SYS_FFMPEG "%NAME%"

if "%GPU_AVAILABLE%"=="0" goto :LG_SKIP_CUDA
call :RUN_PROGRESS "condainstall" "%ENV_PATH%" 700 "%NAME%: installing cudatoolkit=11.2, cudnn=8.1 via conda-forge (large download, please wait)" "conda-forge" "cudatoolkit=11.2 cudnn=8.1.0" "-q --copy"
if errorlevel 1 call :ERR "%NAME%: cudatoolkit/cudnn install FAILED - see log. Falling back to CPU TensorFlow."
goto :LG_AFTER_CUDA

:LG_SKIP_CUDA
call :INFO "%NAME%: GPU not available - skipping cudatoolkit/cudnn, using CPU TensorFlow"

:LG_AFTER_CUDA
call :RUN_PROGRESS "pipupgrade" "%ENV_PATH%" 2 "%NAME%: upgrading pip" "none" "none" "none"
call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 450 "%NAME%: installing tensorflow, numpy, librosa, mutagen (large download, please wait)" "tensorflow<2.11 numpy==1.23.5 librosa mutagen" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: pip install FAILED - see log"
    goto :AFTER_LOCAL_GENRE
)

call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 3 "%NAME%: installing musicnn" "musicnn" "--no-deps" "none"
if errorlevel 1 (
    call :ERR "%NAME%: musicnn install FAILED - see log"
    goto :AFTER_LOCAL_GENRE
)

call :VERIFY "%ENV_PATH%" "import tensorflow, musicnn, librosa, mutagen; print('OK')" "%NAME%"
if errorlevel 1 goto :AFTER_LOCAL_GENRE

call :TF_GPU_CHECK "%ENV_PATH%" "%NAME%"

>nul timeout /t 2
call :BLOCK_END
echo done > "%ENV_PATH%\.setup_ok"
call :OK "%NAME% environment ready"
goto :AFTER_LOCAL_GENRE

:SKIP_LOCAL_GENRE
call :SKIP "%NAME%"

:AFTER_LOCAL_GENRE

REM ===========================================================================
REM  2f. local_lyrics_tagger   (Demucs + DeepFilterNet3 + faster-whisper, GPU
REM                             via official PyTorch cu121 wheels)
REM ===========================================================================
set "NAME=local_lyrics_tagger"
set "SCRIPT_FILE=%NAME%.py"
set "ENV_PATH=%VENVS_DIR%\%NAME%"
if not exist "%SCRIPTS_DIR%\%SCRIPT_FILE%" (
    echo   [SKIP]  %NAME% - "%SCRIPT_FILE%" not found in scripts folder
    call :LOG "[SKIP] %NAME% - %SCRIPT_FILE% not found in %SCRIPTS_DIR%, skipping environment"
    goto :AFTER_LOCAL_LYRICS
)
call :HEADER "%NAME%" "%C_MAGENTA%"
set "ENV_COLOR=magenta"
if exist "%ENV_PATH%\.setup_ok" goto :SKIP_LOCAL_LYRICS
call :BLOCK_START

call :RUN_PROGRESS "condacreate" "%ENV_PATH%" 35 "%NAME%: creating conda env (python=3.11)" "3.11" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: conda create FAILED - see log"
    goto :AFTER_LOCAL_LYRICS
)

call :LOG "%NAME%: skipping conda-forge ffmpeg (using system ffmpeg instead - see :CHECK_SYS_FFMPEG)"
call :CHECK_SYS_FFMPEG "%NAME%"

call :RUN_PROGRESS "pipupgrade" "%ENV_PATH%" 2 "%NAME%: upgrading pip" "none" "none" "none"

if "%GPU_AVAILABLE%"=="0" goto :LL_CPU_TORCH
call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 2800 "%NAME%: installing torch/torchvision/torchaudio (CUDA 12.1, large download, please wait)" "torch torchvision torchaudio" "--index-url https://download.pytorch.org/whl/cu121" "none"
goto :LL_AFTER_TORCH

:LL_CPU_TORCH
call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 200 "%NAME%: GPU not available - installing CPU-only torch" "torch torchvision torchaudio" "none" "none"

:LL_AFTER_TORCH
if errorlevel 1 (
    call :ERR "%NAME%: torch install FAILED - see log"
    goto :AFTER_LOCAL_LYRICS
)

call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 300 "%NAME%: installing demucs, deepfilternet, faster-whisper, mutagen, soundfile" "demucs deepfilternet faster-whisper mutagen soundfile" "none" "none"
if errorlevel 1 (
    call :ERR "%NAME%: pip install FAILED - see log"
    goto :AFTER_LOCAL_LYRICS
)

call :VERIFY "%ENV_PATH%" "import torch, demucs, df, faster_whisper, mutagen; print('OK')" "%NAME%"
if errorlevel 1 goto :AFTER_LOCAL_LYRICS

REM ---- soundfile registers torchaudio's save/load backend on Windows; without
REM      it, torchaudio.save() inside demucs fails at runtime with
REM      "Couldn't find appropriate backend to handle uri ... .wav" even
REM      though every package above imports fine. Verify it's actually
REM      registered, not just installed.
call :INFO "%NAME%: verifying torchaudio audio backend is registered"
call conda run -p "%ENV_PATH%" python -c "import torchaudio; b = torchaudio.list_audio_backends(); assert b, 'no torchaudio audio backend registered'; print('backends:', b)" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    call :ERR "%NAME%: torchaudio has no registered audio backend - reinstalling soundfile"
    call :RUN_PROGRESS "pipinstall" "%ENV_PATH%" 5 "%NAME%: reinstalling soundfile" "soundfile" "--upgrade --force-reinstall" "none"
    call conda run -p "%ENV_PATH%" python -c "import torchaudio; b = torchaudio.list_audio_backends(); assert b, 'still no torchaudio audio backend registered'; print('backends:', b)" >> "%LOGFILE%" 2>&1
    if errorlevel 1 (
        call :ERR "%NAME%: torchaudio backend still missing after reinstall - see log (demucs will fail at runtime)"
        goto :AFTER_LOCAL_LYRICS
    )
)

call :TORCH_GPU_CHECK "%ENV_PATH%" "%NAME%"

>nul timeout /t 2
call :BLOCK_END
echo done > "%ENV_PATH%\.setup_ok"
call :OK "%NAME% environment ready"
goto :AFTER_LOCAL_LYRICS

:SKIP_LOCAL_LYRICS
call :SKIP "%NAME%"

:AFTER_LOCAL_LYRICS

echo.
echo %C_BOLD%[3/3] Summary%C_RESET%
echo.
call :LOG "==== SETUP RUN FINISHED : %date% %time% ===="
echo %C_CYAN%================================================================%C_RESET%
echo   Setup finished. Check console messages above for %C_RED%[ERROR]%C_RESET%/%C_YELLOW%[WARN]%C_RESET%.
echo   Full details, tracebacks and pip output: %LOGFILE%
echo %C_CYAN%================================================================%C_RESET%
echo.
echo   To run a script manually with its environment:
echo     conda run -p "%VENVS_DIR%\SCRIPT_NAME" python "%SCRIPTS_DIR%\SCRIPT_FILE.py"
echo   or:
echo     conda activate "%VENVS_DIR%\SCRIPT_NAME"
echo.
echo   %C_DIM%all set up, happy tagging (｡•̀ᴗ-)✧%C_RESET%
echo.
del "%GET_ROW_PS1%" >nul 2>&1
del "%CLEAR_BLOCK_PS1%" >nul 2>&1
del "%RUN_PROGRESS_PY%" >nul 2>&1
pause
exit /b 0


REM ############################################################################
REM  SUBROUTINES
REM ############################################################################

:LOG
>> "%LOGFILE%" echo [%date% %time%] %~1
goto :eof

:HEADER
REM  %~1 = display name, %~2 = optional ANSI color (defaults to cyan)
set "H_COLOR=%~2"
if "%H_COLOR%"=="" set "H_COLOR=%C_CYAN%"
echo %H_COLOR%----------------------------------------------------------------%C_RESET%
echo %H_COLOR%  %C_BOLD%%~1%C_RESET%
echo %H_COLOR%----------------------------------------------------------------%C_RESET%
call :LOG "===================== %~1 ====================="
goto :eof

:OK
echo   %C_GREEN%[OK]%C_RESET%    %~1
call :LOG "[OK] %~1"
goto :eof

:ERR
echo   %C_RED%[ERROR]%C_RESET% %~1
call :LOG "[ERROR] %~1"
goto :eof

:WARN
echo   %C_YELLOW%[WARN]%C_RESET%  %~1
call :LOG "[WARN] %~1"
goto :eof

REM ===========================================================================
REM  :BLOCK_START / :BLOCK_END
REM  Used around each venv's setup so the console can show live progress
REM  (what's being downloaded/installed) while it happens, then collapse
REM  down to just the final "[OK] ... environment ready" line once done -
REM  the full detail always stays in the log either way.
REM
REM  :BLOCK_START records the console row the block's progress output is
REM  about to start on. :BLOCK_END blanks every row from there down to
REM  (but not including) wherever the cursor ended up, then resets the
REM  cursor back to that same starting row so the next line printed (the
REM  final "[OK] ... environment ready") lands right where the progress
REM  output begins. This also incidentally clears the stray,
REM  harmless "The system cannot find the file specified." line some large
REM  conda/pip installs write to the console asynchronously (it never goes
REM  through stdout/stderr redirection) - since the whole range gets
REM  wiped regardless of its contents, there's nothing to match or miss.
REM ===========================================================================
:BLOCK_START
set "ROWFILE=%TEMP%\block_row_%RANDOM%.txt"
powershell -NoProfile -ExecutionPolicy Bypass -File "%GET_ROW_PS1%" > "%ROWFILE%" 2>nul
set "BLOCK_START_ROW="
if exist "%ROWFILE%" set /p BLOCK_START_ROW=<"%ROWFILE%"
del "%ROWFILE%" >nul 2>&1
goto :eof

:BLOCK_END
set "ROWFILE=%TEMP%\block_row_%RANDOM%.txt"
powershell -NoProfile -ExecutionPolicy Bypass -File "%GET_ROW_PS1%" > "%ROWFILE%" 2>nul
set "BLOCK_END_ROW="
if exist "%ROWFILE%" set /p BLOCK_END_ROW=<"%ROWFILE%"
del "%ROWFILE%" >nul 2>&1
if "%BLOCK_START_ROW%"=="" goto :eof
if "%BLOCK_END_ROW%"=="" goto :eof
if "%BLOCK_END_ROW%"=="%BLOCK_START_ROW%" goto :eof
powershell -NoProfile -ExecutionPolicy Bypass -File "%CLEAR_BLOCK_PS1%" %BLOCK_START_ROW% %BLOCK_END_ROW% >nul 2>&1
goto :eof

:INFO
echo   %C_CYAN%[INFO]%C_RESET%  %~1
call :LOG "[INFO] %~1"
goto :eof

REM ---------------------------------------------------------------------------
REM  :RUN_PROGRESS  <Mode>  <EnvPath>  <SizeMB>  <Label>  <Arg1>  <Arg2>  <Arg3>
REM  Runs one conda/pip step through RUN_PROGRESS_PY: a tqdm progress bar,
REM  colour-matched to the current environment's header colour (%ENV_COLOR%,
REM  set once per section - see each "call :HEADER" block), renders on the
REM  console while the real command runs. The real command's output still
REM  goes to %LOGFILE%. Returns the real command's exit code as this call's
REM  errorlevel, so every "if errorlevel 1 ..." check after a call site
REM  works correctly. Falls back to running
REM  the command directly (no bar) if conda's base python.exe couldn't be
REM  found during the requirements check.
REM
REM  Mode meanings (unused Arg slots are passed as "none"):
REM    condacreate   Arg1=python version                (e.g. 3.11)
REM    pipupgrade    (no Args used)
REM    pipinstall    Arg1=packages (space separated)     Arg2=extra flags/URL
REM    condainstall  Arg1=channel  Arg2=packages          Arg3=extra flags
REM ---------------------------------------------------------------------------
:RUN_PROGRESS
set "RP_MODE=%~1"
set "RP_ENV=%~2"
set "RP_SIZE=%~3"
set "RP_LABEL=%~4"
set "RP_ARG1=%~5"
set "RP_ARG2=%~6"
set "RP_ARG3=%~7"
call :INFO "%RP_LABEL%"
if not defined BASE_PY goto :RUN_PROGRESS_FALLBACK

"%BASE_PY%" "%RUN_PROGRESS_PY%" "%RP_MODE%" "%RP_ENV%" %RP_SIZE% "%ENV_COLOR%" "%LOGFILE%" "%RP_ARG1%" "%RP_ARG2%" "%RP_ARG3%"
set "RP_EXIT=%ERRORLEVEL%"
exit /b %RP_EXIT%

:RUN_PROGRESS_FALLBACK
REM conda base python.exe unavailable - run directly with no bar, using
REM simple best-effort package quoting.
if /I "%RP_MODE%"=="condacreate" (
    call conda create --prefix "%RP_ENV%" python=%RP_ARG1% -y >> "%LOGFILE%" 2>&1
) else if /I "%RP_MODE%"=="pipupgrade" (
    call conda run -p "%RP_ENV%" python -m pip install --upgrade pip >> "%LOGFILE%" 2>&1
) else if /I "%RP_MODE%"=="pipinstall" (
    set "RPF_PKGS="
    for %%p in (%RP_ARG1%) do set "RPF_PKGS=!RPF_PKGS! "%%p""
    if "%RP_ARG2%"=="none" (
        call conda run -p "%RP_ENV%" python -m pip install !RPF_PKGS! >> "%LOGFILE%" 2>&1
    ) else (
        call conda run -p "%RP_ENV%" python -m pip install !RPF_PKGS! %RP_ARG2% >> "%LOGFILE%" 2>&1
    )
) else if /I "%RP_MODE%"=="condainstall" (
    set "RPF_PKGS="
    for %%p in (%RP_ARG2%) do set "RPF_PKGS=!RPF_PKGS! "%%p""
    if "%RP_ARG3%"=="none" (
        call conda install -p "%RP_ENV%" -c %RP_ARG1% !RPF_PKGS! -y >> "%LOGFILE%" 2>&1
    ) else (
        call conda install -p "%RP_ENV%" -c %RP_ARG1% !RPF_PKGS! -y %RP_ARG3% >> "%LOGFILE%" 2>&1
    )
)
set "RP_EXIT=%ERRORLEVEL%"
exit /b %RP_EXIT%

:SKIP
echo   %C_DIM%[SKIP]  %~1 already installed - skipping%C_RESET%
call :LOG "[SKIP] %~1 - .setup_ok marker found, skipping"
goto :eof

REM ---------------------------------------------------------------------------
REM  :CHECK_SYS_FFMPEG  <display_name>
REM  Confirms ffmpeg is reachable via the SAME lookup Python's subprocess.run
REM  uses (PATH search), specifically for envs that rely on the system copy
REM  instead of installing their own via conda-forge. Non-fatal - just warns.
REM ---------------------------------------------------------------------------
:CHECK_SYS_FFMPEG
where ffmpeg >nul 2>>"%LOGFILE%"
if errorlevel 1 goto :CHECK_SYS_FFMPEG_MISSING
call :INFO "%~1: system ffmpeg found on PATH"
goto :eof

:CHECK_SYS_FFMPEG_MISSING
call :WARN "%~1: system ffmpeg NOT found on PATH - the script will fail at runtime"
echo      Download from https://www.gyan.dev/ffmpeg/builds/ and add its bin
echo      folder to PATH, then open a NEW terminal before running the script.
goto :eof

REM ---------------------------------------------------------------------------
REM  :VERIFY  <env_path>  <python_code>  <display_name>
REM  Runs a python -c import check inside the env, prints OK/ERROR.
REM  Returns errorlevel 1 on failure so the caller can "goto" away.
REM ---------------------------------------------------------------------------
:VERIFY
set "V_PATH=%~1"
set "V_CODE=%~2"
set "V_NAME=%~3"
call conda run -p "%V_PATH%" python -c "%V_CODE%" >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :VERIFY_FAIL
call :OK "%V_NAME%: package imports verified"
exit /b 0

:VERIFY_FAIL
call :ERR "%V_NAME%: import verification FAILED - see log for the traceback"
exit /b 1

REM ---------------------------------------------------------------------------
REM  :TORCH_GPU_CHECK  <env_path>  <display_name>
REM ---------------------------------------------------------------------------
:TORCH_GPU_CHECK
set "T_PATH=%~1"
set "T_NAME=%~2"
set "TMPFILE=%TEMP%\gpu_check_%RANDOM%.txt"
call conda run -p "%T_PATH%" python -c "import torch;print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO')" > "%TMPFILE%" 2>nul
set "T_RESULT="
if exist "%TMPFILE%" set /p T_RESULT=<"%TMPFILE%"
del "%TMPFILE%" >nul 2>&1
call :LOG "%T_NAME%: torch GPU check result = %T_RESULT%"
if "%T_RESULT%"=="CUDA_OK" goto :TORCH_GPU_YES
call :WARN "%T_NAME%: torch did NOT detect a CUDA GPU"
echo      Check that the NVIDIA driver is up to date and this machine has a
echo      CUDA-capable GPU. faster-whisper/demucs will fall back to CPU.
goto :eof

:TORCH_GPU_YES
call :INFO "%T_NAME%: torch reports CUDA GPU is available"
goto :eof

REM ---------------------------------------------------------------------------
REM  :TF_GPU_CHECK  <env_path>  <display_name>
REM ---------------------------------------------------------------------------
:TF_GPU_CHECK
set "G_PATH=%~1"
set "G_NAME=%~2"
set "TMPFILE=%TEMP%\tf_gpu_check_%RANDOM%.txt"
call conda run -p "%G_PATH%" python -c "import tensorflow as tf;print('GPU_OK' if len(tf.config.list_physical_devices('GPU'))>0 else 'GPU_NO')" > "%TMPFILE%" 2>nul
set "G_RESULT="
if exist "%TMPFILE%" set /p G_RESULT=<"%TMPFILE%"
del "%TMPFILE%" >nul 2>&1
call :LOG "%G_NAME%: tensorflow GPU check result = %G_RESULT%"
if "%G_RESULT%"=="GPU_OK" goto :TF_GPU_YES
call :WARN "%G_NAME%: tensorflow did NOT detect a GPU device"
echo      Common cause: missing zlibwapi.dll required by cuDNN 8.1 on Windows.
echo      See https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html
echo      MusicNN will still run, just on CPU (slower).
goto :eof

:TF_GPU_YES
call :INFO "%G_NAME%: tensorflow reports a GPU device is available"
goto :eof