# Documentation: Local STT Hotkey Script (Windows)

Date: 2026-03-15
Status: design phase (no code)

## 1. Short Answer to "Is This Realistic?"

Yes, it is realistic.

A Python script can be built that:
- starts with Windows;
- runs in the background;
- starts/stops microphone recording using a global hotkey;
- saves the full recording to an audio file (WAV);
- fully transcribes the file locally after recording stops (no cloud);
- inserts recognized text into the active input field at the current cursor position.

## 2. What We Build (MVP)

MVP features:
- Global hotkeys:
  - `Ctrl+Alt+R` -> start/stop recording.
  - `Ctrl+Alt+T` -> force transcription of the last file.
  - `Ctrl+Alt+Q` -> exit the background process.
- Full recording to file:
  - Format: `WAV`, mono, 16 kHz, PCM 16-bit.
  - Folder: `%LOCALAPPDATA%/LocalSTT/recordings/`.
- Full post-recording transcription:
  - Process the entire file after recording ends.
  - Recognition language is fixed: Russian only (`ru`).
  - Automatic language detection is disabled.
- Insert text at cursor position:
  - No manual copy/paste required by the user.
- Local operation:
  - Internet is not required during transcription.

## 3. Scope and Limitations

Out of MVP scope (possible later):
- streaming (on-the-fly) transcription during recording;
- punctuation/formatting via LLM;
- speaker diarization;
- GUI with advanced settings.

Limitations:
- quality depends heavily on microphone quality and background noise;
- larger STT models require more RAM/VRAM;
- hotkeys may conflict with system or application shortcuts.

## 4. Recommended Local Stack

Primary option (recommended):
- STT: `faster-whisper`
- Audio recording: `sounddevice`
- Global hotkeys: `pynput` (or `keyboard` as an alternative)
- Insert into active field: clipboard + `Ctrl+V` simulation (`pyperclip` + `pyautogui`)

Why this stack:
- `faster-whisper` is usually faster than classic `openai-whisper` on CPU/GPU;
- `sounddevice` is stable for simple recording workflows;
- `pynput` is often convenient for global hotkeys in a standard Windows user context.

## 5. STT Alternatives (Also Local)

### Option A: faster-whisper (recommended)
Mode for this project:
- Russian language only (`ru`);
- no automatic language detection;
- output should be Russian text.

Pros:
- good quality;
- wide model selection (tiny/base/small/medium/large);
- solid speed on modern CPU/GPU.

Cons:
- models require disk space;
- latency may be noticeable on weaker hardware.

### Option B: Vosk
Pros:
- very lightweight offline stack;
- quick startup on low-end PCs.

Cons:
- average recognition quality is lower than Whisper-family models.

## 6. Library Versions (Checked on 2026-03-15)

Reference snapshot:
- `faster-whisper`: `1.2.1` (release: 2025-10-31)
- `sounddevice`: `0.5.5` (release: 2026-01-23)
- `pynput`: `1.8.1` (release: 2025-03-17)
- `keyboard`: `0.13.5` (release: 2020-03-23)
- `pyautogui`: `0.9.54` (release: 2023-05-24)
- `pyperclip`: `1.11.0` (release: 2025-09-26)
- `vosk`: `0.3.45` (release: 2022-12-14)

Important: re-check versions before actual installation.

## 7. Script Architecture

Components:
- `HotkeyListener`
  - listens for global key combinations;
  - triggers required actions.
- `AudioRecorder`
  - starts/stops recording;
  - writes full WAV files.
- `TranscriptionEngine`
  - accepts a WAV path;
  - always runs in Russian language mode (`ru`);
  - returns text.
- `TextInjector`
  - inserts text into the current input focus.
- `AppController`
  - controls states (Idle/Recording/Transcribing);
  - logs events.

States:
- `Idle` -> waiting for hotkeys
- `Recording` -> recording in progress
- `Transcribing` -> recording completed, recognition in progress
- return to `Idle`

## 8. User Workflow

1. User presses the record hotkey.
2. The script starts microphone recording to a file.
3. User presses the record hotkey again (stop).
4. The script closes the WAV file.
5. The script transcribes the full file locally.
6. The script inserts recognized text into the active input field (at the cursor).
7. The script returns to waiting mode.

## 9. PC Setup and Auto-Start

### Option 1 (personal use): Python + venv
- Install Python 3.11+.
- Create a virtual environment.
- Install dependencies.
- Run the script manually or through a shortcut.

### Option 2 (installed-like flow): build to EXE
- Package the script into an `exe` (for example, with PyInstaller).
- Place the EXE in the application folder.
- Add auto-start:
  - via Task Scheduler (recommended), or
  - via the Startup folder.

Task Scheduler is recommended:
- trigger: "At log on";
- run with standard user privileges;
- restart on failure.

## 10. Cursor Text Insertion: Safe Approach

Best practice:
- save current clipboard contents;
- put recognized text into the clipboard;
- send `Ctrl+V` to the active window;
- restore clipboard contents when possible.

Benefit:
- works in most applications.

Risk:
- some apps/games block synthetic input;
- a small timing delay (50-150 ms) may be needed between steps.

## 11. Logs and Diagnostics

What to log:
- recording start/stop;
- file path;
- recording duration;
- transcription duration;
- result length;
- audio/model/insertion errors.

File:
- `%LOCALAPPDATA%/LocalSTT/logs/app.log`

## 12. Performance and Hardware

Practical guidelines for Whisper models:
- `tiny/base` -> faster, lower accuracy;
- `small/medium` -> balanced;
- `large` -> higher accuracy, significantly heavier.

For first run, starting with `small` is usually reasonable.

## 13. Risks and Mitigations

- Hotkey conflicts:
  - make key bindings configurable in settings.
- Low quality in noisy environments:
  - add input-device selection and noise suppression later.
- Slow transcription:
  - use a smaller model or GPU.
- Insertion failures:
  - fallback: show text in a notification or copy window.

## 14. MVP Readiness Criteria

MVP is considered ready if:
- the script starts reliably with Windows;
- start/stop hotkey works in at least 3 apps (for example: browser, messenger, notes);
- a full-recording WAV file is created;
- transcription runs fully locally;
- transcription runs strictly in Russian mode (`ru`) without auto-detection;
- text is inserted into the active field at the cursor;
- event and error logs are available.

## 15. Next Step

After approving this documentation, proceed to:
- create project structure (`src`, `config`, `logs`);
- implement MVP by modules;
- build EXE and configure auto-start.

## 16. Official Sources (Used for Validation)

- Faster-Whisper (GitHub Releases): https://github.com/SYSTRAN/faster-whisper/releases
- Vosk (official website): https://alphacephei.com/vosk/
- Vosk API (GitHub): https://github.com/alphacep/vosk-api
- SoundDevice (PyPI): https://pypi.org/project/sounddevice/
- Keyboard (PyPI): https://pypi.org/project/keyboard/
