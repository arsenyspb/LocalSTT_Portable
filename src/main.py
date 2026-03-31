import ctypes
import json
import logging
import os
import queue
import sys
import threading
import time
import wave
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
from pynput.keyboard import Controller, Key


@dataclass
class AppConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    model_size: str = "small"
    model_path: str = "models/faster-whisper-small"
    offline_only: bool = True
    compute_type: str = "int8"
    device: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    paste_delay_sec: float = 0.2
    restore_clipboard: bool = True
    input_device: str | None = None
    popup_duration_sec: float = 1.2


class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue[str]) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record))
        except Exception:
            pass


def _to_int_if_numeric(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _normalize_device_name(name: str) -> str:
    cleaned = (name or "").strip()
    marker = " (@"
    pos = cleaned.find(marker)
    if pos > 0:
        cleaned = cleaned[:pos].strip()
    cleaned = " ".join(cleaned.split())
    return cleaned


class LocalSTTApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        if getattr(sys, "frozen", False):
            self.project_root = Path(sys.executable).resolve().parent
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        local_app_data = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "LocalSTT"
        self.recordings_dir = local_app_data / "recordings"
        self.logs_dir = local_app_data / "logs"
        self.settings_file = local_app_data / "settings.json"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self._configure_logging(self.logs_dir / "app.log")

        pyautogui.FAILSAFE = False

        self.is_recording = False
        self.is_transcribing = False
        self.stop_event = threading.Event()

        self.audio_chunks: list[np.ndarray] = []
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.popup_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.mic_monitor_stream: sd.InputStream | None = None
        self.mic_level: float = 0.0

        self.last_audio_file: Path | None = None
        self.keyboard_controller = Controller()
        self.target_hwnd: int | None = None
        self.action_lock = threading.Lock()

        self.ui_root = None
        self.ui_log_text = None
        self.ui_tabs = None
        self.ui_mic_var = None
        self.ui_mic_combo = None
        self.ui_mic_level_var = None
        self.ui_mic_test_btn = None
        self.ui_vad_var = None
        self.ui_restore_clipboard_var = None
        self.ui_status_var = None
        self.ui_toast_window = None
        self.ui_toast_label = None
        self.ui_toast_timer_id = None

        self._load_settings_from_file()

        model_source, local_files_only = self._resolve_model_source()
        if self.config.offline_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        logging.info("Loading faster-whisper model from: %s", model_source)
        self.model = WhisperModel(
            model_source,
            device=self.config.device,
            compute_type=self.config.compute_type,
            local_files_only=local_files_only,
        )
        logging.info("Model loaded")

        self.input_device = self._resolve_input_device(self.config.input_device)
        self._log_audio_input_info()

        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "<ctrl>+<shift>+q": lambda: self._dispatch_hotkey("toggle_recording", self.toggle_recording),
                "<ctrl>+<shift>+w": lambda: self._dispatch_hotkey("transcribe_last", self.transcribe_last_file),
                "<ctrl>+<shift>+e": lambda: self._dispatch_hotkey("shutdown", self.shutdown),
            }
        )

    def _runtime_roots(self) -> list[Path]:
        roots: list[Path] = []
        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                roots.append(Path(meipass))
            roots.append(self.project_root / "_internal")
        roots.append(self.project_root)
        return roots

    def _resolve_existing_path(self, relative_path: str) -> Path | None:
        rel = Path(relative_path)
        for root in self._runtime_roots():
            candidate = root / rel
            if candidate.exists():
                return candidate
        return None

    def _resolve_model_source(self):
        model_path = Path(self.config.model_path)
        if not model_path.is_absolute():
            found = self._resolve_existing_path(str(model_path))
            if found is not None:
                model_path = found
            else:
                model_path = self.project_root / model_path

        if model_path.exists():
            return str(model_path), True

        if self.config.offline_only:
            raise FileNotFoundError(
                f"Offline model not found at '{model_path}'. Download model once before startup."
            )

        return self.config.model_size, False

    def _resolve_icon_path(self) -> Path | None:
        candidates = [
            Path(__file__).resolve().parent / "icon.png",
            self.project_root / "src" / "icon.png",
        ]
        for root in self._runtime_roots():
            candidates.append(root / "src" / "icon.png")

        for p in candidates:
            if p.exists():
                return p
        return None

    def _configure_logging(self, log_file: Path) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        queue_handler = QueueLogHandler(self.log_queue)
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)

    def _load_settings_from_file(self) -> None:
        try:
            if not self.settings_file.exists():
                return
            with self.settings_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.config.input_device = data.get("input_device", self.config.input_device)
            self.config.vad_filter = bool(data.get("vad_filter", self.config.vad_filter))
            self.config.restore_clipboard = bool(data.get("restore_clipboard", self.config.restore_clipboard))
            self.config.model_size = str(data.get("model_size", self.config.model_size))
            self.config.model_path = str(data.get("model_path", self.config.model_path))
        except Exception:
            logging.exception("Failed to load settings")

    def _save_settings_to_file(self) -> None:
        try:
            payload: dict[str, Any] = {
                "input_device": self.config.input_device,
                "vad_filter": self.config.vad_filter,
                "restore_clipboard": self.config.restore_clipboard,
                "model_size": self.config.model_size,
                "model_path": self.config.model_path,
            }
            with self.settings_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logging.exception("Failed to save settings")

    def _dispatch_hotkey(self, hotkey_name: str, action) -> None:
        logging.info("Hotkey pressed: %s", hotkey_name)

        def _runner() -> None:
            try:
                with self.action_lock:
                    action()
            except Exception:
                logging.exception("Hotkey action failed: %s", hotkey_name)

        threading.Thread(target=_runner, daemon=True).start()

    def _audio_callback(self, indata: np.ndarray, frames: int, callback_time, status) -> None:
        if status:
            logging.warning("Audio status: %s", status)
        self.audio_queue.put(indata.copy())

    def _resolve_input_device(self, input_device: str | None):
        if input_device is None or input_device.strip() == "":
            return None
        return _to_int_if_numeric(input_device.strip())

    def _input_devices_list(self) -> list[tuple[int, str]]:
        result: list[tuple[int, str]] = []
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) > 0:
                    raw_name = str(dev.get("name", f"Input {idx}"))
                    hostapi_index = int(dev.get("hostapi", -1))
                    hostapi_name = ""
                    if hostapi_index >= 0:
                        try:
                            hostapi_name = str(sd.query_hostapis(hostapi_index).get("name", "")).strip()
                        except Exception:
                            hostapi_name = ""

                    display_name = _normalize_device_name(raw_name)
                    if hostapi_name:
                        display_name = f"{display_name} [{hostapi_name}]"

                    result.append((idx, display_name))
        except Exception:
            logging.exception("Failed to enumerate input devices")
        return result

    def _find_first_input_device(self):
        for idx, _name in self._input_devices_list():
            return idx
        return None

    def _has_any_input_device(self) -> bool:
        return len(self._input_devices_list()) > 0

    def _get_effective_input_device(self):
        if self.input_device is not None:
            return self.input_device
        try:
            default_in, _ = sd.default.device
            if isinstance(default_in, int) and default_in >= 0:
                return default_in
        except Exception:
            logging.exception("Could not read default audio device")
        return self._find_first_input_device()

    def _iter_input_device_candidates(self):
        seen = set()
        preferred = self._get_effective_input_device()
        if preferred is not None:
            preferred = _to_int_if_numeric(preferred)
            if preferred not in seen:
                seen.add(preferred)
                yield preferred
        for idx, _name in self._input_devices_list():
            if idx not in seen:
                seen.add(idx)
                yield idx

    def _sample_rate_candidates(self, device):
        candidates = [int(self.config.sample_rate)]
        try:
            info = sd.query_devices(device, "input")
            default_sr = int(float(info.get("default_samplerate", 0) or 0))
            if default_sr > 0 and default_sr not in candidates:
                candidates.append(default_sr)
        except Exception:
            logging.exception("Could not query sample rate for device %s", device)

        for sr in [16000, 48000, 44100, 8000]:
            if sr not in candidates:
                candidates.append(sr)
        return candidates

    def _open_input_stream_with_fallback(self):
        last_error = None
        for device in self._iter_input_device_candidates():
            for rate in self._sample_rate_candidates(device):
                try:
                    sd.check_input_settings(
                        device=device,
                        channels=self.config.channels,
                        dtype=self.config.dtype,
                        samplerate=rate,
                    )
                    stream = sd.InputStream(
                        samplerate=rate,
                        channels=self.config.channels,
                        dtype=self.config.dtype,
                        device=device,
                        callback=self._audio_callback,
                    )
                    stream.start()
                    return stream, device, rate
                except Exception as exc:
                    last_error = exc
                    logging.warning("Input probe failed: device=%s samplerate=%s error=%s", device, rate, exc)
        if last_error is not None:
            raise last_error
        raise RuntimeError("No input devices available")

    def _log_audio_input_info(self) -> None:
        try:
            effective = self._get_effective_input_device()
            if effective is None:
                logging.error("No input microphone device found")
                return
            info = sd.query_devices(effective, "input")
            logging.info("Using microphone: %s", info.get("name", effective))
        except Exception:
            logging.exception("Could not resolve input device")

    def _release_modifiers(self) -> None:
        for key in [Key.ctrl, Key.shift, Key.alt, Key.cmd]:
            try:
                self.keyboard_controller.release(key)
            except Exception:
                pass
        for key_name in ["ctrl", "shift", "alt", "winleft", "winright"]:
            try:
                pyautogui.keyUp(key_name)
            except Exception:
                pass

    def _drain_audio_queue(self) -> None:
        while not self.audio_queue.empty():
            self.audio_chunks.append(self.audio_queue.get_nowait())

    def _get_foreground_window(self) -> int | None:
        try:
            hwnd = int(ctypes.windll.user32.GetForegroundWindow())
            return hwnd if hwnd != 0 else None
        except Exception:
            return None

    def _get_window_title(self, hwnd: int | None) -> str:
        if hwnd is None:
            return ""
        try:
            user32 = ctypes.windll.user32
            length = int(user32.GetWindowTextLengthW(hwnd))
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        except Exception:
            return ""

    def _get_window_class(self, hwnd: int | None) -> str:
        if hwnd is None:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetClassNameW(hwnd, buf, 255)
            return buf.value
        except Exception:
            return ""

    def _get_window_pid(self, hwnd: int | None) -> int | None:
        if hwnd is None:
            return None
        try:
            pid = wintypes.DWORD(0)
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return int(pid.value)
        except Exception:
            return None

    def _describe_window(self, hwnd: int | None) -> str:
        if hwnd is None:
            return "hwnd=None"
        return (
            f"hwnd={hwnd} pid={self._get_window_pid(hwnd)} "
            f"class='{self._get_window_class(hwnd)}' title='{self._get_window_title(hwnd)}'"
        )

    def _capture_target_window(self, reason: str) -> None:
        hwnd = self._get_foreground_window()
        if hwnd is None:
            logging.info("Target capture skipped (%s): no foreground window", reason)
            return
        if self._get_window_pid(hwnd) == os.getpid():
            logging.info("Target capture skipped (%s): foreground is app process", reason)
            return
        self.target_hwnd = hwnd
        logging.info("Target window captured (%s): %s", reason, self._describe_window(hwnd))

    def _activate_window(self, hwnd: int | None) -> None:
        if hwnd is None:
            return
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)

            current_foreground = user32.GetForegroundWindow()
            current_thread = user32.GetWindowThreadProcessId(current_foreground, None)
            target_thread = user32.GetWindowThreadProcessId(hwnd, None)
            this_thread = kernel32.GetCurrentThreadId()

            attached_current = False
            attached_target = False
            try:
                if current_thread and current_thread != this_thread:
                    attached_current = bool(user32.AttachThreadInput(this_thread, current_thread, True))
                if target_thread and target_thread != this_thread:
                    attached_target = bool(user32.AttachThreadInput(this_thread, target_thread, True))

                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                user32.SetFocus(hwnd)
                time.sleep(0.06)
            finally:
                if attached_current:
                    user32.AttachThreadInput(this_thread, current_thread, False)
                if attached_target:
                    user32.AttachThreadInput(this_thread, target_thread, False)
        except Exception:
            logging.warning("Could not activate target window")

    def _send_vk(self, vk: int, key_up: bool = False) -> None:
        flags = 0x0002 if key_up else 0
        ctypes.windll.user32.keybd_event(vk, 0, flags, 0)

    def _send_shortcut_vk(self, modifier_vk: int, key_vk: int) -> None:
        self._send_vk(modifier_vk, key_up=False)
        time.sleep(0.01)
        self._send_vk(key_vk, key_up=False)
        time.sleep(0.01)
        self._send_vk(key_vk, key_up=True)
        time.sleep(0.01)
        self._send_vk(modifier_vk, key_up=True)

    def _send_wm_paste(self, hwnd: int | None) -> bool:
        if hwnd is None:
            return False
        try:
            ctypes.windll.user32.SendMessageW(hwnd, 0x0302, 0, 0)
            return True
        except Exception:
            logging.exception("WM_PASTE failed")
            return False

    def _generate_audio_path(self) -> Path:
        return self.recordings_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    def start_recording(self) -> None:
        if self.is_recording:
            return
        if not self._has_any_input_device():
            logging.error("No microphone found")
            self._set_status("Microphone not found")
            self._show_popup("NO MICROPHONE FOUND", bg="#9a1b1b")
            return

        self.audio_chunks.clear()
        self.audio_queue = queue.Queue()

        try:
            self.stream, chosen_device, chosen_rate = self._open_input_stream_with_fallback()
            self.is_recording = True
            logging.info("Recording started (device=%s, samplerate=%s)", chosen_device, chosen_rate)
            self._set_status("Recording...")
            self._show_popup("RECORDING...", bg="#9a1b1b", persistent=True)
        except Exception:
            logging.exception("Failed to start recording")
            self._set_status("Microphone open failed")
            self._show_popup("MICROPHONE OPEN FAILED", bg="#9a1b1b")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        assert self.stream is not None
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.is_recording = False

        self._drain_audio_queue()
        if not self.audio_chunks:
            logging.warning("Recording stopped but no audio data was captured")
            self._set_status("No audio captured")
            self._show_popup("NO AUDIO CAPTURED", bg="#7d5a11")
            return

        audio_data = np.concatenate(self.audio_chunks, axis=0)
        if audio_data.ndim > 1 and audio_data.shape[1] == 1:
            audio_data = audio_data[:, 0]

        output_path = self._generate_audio_path()
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(np.dtype(self.config.dtype).itemsize)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        self.last_audio_file = output_path
        duration_sec = len(audio_data) / float(self.config.sample_rate)
        logging.info("Recording saved: %s (%.2f sec)", output_path, duration_sec)
        self._set_status("Recording stopped")
        self._show_popup("RECORDING STOPPED", bg="#1e6b2d", persistent=True)

        self.transcribe_file_async(output_path)

    def toggle_recording(self) -> None:
        try:
            if self.is_recording:
                self.stop_recording()
                return
            self._capture_target_window("toggle_recording_hotkey_start")
            self.start_recording()
        except Exception:
            logging.exception("Toggle recording failed")

    def transcribe_last_file(self) -> None:
        try:
            self._capture_target_window("transcribe_last_hotkey")
            if self.last_audio_file is None:
                logging.warning("No previous recording found")
                self._set_status("No last recording")
                self._show_popup("NO LAST RECORDING", bg="#7d5a11")
                return
            self._set_status("Transcribing...")
            self._show_popup("TRANSCRIBING LAST FILE...", bg="#0d5f8a")
            self.transcribe_file_async(self.last_audio_file)
        except Exception:
            logging.exception("Transcribe last failed")
            self._show_popup("TRANSCRIBE ERROR", bg="#9a1b1b")

    def transcribe_file_async(self, audio_path: Path) -> None:
        if self.is_transcribing:
            logging.warning("Transcription already in progress")
            return
        threading.Thread(target=self._transcribe_and_paste, args=(audio_path,), daemon=True).start()

    def _transcribe_and_paste(self, audio_path: Path) -> None:
        self.is_transcribing = True
        start = time.perf_counter()
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
            )
            text = "".join(segment.text for segment in segments).strip()
            elapsed = time.perf_counter() - start
            logging.info(
                "Transcription done in %.2f sec | detected language=%s prob=%.3f | chars=%d",
                elapsed,
                getattr(info, "language", "unknown"),
                float(getattr(info, "language_probability", 0.0)),
                len(text),
            )
            if text:
                self._paste_text(text)
                logging.info("Text pasted to active window")
                self._set_status("Done")
                self._hide_popup()
            else:
                logging.warning("Transcription returned empty text")
                self._set_status("Empty transcription")
                self._show_popup("EMPTY TRANSCRIPTION", bg="#7d5a11")
        except Exception as exc:
            if self.config.vad_filter and "silero_vad_v6.onnx" in str(exc):
                try:
                    logging.warning("VAD asset missing, retrying without VAD")
                    segments, info = self.model.transcribe(
                        str(audio_path),
                        beam_size=self.config.beam_size,
                        vad_filter=False,
                    )
                    text = "".join(segment.text for segment in segments).strip()
                    if text:
                        self._paste_text(text)
                        logging.info("Text pasted to active window")
                        self._set_status("Done (without VAD)")
                        self._hide_popup()
                    else:
                        self._set_status("Empty transcription")
                        self._show_popup("EMPTY TRANSCRIPTION", bg="#7d5a11")
                except Exception:
                    logging.exception("Transcription failed")
                    self._set_status("Transcription failed")
                    self._show_popup("TRANSCRIPTION FAILED", bg="#9a1b1b")
            else:
                logging.exception("Transcription failed")
                self._set_status("Transcription failed")
                self._show_popup("TRANSCRIPTION FAILED", bg="#9a1b1b")
        finally:
            self.is_transcribing = False

    def _paste_text(self, text: str) -> None:
        previous_clipboard = None
        try:
            previous_clipboard = pyperclip.paste()
        except Exception:
            logging.warning("Could not read clipboard")

        pyperclip.copy(text)
        time.sleep(max(0.25, self.config.paste_delay_sec))

        if self.target_hwnd is None:
            self.target_hwnd = self._get_foreground_window()

        self._activate_window(self.target_hwnd)
        time.sleep(max(0.3, self.config.paste_delay_sec))
        self._release_modifiers()
        time.sleep(0.08)

        self._send_wm_paste(self.target_hwnd)
        time.sleep(max(0.12, self.config.paste_delay_sec))
        self._send_shortcut_vk(0x11, 0x56)
        time.sleep(max(0.12, self.config.paste_delay_sec))
        self._send_shortcut_vk(0x10, 0x2D)

        if self.config.restore_clipboard and previous_clipboard is not None:
            try:
                time.sleep(max(0.6, self.config.paste_delay_sec))
                pyperclip.copy(previous_clipboard)
            except Exception:
                logging.warning("Could not restore clipboard")

    def _set_status(self, text: str) -> None:
        if self.ui_status_var is not None:
            try:
                self.ui_status_var.set(text)
            except Exception:
                pass
        logging.info("STATUS: %s", text)

    def _show_popup(self, text: str, bg: str = "#1e6b2d", persistent: bool = False) -> None:
        self.popup_queue.put(("show", {"text": text, "bg": bg, "persistent": persistent}))

    def _hide_popup(self) -> None:
        self.popup_queue.put(("hide", {}))

    def _process_popup_queue(self) -> None:
        if self.ui_toast_window is None or self.ui_toast_label is None or self.ui_root is None:
            return

        def _cancel_timer() -> None:
            if self.ui_toast_timer_id is not None:
                try:
                    self.ui_root.after_cancel(self.ui_toast_timer_id)
                except Exception:
                    pass
                self.ui_toast_timer_id = None

        while True:
            try:
                action, payload = self.popup_queue.get_nowait()
            except queue.Empty:
                break

            if action == "show":
                _cancel_timer()
                text = str(payload.get("text", ""))
                bg = str(payload.get("bg", "#1e6b2d"))
                persistent = bool(payload.get("persistent", False))

                self.ui_toast_label.configure(text=text, bg=bg)
                self.ui_toast_window.configure(bg=bg)
                self.ui_toast_window.deiconify()
                self.ui_toast_window.lift()

                if not persistent:
                    timeout_ms = int(self.config.popup_duration_sec * 1000)

                    def _hide_after() -> None:
                        if self.ui_toast_window is not None:
                            self.ui_toast_window.withdraw()
                        self.ui_toast_timer_id = None

                    self.ui_toast_timer_id = self.ui_root.after(timeout_ms, _hide_after)

            elif action == "hide":
                _cancel_timer()
                self.ui_toast_window.withdraw()

    def _mic_test_callback(self, indata: np.ndarray, frames: int, callback_time, status) -> None:
        if status:
            logging.warning("Mic test status: %s", status)
        try:
            data = indata.astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(data))))
            self.mic_level = min(100.0, max(0.0, rms / 327.68))
        except Exception:
            self.mic_level = 0.0

    def _start_mic_test(self) -> None:
        if self.mic_monitor_stream is not None:
            return
        try:
            device = self._get_effective_input_device()
            if device is None:
                self._set_status("No microphone available")
                return
            self.mic_monitor_stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                device=device,
                callback=self._mic_test_callback,
            )
            self.mic_monitor_stream.start()
            self._set_status("Microphone test enabled")
        except Exception:
            logging.exception("Failed to start mic test")
            self._set_status("Failed to start microphone test")
            self.mic_monitor_stream = None

    def _stop_mic_test(self) -> None:
        if self.mic_monitor_stream is None:
            return
        try:
            self.mic_monitor_stream.stop()
            self.mic_monitor_stream.close()
        except Exception:
            pass
        self.mic_monitor_stream = None
        self.mic_level = 0.0
        self._set_status("Microphone test disabled")

    def _toggle_mic_test(self) -> None:
        if self.mic_monitor_stream is None:
            self._start_mic_test()
            if self.ui_mic_test_btn is not None:
                self.ui_mic_test_btn.configure(text="Stop test")
        else:
            self._stop_mic_test()
            if self.ui_mic_test_btn is not None:
                self.ui_mic_test_btn.configure(text="Test microphone")

    def _apply_selected_mic(self) -> None:
        if self.ui_mic_var is None:
            return
        value = self.ui_mic_var.get().strip()
        if not value:
            return
        idx = value.split(":", 1)[0].strip()
        self.config.input_device = idx
        self.input_device = self._resolve_input_device(idx)
        self._save_settings_to_file()
        self._set_status(f"Microphone selected: {idx}")
        self._log_audio_input_info()

    def _refresh_mic_devices(self) -> None:
        if self.ui_mic_combo is None:
            return
        items = [f"{idx}: {name}" for idx, name in self._input_devices_list()]
        self.ui_mic_combo["values"] = items

        selected = str(self.config.input_device) if self.config.input_device is not None else ""
        if selected:
            for item in items:
                if item.startswith(f"{selected}:"):
                    self.ui_mic_var.set(item)
                    break
        elif items:
            self.ui_mic_var.set(items[0])

    def _ui_poll(self) -> None:
        if self.stop_event.is_set() or self.ui_root is None:
            return

        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if self.ui_log_text is not None:
                self.ui_log_text.configure(state="normal")
                self.ui_log_text.insert("end", line + "\n")
                self.ui_log_text.see("end")
                self.ui_log_text.configure(state="disabled")

        if self.ui_mic_level_var is not None:
            self.ui_mic_level_var.set(self.mic_level)

        self._process_popup_queue()

        self.ui_root.after(100, self._ui_poll)

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        self.ui_root = root
        root.title("LocalSTT")

        width = 450
        height = 600
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        margin_right = 24

        x = max(0, screen_w - width - margin_right)
        y = max(0, (screen_h - height) // 2)

        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(450, 600)
        root.maxsize(450, 600)

        icon_path = self._resolve_icon_path()
        if icon_path is not None:
            try:
                icon_img = tk.PhotoImage(file=str(icon_path))
                root.iconphoto(True, icon_img)
                root._icon_img_ref = icon_img
            except Exception:
                logging.warning("Failed to set window icon")

        # Top floating notification line.
        toast = tk.Toplevel(root)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.withdraw()

        toast_w = 280
        toast_h = 42
        toast_x = max(0, (screen_w - toast_w) // 2)
        toast_y = 40
        toast.geometry(f"{toast_w}x{toast_h}+{toast_x}+{toast_y}")

        toast_label = tk.Label(
            toast,
            text="",
            bg="#1e6b2d",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=8,
        )
        toast_label.pack(fill="both", expand=True)

        self.ui_toast_window = toast
        self.ui_toast_label = toast_label

        container = ttk.Frame(root, padding=6)
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)
        self.ui_tabs = notebook

        tab_mic = ttk.Frame(notebook)
        tab_desc = ttk.Frame(notebook)
        tab_logs = ttk.Frame(notebook)
        notebook.add(tab_mic, text="Microphone")
        notebook.add(tab_desc, text="Description")
        notebook.add(tab_logs, text="Logs")

        log_text = tk.Text(tab_logs, wrap="word", font=("Consolas", 8), state="disabled")
        log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.ui_log_text = log_text

        desc = tk.Text(tab_desc, wrap="word", font=("Segoe UI", 9))
        desc.pack(fill="both", expand=True, padx=6, pady=6)
        desc.insert(
            "1.0",
            "LocalSTT portable\n\n"
            "Hotkeys:\n"
            "Ctrl+Shift+Q - start/stop recording\n"
            "Ctrl+Shift+W - transcribe last recording\n"
            "Ctrl+Shift+E - exit\n\n"
            "How it works:\n"
            "1) Start recording with a hotkey.\n"
            "2) After stop, audio is sent to faster-whisper.\n"
            "3) Recognized text is pasted into the active window.\n\n"
            "The Logs tab shows progress in real time."
        )
        desc.configure(state="disabled")

        mic_frame = ttk.Frame(tab_mic, padding=6)
        mic_frame.pack(fill="both", expand=True)

        ttk.Label(mic_frame, text="Input device:").pack(anchor="w")
        self.ui_mic_var = tk.StringVar()
        mic_combo = ttk.Combobox(mic_frame, textvariable=self.ui_mic_var, state="readonly")
        mic_combo.pack(fill="x", pady=(2, 6))
        self.ui_mic_combo = mic_combo

        row = ttk.Frame(mic_frame)
        row.pack(fill="x", pady=(0, 6))
        ttk.Button(row, text="Refresh", command=self._refresh_mic_devices).pack(side="left")
        ttk.Button(row, text="Select", command=self._apply_selected_mic).pack(side="left", padx=(6, 0))

        self.ui_mic_level_var = tk.DoubleVar(value=0.0)
        ttk.Label(mic_frame, text="Input level:").pack(anchor="w", pady=(6, 2))
        ttk.Progressbar(mic_frame, maximum=100.0, variable=self.ui_mic_level_var).pack(fill="x")

        self.ui_mic_test_btn = ttk.Button(mic_frame, text="Test microphone", command=self._toggle_mic_test)
        self.ui_mic_test_btn.pack(fill="x", pady=(8, 6))

        self.ui_vad_var = tk.BooleanVar(value=self.config.vad_filter)
        self.ui_restore_clipboard_var = tk.BooleanVar(value=self.config.restore_clipboard)

        def _apply_simple_settings() -> None:
            self.config.vad_filter = bool(self.ui_vad_var.get())
            self.config.restore_clipboard = bool(self.ui_restore_clipboard_var.get())
            self._save_settings_to_file()
            self._set_status("Settings saved")

        ttk.Checkbutton(mic_frame, text="VAD filter", variable=self.ui_vad_var).pack(anchor="w")
        ttk.Checkbutton(
            mic_frame,
            text="Restore clipboard",
            variable=self.ui_restore_clipboard_var,
        ).pack(anchor="w")
        ttk.Button(mic_frame, text="Save settings", command=_apply_simple_settings).pack(fill="x", pady=(6, 0))

        self.ui_status_var = tk.StringVar(value="Done")
        ttk.Label(container, textvariable=self.ui_status_var).pack(fill="x", pady=(6, 0))

        self._refresh_mic_devices()
        notebook.select(tab_logs)

        def _on_close() -> None:
            self.shutdown()

        root.protocol("WM_DELETE_WINDOW", _on_close)
        self._ui_poll()
        root.mainloop()

    def shutdown(self) -> None:
        if self.stop_event.is_set():
            return
        logging.info("Shutdown requested")
        self.stop_event.set()
        self._show_popup("LOCALSTT STOPPED", bg="#5d2f87")

        try:
            if self.is_recording:
                self.stop_recording()
        except Exception:
            logging.exception("Error while stopping recording")

        self._stop_mic_test()

        try:
            self.hotkeys.stop()
        except Exception:
            pass

        if self.ui_root is not None:
            try:
                if self.ui_toast_window is not None:
                    self.ui_toast_window.withdraw()
                self.ui_root.after(0, self.ui_root.destroy)
            except Exception:
                pass

    def run(self) -> None:
        logging.info("LocalSTT started")
        logging.info("Hotkeys: Ctrl+Shift+Q/W/E")
        self.hotkeys.start()
        self._build_ui()
        logging.info("LocalSTT stopped")


def main() -> None:
    offline_flag = os.environ.get("LOCALSTT_OFFLINE_ONLY", "1").strip().lower()
    offline_only = offline_flag in {"1", "true", "yes", "on"}

    restore_clipboard_flag = os.environ.get("LOCALSTT_RESTORE_CLIPBOARD", "1").strip().lower()
    restore_clipboard = restore_clipboard_flag in {"1", "true", "yes", "on"}

    config = AppConfig(
        model_size=os.environ.get("LOCALSTT_MODEL", "small"),
        model_path=os.environ.get("LOCALSTT_MODEL_PATH", "models/faster-whisper-small"),
        offline_only=offline_only,
        compute_type=os.environ.get("LOCALSTT_COMPUTE", "int8"),
        device=os.environ.get("LOCALSTT_DEVICE", "auto"),
        restore_clipboard=restore_clipboard,
        input_device=os.environ.get("LOCALSTT_INPUT_DEVICE"),
    )

    if "LOCALSTT_MODEL_PATH" not in os.environ:
        size_to_path = {
            "small": "models/faster-whisper-small",
            "medium": "models/faster-whisper-medium",
        }
        config.model_path = size_to_path.get(config.model_size, config.model_path)

    app = LocalSTTApp(config)
    app.run()


if __name__ == "__main__":
    main()
