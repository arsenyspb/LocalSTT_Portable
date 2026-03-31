import logging
import re
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class TranscriptionCancelledError(Exception):
    pass


def prepare_audio_samples(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if audio.size == 0:
        return np.array([], dtype=np.float32), sample_rate

    original_dtype = audio.dtype
    if audio.ndim > 1:
        if audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            audio = audio.reshape(-1, audio.shape[1]).mean(axis=1)

    audio = audio.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        if np.issubdtype(original_dtype, np.unsignedinteger):
            midpoint = float(info.max) / 2.0
            if midpoint > 0:
                audio = (audio - midpoint) / midpoint
        else:
            max_abs = float(max(abs(info.min), info.max))
            if max_abs > 0:
                audio /= max_abs
    audio = np.clip(audio, -1.0, 1.0)

    target_rate = 16000
    if sample_rate != target_rate and audio.size > 0:
        src_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_len = max(1, int(round(audio.shape[0] * (target_rate / float(sample_rate)))))
        dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)
        sample_rate = target_rate

    return audio.astype(np.float32), sample_rate


def load_audio_for_transcription(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_rate = int(wav_file.getframerate())
        sampwidth = int(wav_file.getsampwidth())
        raw = wav_file.readframes(wav_file.getnframes())

    dtype_map = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    dtype = dtype_map.get(sampwidth)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    audio = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    return prepare_audio_samples(audio, sample_rate)


def _build_chunk_ranges(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[tuple[int, int]]:
    if total_samples <= 0:
        return []

    overlap_samples = max(0, min(overlap_samples, chunk_samples - 1))
    step_samples = max(1, chunk_samples - overlap_samples)
    ranges: list[tuple[int, int]] = []
    start_idx = 0
    while start_idx < total_samples:
        end_idx = min(total_samples, start_idx + chunk_samples)
        ranges.append((start_idx, end_idx))
        if end_idx >= total_samples:
            break
        start_idx += step_samples
    return ranges


def _normalize_token_for_match(token: str) -> str:
    return re.sub(r"[\W_]+", "", token.casefold())


def merge_chunk_transcript(existing_text: str, new_text: str) -> str:
    existing = existing_text.strip()
    incoming = new_text.strip()
    if not existing:
        return incoming
    if not incoming:
        return existing

    existing_words = existing.split()
    incoming_words = incoming.split()
    existing_norm = [_normalize_token_for_match(word) for word in existing_words]
    incoming_norm = [_normalize_token_for_match(word) for word in incoming_words]

    max_overlap_words = min(len(existing_words), len(incoming_words), 12)
    for overlap_size in range(max_overlap_words, 0, -1):
        if existing_norm[-overlap_size:] == incoming_norm[:overlap_size]:
            remainder = " ".join(incoming_words[overlap_size:]).strip()
            if not remainder:
                return existing
            return f"{existing} {remainder}".strip()

    if incoming and incoming[0] in ",.;:!?)]}":
        return f"{existing}{incoming}"
    return f"{existing} {incoming}".strip()


def transcribe_audio_window(
    *,
    model,
    audio: np.ndarray,
    sample_rate: int,
    beam_size: int,
    vad_filter: bool,
    language: str | None = None,
) -> tuple[str, str, float]:
    prepared_audio, _prepared_sample_rate = prepare_audio_samples(audio, sample_rate)
    if prepared_audio.size == 0:
        return "", "unknown", 0.0

    transcribe_kwargs = {
        "beam_size": beam_size,
        "vad_filter": vad_filter,
    }
    if language is not None:
        transcribe_kwargs["language"] = language

    try:
        segments, info = model.transcribe(prepared_audio, **transcribe_kwargs)
    except Exception as exc:
        if vad_filter and "silero_vad_v6.onnx" in str(exc):
            logging.warning("VAD asset missing for chunk, retrying without VAD")
            fallback_kwargs = dict(transcribe_kwargs)
            fallback_kwargs["vad_filter"] = False
            segments, info = model.transcribe(prepared_audio, **fallback_kwargs)
        else:
            raise

    text = "".join(segment.text for segment in segments).strip()
    detected_language = str(getattr(info, "language", "unknown") or "unknown")
    detected_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    return text, detected_language, detected_probability


def transcribe_audio_chunked(
    *,
    model,
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_sec: float,
    overlap_duration_sec: float,
    beam_size: int,
    vad_filter: bool,
    language: str | None = None,
    on_status: Callable[[str], None] | None = None,
    on_chunk_done: Callable[[int, int, float, float, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[str, str, float, int]:
    if audio.size == 0:
        return "", "unknown", 0.0, 0

    chunk_sec = max(0.25, float(chunk_duration_sec))
    chunk_samples = max(1, int(sample_rate * chunk_sec))
    overlap_sec = max(0.0, min(float(overlap_duration_sec), max(0.0, chunk_sec - 0.05)))
    overlap_samples = max(0, int(sample_rate * overlap_sec))
    chunk_ranges = _build_chunk_ranges(audio.shape[0], chunk_samples, overlap_samples)
    total_chunks = len(chunk_ranges)

    merged_text = ""
    lang_score: dict[str, float] = {}

    for idx, (start_idx, end_idx) in enumerate(chunk_ranges, start=1):
        if should_cancel is not None and should_cancel():
            raise TranscriptionCancelledError()

        chunk = audio[start_idx:end_idx]
        if chunk.size == 0:
            continue

        if on_status is not None:
            on_status(f"Transcribing chunk {idx}/{total_chunks}...")

        chunk_text, detected_language, detected_probability = transcribe_audio_window(
            model=model,
            audio=chunk,
            sample_rate=sample_rate,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=language,
        )
        if chunk_text:
            merged_text = merge_chunk_transcript(merged_text, chunk_text)

        if detected_language:
            lang_score[detected_language] = lang_score.get(detected_language, 0.0) + detected_probability

        if on_chunk_done is not None:
            start_sec = start_idx / float(sample_rate)
            end_sec = end_idx / float(sample_rate)
            on_chunk_done(idx, total_chunks, start_sec, end_sec, len(chunk_text))

        if should_cancel is not None and should_cancel():
            raise TranscriptionCancelledError()

    detected_language = "unknown"
    detected_probability = 0.0
    if lang_score:
        detected_language, detected_probability = max(lang_score.items(), key=lambda item: item[1])

    return merged_text.strip(), detected_language, detected_probability, total_chunks


@dataclass
class LiveOverlapResult:
    text: str = ""
    detected_language: str = "unknown"
    detected_score: float = 0.0
    chunk_count: int = 0
    cancelled: bool = False
    error: Exception | None = None


class LiveOverlapSession:
    def __init__(
        self,
        *,
        model,
        sample_rate: int,
        dtype: str,
        channels: int,
        beam_size: int,
        vad_filter: bool,
        language: str | None,
        chunk_duration_sec: float,
        chunk_overlap_sec: float,
        cancel_event: threading.Event,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.sample_rate = int(sample_rate)
        self.dtype = dtype
        self.channels = int(channels)
        self.beam_size = int(beam_size)
        self.vad_filter = bool(vad_filter)
        self.language = language
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.chunk_overlap_sec = float(chunk_overlap_sec)
        self.cancel_event = cancel_event
        self.on_status = on_status

        self.result = LiveOverlapResult()

        self._thread: threading.Thread | None = None
        self._recording_finished = False
        self._audio_lock = threading.Lock()
        self._audio_ready = threading.Event()
        self._audio_bytes = bytearray()
        self._frame_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def append_chunk(self, audio_chunk: np.ndarray) -> None:
        if audio_chunk.size == 0:
            return
        with self._audio_lock:
            self._audio_bytes.extend(audio_chunk.tobytes())
            self._frame_count += int(audio_chunk.shape[0])
        self._audio_ready.set()

    def finish_recording(self) -> None:
        self._recording_finished = True
        self._audio_ready.set()

    def wait(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join()
        self._thread = None

    def _get_frame_count(self) -> int:
        with self._audio_lock:
            return int(self._frame_count)

    def _get_audio_slice(self, start_frame: int, end_frame: int) -> np.ndarray:
        dtype = np.dtype(self.dtype)
        bytes_per_frame = dtype.itemsize * self.channels
        start_byte = max(0, start_frame) * bytes_per_frame
        end_byte = max(start_frame, end_frame) * bytes_per_frame
        with self._audio_lock:
            raw_slice = bytes(self._audio_bytes[start_byte:end_byte])

        if not raw_slice:
            return np.array([], dtype=dtype)

        audio = np.frombuffer(raw_slice, dtype=dtype).copy()
        if self.channels > 1:
            audio = audio.reshape(-1, self.channels)
        return audio

    def _run(self) -> None:
        chunk_sec = max(0.25, self.chunk_duration_sec)
        overlap_sec = max(0.0, min(self.chunk_overlap_sec, max(0.0, chunk_sec - 0.05)))
        chunk_samples = max(1, int(round(self.sample_rate * chunk_sec)))
        overlap_samples = max(0, int(round(self.sample_rate * overlap_sec)))
        step_samples = max(1, chunk_samples - overlap_samples)
        next_chunk_start = 0
        merged_text = ""
        language_scores: dict[str, float] = {}
        processed_chunks = 0

        try:
            while True:
                if self.cancel_event.is_set():
                    raise TranscriptionCancelledError()

                available_frames = self._get_frame_count()
                full_chunk_ready = available_frames >= next_chunk_start + chunk_samples
                final_tail_ready = self._recording_finished and (available_frames > next_chunk_start)

                if not full_chunk_ready and not final_tail_ready:
                    if self._recording_finished:
                        break
                    self._audio_ready.wait(timeout=0.1)
                    self._audio_ready.clear()
                    continue

                chunk_end = min(available_frames, next_chunk_start + chunk_samples)
                raw_chunk = self._get_audio_slice(next_chunk_start, chunk_end)
                if raw_chunk.size == 0:
                    if self._recording_finished:
                        break
                    self._audio_ready.wait(timeout=0.05)
                    self._audio_ready.clear()
                    continue

                processed_chunks += 1
                if self.on_status is not None:
                    self.on_status(f"Live transcribing chunk {processed_chunks}...")

                chunk_text, detected_language, detected_probability = transcribe_audio_window(
                    model=self.model,
                    audio=raw_chunk,
                    sample_rate=self.sample_rate,
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                    language=self.language,
                )
                if chunk_text:
                    merged_text = merge_chunk_transcript(merged_text, chunk_text)
                if detected_language:
                    language_scores[detected_language] = language_scores.get(detected_language, 0.0) + detected_probability

                logging.info(
                    "Live chunk %d transcribed | %.2f-%.2f sec | chars=%d | overlap=%.2f sec",
                    processed_chunks,
                    next_chunk_start / float(self.sample_rate),
                    chunk_end / float(self.sample_rate),
                    len(chunk_text),
                    overlap_sec,
                )

                next_chunk_start += step_samples
                if self._recording_finished and chunk_end >= available_frames:
                    break

            detected_language = "unknown"
            detected_score = 0.0
            if language_scores:
                detected_language, detected_score = max(language_scores.items(), key=lambda item: item[1])

            self.result = LiveOverlapResult(
                text=merged_text.strip(),
                detected_language=detected_language,
                detected_score=detected_score,
                chunk_count=processed_chunks,
            )
        except TranscriptionCancelledError:
            self.result = LiveOverlapResult(cancelled=True, chunk_count=processed_chunks)
            logging.info("Live transcription cancelled")
        except Exception as exc:
            self.result = LiveOverlapResult(error=exc, chunk_count=processed_chunks)
            logging.exception("Live transcription failed")
        finally:
            self._audio_ready.set()