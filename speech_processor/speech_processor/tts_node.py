#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import io, os, re, hashlib, threading, itertools
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

import requests
from pydub import AudioSegment
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String, UInt8MultiArray


from transformers import VitsModel, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
import torch


# -------------------------- Config / Enums --------------------------

class AudioFormat(Enum):
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"

class TTSProvider(Enum):
    ELEVENLABS = "elevenlabs"

@dataclass
class TTSConfig:
    api_key: str
    provider: TTSProvider = TTSProvider.ELEVENLABS
    voice_name: str = "XrExE9yKIg1WjnnlVkGX"
    use_cache: bool = True
    cache_dir: str = "tts_cache"
    language: str = "en"
    # ElevenLabs
    stability: float = 0.5
    similarity_boost: float = 0.5
    model_id: str = "eleven_turbo_v2_5"


# -------------------------- Cache --------------------------

class AudioCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, text: str, voice_name: str, provider: str) -> str:
        key = f"{text}_{voice_name}_{provider}"
        return os.path.join(self.cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.mp3")

    def get(self, text: str, voice_name: str, provider: str) -> Optional[bytes]:
        if not self.enabled: return None
        with self._lock:
            p = self._path(text, voice_name, provider)
            if os.path.exists(p):
                with open(p, "rb") as f: return f.read()
        return None

    def put(self, text: str, voice_name: str, provider: str, audio_data: bytes) -> bool:
        if not self.enabled or not audio_data: return False
        with self._lock:
            try:
                with open(self._path(text, voice_name, provider), "wb") as f:
                    f.write(audio_data)
                return True
            except Exception:
                return False

    def stats(self) -> Dict[str, Any]:
        if not self.enabled: return {"enabled": False}
        try:
            files = os.listdir(self.cache_dir)
            total = sum(os.path.getsize(os.path.join(self.cache_dir, f))
                        for f in files if os.path.isfile(os.path.join(self.cache_dir, f)))
            return {"enabled": True, "file_count": len(files), "total_size_mb": round(total/1048576, 2)}
        except Exception:
            return {"enabled": True, "error": "Unable to read cache stats"}


# -------------------------- Provider --------------------------

class TTSProvider_ElevenLabs:
    def __init__(self, config: TTSConfig):
        self.cfg = config
        self.base = "https://api.elevenlabs.io/v1"

    def synthesize(self, text: str) -> Optional[bytes]:
        url = f"{self.base}/text-to-speech/{self.cfg.voice_name}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.cfg.api_key,
        }
        data = {
            "text": text,
            "model_id": self.cfg.model_id,
            "voice_settings": {
                "stability": self.cfg.stability,
                "similarity_boost": self.cfg.similarity_boost
            },
        }
        try:
            r = requests.post(url, json=data, headers=headers, timeout=30)
            r.raise_for_status()
            return r.content
        except requests.exceptions.RequestException:
            return None

class TTSProvider_MMS:
    """
    Minimal provider for Meta's MMS-TTS (VITS-based) via Hugging Face.
    Produces WAV bytes (PCM16). Default English checkpoint: facebook/mms-tts-eng
    """
    def __init__(self, model_id: str = "facebook/mms-tts-eng", device: str | None = None, use_half: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VitsModel.from_pretrained(model_id)
        if use_half and self.device.startswith("cuda"):
            self.model = self.model.half()
        self.model = self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.sr = int(getattr(self.model.config, "sampling_rate", 24000))  # commonly 16000 or 24000
        # Target sample rate for your player (your TTSPlayer node defaults to 16000)
        self.target_sr = 16000

    @torch.inference_mode()
    def synthesize(self, text: str) -> bytes:
        # 1) Text → waveform
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**inputs).waveform  # [1, T] float32 in [-1,1]
        wav = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # 2) (Optional) Resample to 16 kHz mono to match your player
        if self.sr != self.target_sr:

            wav = librosa.resample(wav, orig_sr=self.sr, target_sr=self.target_sr)
            sr = self.target_sr
        else:
            sr = self.sr

        # 3) Encode as WAV (PCM16) bytes
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
# -------------------------- Main Node --------------------------

class EnhancedTTSNode(Node):
    def __init__(self):
        super().__init__("tts_node")

        # Params
        self.declare_parameter("api_key", "")
        self.declare_parameter("provider", "elevenlabs")
        self.declare_parameter("voice_name", "XrExE9yKIg1WjnnlVkGX")
        self.declare_parameter("use_cache", True)
        self.declare_parameter("cache_dir", "tts_cache")
        self.declare_parameter("language", "en")
        self.declare_parameter("stability", 0.5)
        self.declare_parameter("similarity_boost", 0.5)
        self.declare_parameter("model_id", "eleven_turbo_v2_5")
        # ROS topic to publish WAV bytes for the Pi player:
        self.declare_parameter("wav_topic", "/tts_wav")
        # Stitching options
        self.declare_parameter("inter_silence_ms", 120)
        self.declare_parameter("pad_tail_ms", 900)
        # Text chunking (reduce provider calls, but keep natural prosody)
        self.declare_parameter("max_chars", 220)

        # Config
        provider = TTSProvider(self.get_parameter("provider").get_parameter_value().string_value)
        self.cfg = TTSConfig(
            api_key=self.get_parameter("api_key").get_parameter_value().string_value,
            provider=provider,
            voice_name=self.get_parameter("voice_name").get_parameter_value().string_value,
            use_cache=self.get_parameter("use_cache").get_parameter_value().bool_value,
            cache_dir=self.get_parameter("cache_dir").get_parameter_value().string_value,
            language=self.get_parameter("language").get_parameter_value().string_value,
            stability=self.get_parameter("stability").get_parameter_value().double_value,
            similarity_boost=self.get_parameter("similarity_boost").get_parameter_value().double_value,
            model_id=self.get_parameter("model_id").get_parameter_value().string_value,
        )
        self.inter_silence_ms = int(self.get_parameter("inter_silence_ms").value)
        self.pad_tail_ms = int(self.get_parameter("pad_tail_ms").value)
        self.max_chars = int(self.get_parameter("max_chars").value)
        self.wav_topic = self.get_parameter("wav_topic").get_parameter_value().string_value

        # Components
        self.cache = AudioCache(self.cfg.cache_dir, self.cfg.use_cache)
        self.provider = TTSProvider_MMS(model_id="facebook/mms-tts-eng")  #TTSProvider_ElevenLabs(self.cfg)

        # ROS I/O
        self.subscription = self.create_subscription(String, "/tts", self.tts_callback, 10)

        qos = QoSProfile(depth=5)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.history = QoSHistoryPolicy.KEEP_LAST
        self.wav_pub = self.create_publisher(UInt8MultiArray, self.wav_topic, qos)

        # Log
        st = self.cache.stats()
        self.get_logger().info("🎤 TTS ready → publishing WAV to %s" % self.wav_topic)
        self.get_logger().info(f"   Provider: {self.cfg.provider.value}, Voice: {self.cfg.voice_name}, Cache: {st}")

    # ---------- Text handling ----------

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) <= max_chars: return [text]
        parts = re.split(r'(?<=[.!?])\s+', text)
        chunks, buf = [], ""
        def flush():
            nonlocal buf
            if buf.strip(): chunks.append(buf.strip())
            buf = ""
        for p in parts:
            if not p: continue
            if len(buf) + 1 + len(p) <= max_chars:
                buf = (buf + " " + p).strip() if buf else p
            else:
                if len(p) > max_chars:
                    for s in re.split(r'(?<=[;:,])\s+', p):
                        if len(buf) + 1 + len(s) <= max_chars:
                            buf = (buf + " " + s).strip() if buf else s
                        else:
                            flush()
                            if len(s) <= max_chars: buf = s
                            else:
                                cur = ""
                                for w in s.split():
                                    if len(cur) + 1 + len(w) <= max_chars:
                                        cur = (cur + " " + w).strip() if cur else w
                                    else:
                                        if cur: chunks.append(cur)
                                        cur = w
                                if cur: buf = cur
                else:
                    flush(); buf = p
        flush()
        return chunks

    # ---------- Audio helpers ----------

    def _synthesize_chunk_mp3(self, text: str) -> bytes:
        audio = self.cache.get(text, self.cfg.voice_name, self.cfg.provider.value)
        if audio: return audio
        audio = self.provider.synthesize(text)
        if not audio:
            raise RuntimeError("TTS provider returned no audio")
        self.cache.put(text, self.cfg.voice_name, self.cfg.provider.value, audio)
        return audio

    def _concat_audio_chunks_to_wav(self, blobs: List[bytes]) -> bytes:
        if not blobs: return b""
        gap = AudioSegment.silent(duration=self.inter_silence_ms, frame_rate=16000)
        combo = None
        for b in blobs:
            fmt = "wav" if (len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE") else "mp3"
            seg = AudioSegment.from_file(io.BytesIO(b), format=fmt)
            seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # PCM16 mono @16k
            combo = seg if combo is None else (combo + gap + seg)
        if self.pad_tail_ms > 0:
            combo += AudioSegment.silent(duration=self.pad_tail_ms, frame_rate=16000)
        out = io.BytesIO()
        combo.export(out, format="wav", parameters=["-acodec", "pcm_s16le"])
        return out.getvalue()


    def _concat_mp3s_to_wav(self, mp3_blobs: List[bytes]) -> bytes:
        if not mp3_blobs: return b""
        gap = AudioSegment.silent(duration=self.inter_silence_ms, frame_rate=16000)
        combo = None
        for b in mp3_blobs:
            seg = AudioSegment.from_mp3(io.BytesIO(b)).set_channels(1).set_frame_rate(16000).set_sample_width(2)
            combo = seg if combo is None else (combo + gap + seg)
        if self.pad_tail_ms > 0:
            combo += AudioSegment.silent(duration=self.pad_tail_ms, frame_rate=16000)
        out = io.BytesIO()
        combo.export(out, format="wav", parameters=["-acodec", "pcm_s16le"])
        return out.getvalue()

    def _publish_wav(self, wav_bytes: bytes) -> None:
        if not wav_bytes:
            self.get_logger().warn("Empty WAV, nothing to publish.")
            return
        m = UInt8MultiArray()
        m.data = list(wav_bytes)  # UInt8[] payload
        self.wav_pub.publish(m)
        self.get_logger().info(f"📤 Published WAV ({len(wav_bytes)} bytes) to {self.wav_topic}")


    def _synthesize_chunk_bytes(self, text: str) -> bytes:
        # Cache key can stay the same (voice+provider) or expand to include model_id
        audio = self.cache.get(text, self.cfg.voice_name, self.cfg.provider.value)
        if audio:
            return audio
        audio = self.provider.synthesize(text)  # returns bytes (WAV for MMS; MP3 for ElevenLabs)
        if not audio:
            raise RuntimeError("TTS provider returned no audio")
        self.cache.put(text, self.cfg.voice_name, self.cfg.provider.value, audio)
        return audio


    # ---------- Callback ----------

    def tts_callback(self, msg: String) -> None:
        try:
            text = msg.data.strip()
            if not text:
                self.get_logger().warn("Received empty TTS request")
                return

            pieces = self._chunk_text(text, self.max_chars)
            self.get_logger().info(f"🔊 Synthesizing {len(pieces)} piece(s)…")

            clips = []
            for i, piece in enumerate(pieces, 1):
                self.get_logger().info(f"  • chunk {i}/{len(pieces)}")
                audio = self._synthesize_chunk_bytes(piece)   # see below
                clips.append(audio)

            wav = self._concat_audio_chunks_to_wav(clips)
            self._publish_wav(wav)
            self.get_logger().info("✅ TTS done.")

        except Exception as e:
            self.get_logger().error(f"❌ TTS error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedTTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

