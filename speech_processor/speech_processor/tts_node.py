#!/usr/bin/env python3

# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
Enhanced TTS Node

Improved Text-to-Speech functionality with better architecture,
caching, and multiple provider support.
"""

import base64
import io
import json
import os
import time
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import threading
import itertools
import re

from pydub import AudioSegment
from pydub.playback import play
import rclpy
from rclpy.node import Node
import requests
from std_msgs.msg import String
from go2_interfaces.msg import WebRtcReq

import threading, itertools
_PLAYBACK_LOCK = threading.Lock()
_SESSION_COUNTER = itertools.count(1)


# near imports
try:
    from go2_robot_sdk.domain.constants.webrtc_topics import WEBRTC_TOPICS
    AUDIO_TOPIC_STR = WEBRTC_TOPICS["AUDIO_HUB_REQ"]   # "rt/api/audiohub/request"
except Exception:
    AUDIO_TOPIC_STR = "rt/api/audiohub/request"

_SESSION_COUNTER = itertools.count(1)  # monotonic, never repeats during process lifetime

class AudioFormat(Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"


class TTSProvider(Enum):
    """Supported TTS providers"""
    ELEVENLABS = "elevenlabs"
    GOOGLE = "google"
    AMAZON = "amazon"
    OPENAI = "openai"


@dataclass
class TTSConfig:
    """Configuration for TTS functionality"""
    api_key: str
    provider: TTSProvider = TTSProvider.ELEVENLABS
    voice_name: str = "XrExE9yKIg1WjnnlVkGX"
    local_playback: bool = False
    use_cache: bool = True
    cache_dir: str = "tts_cache"
    chunk_size: int = 16 * 1024
    audio_quality: str = "standard"  # standard, high
    language: str = "en"
    
    # ElevenLabs specific settings
    stability: float = 0.5
    similarity_boost: float = 0.5
    model_id: str = "eleven_turbo_v2_5"


class AudioCache:
    """Thread-safe audio cache management"""
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._lock = threading.Lock()
        
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_path(self, text: str, voice_name: str, provider: str) -> str:
        """Generate cache file path"""
        cache_key = f"{text}_{voice_name}_{provider}"
        text_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.mp3")
    
    def get(self, text: str, voice_name: str, provider: str) -> Optional[bytes]:
        """Get cached audio data"""
        if not self.enabled:
            return None
            
        with self._lock:
            cache_path = self.get_cache_path(text, voice_name, provider)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return f.read()
        return None
    
    def put(self, text: str, voice_name: str, provider: str, audio_data: bytes) -> bool:
        """Cache audio data"""
        if not self.enabled or not audio_data:
            return False
            
        with self._lock:
            try:
                cache_path = self.get_cache_path(text, voice_name, provider)
                with open(cache_path, "wb") as f:
                    f.write(audio_data)
                return True
            except Exception:
                return False
    
    def clear(self) -> bool:
        """Clear all cached files"""
        if not self.enabled:
            return True
            
        with self._lock:
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                return True
            except Exception:
                return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
            
        with self._lock:
            try:
                files = os.listdir(self.cache_dir)
                total_size = sum(
                    os.path.getsize(os.path.join(self.cache_dir, f)) 
                    for f in files if os.path.isfile(os.path.join(self.cache_dir, f))
                )
                return {
                    "enabled": True,
                    "file_count": len(files),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "cache_dir": self.cache_dir
                }
            except Exception:
                return {"enabled": True, "error": "Unable to read cache stats"}


class TTSProvider_ElevenLabs:
    """ElevenLabs TTS provider implementation"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """Generate speech using ElevenLabs API"""
        url = f"{self.base_url}/text-to-speech/{self.config.voice_name}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.config.api_key,
        }
        
        data = {
            "text": text,
            "model_id": self.config.model_id,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost
            },
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException:
            return None
    
    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices"""
        url = f"{self.base_url}/voices"
        headers = {"xi-api-key": self.config.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get("voices", [])
        except requests.exceptions.RequestException:
            return []


class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def convert_to_wav(audio_data: bytes, input_format: AudioFormat = AudioFormat.MP3) -> Optional[bytes]:
        """Convert audio data to WAV format"""
        try:
            if input_format == AudioFormat.MP3:
                audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            elif input_format == AudioFormat.OGG:
                audio = AudioSegment.from_ogg(io.BytesIO(audio_data))
            else:
                return audio_data  # Already WAV
            
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            return wav_io.getvalue()
        except Exception:
            return None
    
    @staticmethod
    def get_duration(audio_data: bytes, format: AudioFormat = AudioFormat.WAV) -> float:
        """Get audio duration in seconds"""
        try:
            if format == AudioFormat.WAV:
                audio = AudioSegment.from_wav(io.BytesIO(audio_data))
            elif format == AudioFormat.MP3:
                audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            else:
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception:
            return 0.0
    
    @staticmethod
    def split_into_chunks(data: bytes, chunk_size: int) -> List[bytes]:
        """Split data into chunks"""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


class EnhancedTTSNode(Node):
    """Enhanced TTS Node with improved architecture"""
    
    def __init__(self):
        super().__init__("tts_node")
        
        # Declare parameters
        self._declare_parameters()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize components
        self.cache = AudioCache(self.config.cache_dir, self.config.use_cache)
        self.audio_processor = AudioProcessor()
        
        # Initialize TTS provider
        self.tts_provider = self._create_tts_provider()
        
        if not self.tts_provider:
            self.get_logger().error("Failed to initialize TTS provider!")
            return
        
        # Setup subscriptions and publishers
        self._setup_communication()
        
        # RTC topic constants (imported from domain)
        self.RTC_TOPIC = AUDIO_TOPIC_STR #{"AUDIO_HUB_REQ": "1003"}  # Fallback if import fails
        
        # Log initialization
        self._log_initialization()
    
    def _declare_parameters(self) -> None:
        """Declare all node parameters"""
        self.declare_parameter("api_key", "")
        self.declare_parameter("provider", "elevenlabs")
        self.declare_parameter("voice_name", "XrExE9yKIg1WjnnlVkGX")
        self.declare_parameter("local_playback", False)
        self.declare_parameter("use_cache", True)
        self.declare_parameter("cache_dir", "tts_cache")
        self.declare_parameter("chunk_size", 16384)
        self.declare_parameter("audio_quality", "standard")
        self.declare_parameter("language", "en")
        self.declare_parameter("stability", 0.5)
        self.declare_parameter("similarity_boost", 0.5)
        self.declare_parameter("model_id", "eleven_turbo_v2_5")
    
    def _load_configuration(self) -> TTSConfig:
        """Load configuration from parameters"""
        provider_str = self.get_parameter("provider").get_parameter_value().string_value
        try:
            provider = TTSProvider(provider_str)
        except ValueError:
            provider = TTSProvider.ELEVENLABS
        
        return TTSConfig(
            api_key=self.get_parameter("api_key").get_parameter_value().string_value,
            provider=provider,
            voice_name=self.get_parameter("voice_name").get_parameter_value().string_value,
            local_playback=self.get_parameter("local_playback").get_parameter_value().bool_value,
            use_cache=self.get_parameter("use_cache").get_parameter_value().bool_value,
            cache_dir=self.get_parameter("cache_dir").get_parameter_value().string_value,
            chunk_size=self.get_parameter("chunk_size").get_parameter_value().integer_value,
            audio_quality=self.get_parameter("audio_quality").get_parameter_value().string_value,
            language=self.get_parameter("language").get_parameter_value().string_value,
            stability=self.get_parameter("stability").get_parameter_value().double_value,
            similarity_boost=self.get_parameter("similarity_boost").get_parameter_value().double_value,
            model_id=self.get_parameter("model_id").get_parameter_value().string_value,
        )
    
    def _create_tts_provider(self):
        """Create TTS provider based on configuration"""
        if self.config.provider == TTSProvider.ELEVENLABS:
            if not self.config.api_key:
                self.get_logger().error("ElevenLabs API key not provided!")
                return None
            return TTSProvider_ElevenLabs(self.config)
        else:
            self.get_logger().error(f"Unsupported TTS provider: {self.config.provider}")
            return None
    
    def _setup_communication(self) -> None:
        """Setup ROS2 communication"""
        self.subscription = self.create_subscription(
            String, "/tts", self.tts_callback, 10
        )
        
        self.audio_pub = self.create_publisher(WebRtcReq, "/webrtc_req", 10)
        
        # Service for cache management
        # self.cache_service = self.create_service(
        #     Empty, "clear_tts_cache", self.clear_cache_callback
        # )
    
    def _chunk_text(self, text: str, max_chars: int = 200) -> list[str]:
        """
        Split text at natural boundaries, keeping each chunk <= max_chars.
        Priority: . ! ? ; : ,  — then fall back to word wrap.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) <= max_chars:
            return [text]

        # First pass: split on sentence enders while keeping them
        parts = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        buf = ""

        def flush():
            nonlocal buf
            if buf.strip():
                chunks.append(buf.strip())
            buf = ""

        for part in parts:
            if not part:
                continue
            if len(buf) + 1 + len(part) <= max_chars:
                buf = (buf + " " + part).strip() if buf else part
            else:
                # try softer break inside the part
                if len(part) > max_chars:
                    softs = re.split(r'(?<=[;:,])\s+', part)
                    for s in softs:
                        if len(buf) + 1 + len(s) <= max_chars:
                            buf = (buf + " " + s).strip() if buf else s
                        else:
                            flush()
                            if len(s) <= max_chars:
                                buf = s
                            else:
                                # last resort: word wrap
                                words = s.split()
                                cur = ""
                                for w in words:
                                    if len(cur) + 1 + len(w) <= max_chars:
                                        cur = (cur + " " + w).strip() if cur else w
                                    else:
                                        if cur:
                                            chunks.append(cur)
                                        cur = w
                                if cur:
                                    buf = cur
                else:
                    flush()
                    buf = part
        flush()
        return chunks

    def _synthesize_chunk(self, text: str) -> bytes:
        """Return MP3 bytes for a single chunk, using cache when possible."""
        audio_data = self.cache.get(text, self.config.voice_name, self.config.provider.value)
        if audio_data:
            return audio_data

        audio_data = self.tts_provider.synthesize(text)
        if not audio_data:
            raise RuntimeError("TTS provider returned no audio")
        # Save to cache
        self.cache.put(text, self.config.voice_name, self.config.provider.value, audio_data)
        return audio_data

    def tts_callback(self, msg: String) -> None:
        try:
            full_text = msg.data.strip()
            if not full_text:
                self.get_logger().warn("Received empty TTS request")
                return

            pieces = self._chunk_text(full_text, max_chars=30)
            n = len(pieces)
            self.get_logger().info(f"🎤 TTS request split into {n} chunk(s)")

            # -------- Phase A: PREFETCH all audio --------
            audios: list[bytes] = []
            for i, piece in enumerate(pieces, 1):
                self.get_logger().info(f"🔊 Generating chunk {i}/{n}…")
                try:
                    audio_mp3 = self._synthesize_chunk(piece)
                except Exception as e:
                    self.get_logger().error(f"❌ Failed to synthesize chunk {i}: {e}")
                    return
                audios.append(audio_mp3)

            # -------- Phase B: PLAYBACK sequentially --------
            for i, (piece, audio_mp3) in enumerate(zip(pieces, audios), 1):
                preview = (piece[:60] + "…") if len(piece) > 60 else piece
                self.get_logger().info(f'📤 Playing chunk {i}/{n}: "{preview}"')
                self._play_on_robot(audio_mp3)   # uses the stable, locked sender
                #time.sleep(0.35)                 # small gap between utterances

            self.get_logger().info("✅ TTS completed successfully (all chunks)")

        except Exception as e:
            self.get_logger().error(f"❌ TTS processing error: {e}")

    
    def _play_locally(self, audio_data: bytes) -> None:
        """Play audio locally"""
        try:
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            play(audio)
            self.get_logger().info("🔊 Local playback completed")
        except Exception as e:
            self.get_logger().error(f"❌ Local playback error: {str(e)}")

    def _make_robot_friendly_wav(self, audio_mp3: bytes, pad_ms: int = 400) -> bytes:
        audio = AudioSegment.from_mp3(io.BytesIO(audio_mp3))
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # mono, 16-bit PCM
        if pad_ms > 0:
            audio = audio + AudioSegment.silent(duration=pad_ms, frame_rate=16000)
        out = io.BytesIO()
        audio.export(out, format="wav", parameters=["-acodec", "pcm_s16le"])
        return out.getvalue()

    def _play_on_robot(self, audio_data: bytes) -> None:
        # Serialize: if something is playing, wait your turn
        with _PLAYBACK_LOCK:
            try:
                wav_data = self._make_robot_friendly_wav(audio_data, pad_ms=400)

                # Base64 + chunking (smaller chunks help jitter)
                b64 = base64.b64encode(wav_data)
                chunk_size = 4096  # bytes in base64 stream
                chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
                total = len(chunks)

                # One session id per utterance
                sid = next(_SESSION_COUNTER)

                # Timing knobs (tuned to avoid overlap/flush issues)
                t_after_start   = 0.25   # wait after START before first chunk
                t_between_chunks= 0.12   # 100–120 ms is usually safe
                t_before_end    = 1.10   # let buffer drain fully

                # Optional: send a pre-reset END if you still see overlap
                # self._send_audio_command(4002, "", session_id=sid)
                # time.sleep(0.30)

                # START
                self._send_audio_command(4001, "", session_id=sid)
                time.sleep(t_after_start)

                # CHUNKS — 1..N
                for i, c in enumerate(chunks, 1):
                    block = {
                        "current_block_index": i,
                        "total_block_number": total,
                        "block_content": c.decode("ascii"),
                    }
                    self._send_audio_command(4003, json.dumps(block), session_id=sid)
                    time.sleep(t_between_chunks)

                # No duplicate last chunk (avoid “echo”)
                time.sleep(t_before_end)

                # END
                self._send_audio_command(4002, "", session_id=sid)

            except Exception as e:
                self.get_logger().error(f"❌ Robot playback error: {e}")



    '''
    def _play_on_robot(self, audio_data: bytes) -> None:
        """Send audio to robot for playback"""
        try:
            # Convert to WAV
            wav_data = self.audio_processor.convert_to_wav(audio_data, AudioFormat.MP3)
            if not wav_data:
                self.get_logger().error("❌ Failed to convert audio to WAV")
                return
            
            # Get audio duration for timing
            duration = self.audio_processor.get_duration(wav_data, AudioFormat.WAV)
            
            # Encode and split into chunks
            b64_encoded = base64.b64encode(wav_data).decode("utf-8")
            chunks = self.audio_processor.split_into_chunks(b64_encoded.encode(), self.config.chunk_size)
            total_chunks = len(chunks)
            
            self.get_logger().info(f"📤 Sending audio to robot: {total_chunks} chunks, {duration:.1f}s duration")
            
            # Send start command
            self._send_audio_command(4001, "")
            time.sleep(0.1)
            
            # Send audio chunks
            for chunk_idx, chunk in enumerate(chunks, 1):
                audio_block = {
                    "current_block_index": chunk_idx,
                    "total_block_number": total_chunks,
                    "block_content": chunk.decode(),
                }
                self._send_audio_command(4003, json.dumps(audio_block))
                
                if chunk_idx % 10 == 0:  # Log progress every 10 chunks
                    self.get_logger().info(f"📤 Sent {chunk_idx}/{total_chunks} chunks")
                
                time.sleep(0.15)  # Prevent flooding
            
            # Wait for playback to complete
            self.get_logger().info(f"⏳ Waiting for playback completion ({duration:.1f}s)...")
            time.sleep(duration + 1.0)
            
            # Send end command
            self._send_audio_command(4002, "")
            
            self.get_logger().info("🎵 Robot playback completed")
            
        except Exception as e:
            self.get_logger().error(f"❌ Robot playback error: {str(e)}")
    '''    
    def _send_audio_command(self, api_id: int, parameter: str, session_id: int = 0) -> None:
        req = WebRtcReq()
        req.id = int(session_id)        # same id per utterance; new id for next one
        req.api_id = int(api_id)
        req.priority = 1                # <- REQUIRED for your driver (variant #2)
        req.parameter = parameter
        req.topic = self.RTC_TOPIC
        self.audio_pub.publish(req)    
    def _log_initialization(self) -> None:
        """Log initialization details"""
        cache_stats = self.cache.get_cache_stats()
        
        self.get_logger().info("🎤 Enhanced TTS Node Initialized")
        self.get_logger().info(f"   Provider: {self.config.provider.value}")
        self.get_logger().info(f"   Voice: {self.config.voice_name}")
        self.get_logger().info(f"   Playback: {'Local' if self.config.local_playback else 'Robot'}")
        self.get_logger().info(f"   Language: {self.config.language}")
        self.get_logger().info(f"   Quality: {self.config.audio_quality}")
        
        if cache_stats["enabled"]:
            self.get_logger().info(f"   Cache: Enabled ({cache_stats.get('file_count', 0)} files, "
                                 f"{cache_stats.get('total_size_mb', 0)}MB)")
        else:
            self.get_logger().info("   Cache: Disabled")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = EnhancedTTSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ TTS Node error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main() 
