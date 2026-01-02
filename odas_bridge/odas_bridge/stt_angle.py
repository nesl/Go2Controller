#!/usr/bin/env python3
import queue
import threading
import time
import json
import struct
from collections import deque, Counter

import numpy as np

import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from faster_whisper import WhisperModel

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import UInt8MultiArray, Float32, String, Bool
from builtin_interfaces.msg import Time as TimeMsg

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Vector3Stamped
from visualization_msgs.msg import Marker
import bisect
import math

import os
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from std_srvs.srv import Trigger




from rcl_interfaces.msg import SetParametersResult

def now_to_msg(clock):
    return clock.now().to_msg()

def quat_from_yaw_pitch(yaw, pitch, roll=0.0):
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw


def _resolve_hf_id(name_or_size: str) -> str:
    s = (name_or_size or "").strip().lower()
    if "/" in s:
        return name_or_size  # already a full HF repo id
    mapping = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3",
        "large-v3": "openai/whisper-large-v3",
    }
    return mapping.get(s, s or "openai/whisper-small")

def _build_hf_asr(model_id: str, language: str, device_str: str, chunk_s: float, batch: int):
    use_cuda = (device_str.strip().lower() == "cuda") and torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=False, use_safetensors=True
    )
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=float(chunk_s),
        batch_size=int(batch),
        return_timestamps=True,
    )
    gen_kwargs = {"language": (language or None), "task": "transcribe"}
    return asr, gen_kwargs

def _build_torch_whisper_gate(model_id: str, device_str: str, language: str, translate: bool):
    use_cuda = (device_str.strip().lower() == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=False, use_safetensors=True
    ).to(device)
    model.eval()

    # Force language & task like the pipeline would do
    forced_ids = processor.get_decoder_prompt_ids(
        language=(language or None),
        task=("translate" if translate else "transcribe")
    )
    return processor, model, forced_ids, device

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


class HFWhisperWorker(threading.Thread):
    def __init__(self, asr_pipeline, gen_kwargs, translate, language, out_queue, logger):
        super().__init__(daemon=True)
        self.asr = asr_pipeline
        self.gen_kwargs = dict(gen_kwargs)
        if translate:
            self.gen_kwargs["task"] = "translate"
        self.language = language if language else None
        self.q = queue.Queue()
        self.out_q = out_queue
        self.log = logger
        self.running = True

    def submit(self, audio_int16, stamp: TimeMsg, azimuth_deg: float):
        self.q.put((audio_int16, stamp, azimuth_deg))

    def run(self):
        while self.running:
            try:
                audio_int16, stamp, az = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                t0 = time.perf_counter()
                audio_f32 = (audio_int16.astype(np.float32) / 32768.0)
                result = self.asr(audio_f32, generate_kwargs=self.gen_kwargs)
                lat_ms = (time.perf_counter() - t0) * 1000.0

                text = (result.get("text") or "").strip()
                self.out_q.put({
                    "kind": "final_asr",
                    "text": text,
                    "language": self.gen_kwargs.get("language"),
                    "duration": None,
                    "azimuth_deg": az,
                    "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
                    "lat_ms": lat_ms
                })
            except Exception as e:
                self.log.warn(f"Transcribe error: {e}")


    def stop(self): self.running = False

class TorchWhisperGateWorker(threading.Thread):
    """
    Gate worker using Transformers/PyTorch Whisper.
    Produces: text, avg_logprob (per-token), max_no_speech (proxy), segments.
    """
    def __init__(self, processor, model, forced_ids, device, language, translate, out_q, logger, sample_rate: int):
        super().__init__(daemon=True)
        self.processor = processor
        self.model = model
        self.forced_ids = forced_ids
        self.device = device
        self.language = language
        self.translate = translate
        self.q = queue.Queue()
        self.out_q = out_q
        self.log = logger
        self.running = True
        self.sample_rate = sample_rate

    def submit(self, item: dict):
        self.q.put(item)

    def stop(self):
        self.running = False

    def _avg_token_logprob(self, scores_list, token_ids):
        """
        scores_list: list[T (1, vocab)] for *generated* steps (no prompt/forced ids)
        token_ids:   1D tensor of full sequence ids (includes prompt/forced ids)
        We align to the last len(scores_list) tokens, which correspond to generated steps.
        """
        if not scores_list or token_ids is None or token_ids.numel() == 0:
            return -99.0

        gen_len = len(scores_list)
        toks = token_ids[-gen_len:]  # align tail with scores
        s = 0.0
        n = 0
        for step_scores, tok in zip(scores_list, toks):
            # Cast to float32 to avoid FP16 underflow -> -inf
            step_logp = torch.log_softmax(step_scores.float()[0], dim=-1)
            # Guard: token id might be out of range if something went wrong
            idx = int(tok.item())
            if 0 <= idx < step_logp.numel():
                val = float(step_logp[idx].item())
                if math.isfinite(val):
                    s += val
                    n += 1
        return (s / n) if n else -99.0

    def _voiced_ratio_vad(self, audio_f32: np.ndarray, sr: int) -> float:
        """
        WebRTC VAD with:
          - DC removal
          - light AGC toward target RMS (-20 dBFS) with a clamp
          - optional pre-emphasis to lift speech band
          - EMA noise floor & SNR gate assistance
        Returns fraction of 20ms frames flagged as speech in the window.
        """
        try:
            import webrtcvad
            # --- lazy init state ---
            if not hasattr(self, "_vad"):
                self._vad = webrtcvad.Vad(2)  # 0..3, 2 is a good compromise
            if not hasattr(self, "_noise_rms_ema"):
                self._noise_rms_ema = 1e-3  # start small
            if not hasattr(self, "_speech_bias"):
                self._speech_bias = 0.0     # leaky bias to prevent sticky silence

            x = np.asarray(audio_f32, dtype=np.float32)
            if x.size == 0:
                return 0.0

            # --- DC removal ---
            x = x - float(np.mean(x))

            # --- light AGC toward target rms ---
            rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
            target_rms = 0.1  # ≈ -20 dBFS
            if rms > 0:
                gain = np.clip(target_rms / rms, 0.5, 6.0)  # clamp to avoid pumping
                x = x * gain

            # --- optional pre-emphasis (helps VAD at low levels) ---
            # x[t] ← x[t] - 0.97*x[t-1]
            if x.size > 1:
                x[1:] = x[1:] - 0.97 * x[:-1]

            # --- convert to S16 for WebRTC ---
            i16 = np.clip(x, -1.0, 1.0)
            i16 = (i16 * 32768.0).astype(np.int16)

            # --- 20 ms framing (required by WebRTC VAD) ---
            frame_len = sr // 50  # 20 ms
            n = (len(i16) // frame_len) * frame_len
            if n <= 0:
                return 0.0
            buf = i16[:n].tobytes()
            total = n // frame_len
            step = frame_len * 2  # bytes

            # --- update noise floor (EMA) using low-energy frames ---
            # estimate instantaneous rms over the window (post-AGC)
            inst_rms = float(np.sqrt(np.mean((i16.astype(np.float32) / 32768.0) ** 2)) + 1e-12)
            alpha_noise = 0.05  # slow EMA
            # if current energy is low, treat as noise observation
            if inst_rms < 0.06:  # heuristic
                self._noise_rms_ema = (1 - alpha_noise) * self._noise_rms_ema + alpha_noise * inst_rms

            # compute simple SNR proxy
            snr = 20.0 * math.log10(max(inst_rms, 1e-6) / max(self._noise_rms_ema, 1e-6))
            snr_bonus = 0.0
            if snr > 6.0:   # above ~6 dB, give VAD a nudge
                snr_bonus = min((snr - 6.0) / 12.0, 0.3)  # max +0.3 to voiced ratio

            # --- count voiced frames ---
            vad = self._vad
            voiced = 0
            for i in range(0, len(buf), step):
                frame = buf[i:i + step]
                if len(frame) < step:
                    break
                try:
                    if vad.is_speech(frame, sr):
                        voiced += 1
                except Exception:
                    pass

            vr = voiced / max(total, 1)

            # --- apply bias/hysteresis (prevents sticky high no_speech) ---
            # leaky integrate toward recent decision; encourages continuity
            # increase bias slightly when we see voice; decay otherwise
            if vr > 0.5:
                self._speech_bias = min(self._speech_bias + 0.05, 0.4)  # cap
            else:
                self._speech_bias = max(self._speech_bias - 0.02, 0.0)

            vr = np.clip(vr + snr_bonus + self._speech_bias, 0.0, 1.0)
            return float(vr)

        except Exception:
            return 0.0

    def _compression_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        raw = text.encode("utf-8", errors="ignore")
        comp = zlib.compress(raw)
        return (len(raw) / max(len(comp), 1))

    def run(self):
        while self.running:
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                audio = item["audio_f32"]  # numpy float32 mono in [-1,1]
                # ------- no-speech proxy via VAD over this window -------
                try:
                    # parent node placed this method if you added it as shown above
                    parent_vad_ratio = getattr(self, "parent_vad_ratio", None)
                except Exception:
                    parent_vad_ratio = None
                # If the worker is attached from the node, you can monkey-patch like:
                # gate_worker.parent_vad_ratio = node._vad_voiced_ratio

                # Preprocess for Whisper
                inputs = self.processor(
                    audio, sampling_rate=self.sample_rate, return_tensors="pt"
                ).to(self.device)
                inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
                
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        forced_decoder_ids=self.forced_ids,
                        return_dict_in_generate=True,
                        output_scores=True,
                        do_sample=False,
                        num_beams=1,           # keep fast/greedy for the gate
                        temperature=0.0,
                    )
                lat_ms = (time.perf_counter() - t0) * 1000.0
                
                # Decode text
                seq = out.sequences[0]                   # (seq_len,)
                text = self.processor.batch_decode([seq], skip_special_tokens=True)[0].strip()

                # Avg token logprob
                avg_lp = self._avg_token_logprob(out.scores, seq)

                # Simple segment count: 1 if non-empty, else 0
                seg_cnt = 1 if text else 0

                # --- no_speech proxy from VAD (1 - voiced_ratio) with energy guard ---
                voiced_ratio = self._voiced_ratio_vad(audio, self.sample_rate)
                max_no_speech = float(1.0 - voiced_ratio)


                # You can also gate with compression ratio if you like:
                # cr = self._compression_ratio(text)

                self.out_q.put({
                    "kind": "gate_window",
                    "t_start_ns": int(item["t_start_ns"]),
                    "t_end_ns":   int(item["t_end_ns"]),
                    "text": text,
                    "avg_logprob": float(avg_lp),
                    "max_no_speech": max_no_speech,
                    "segments": seg_cnt,
                    "lat_ms": float(lat_ms),
                })

            except Exception as e:
                self.log.warn(f"TorchWhisperGate window error: {e}")



class STTFasterWhisperNode(Node):
    """
    Subscribes:
      - /mic/audio (UInt8MultiArray): raw PCM16LE bytes for TC channels
      - /audio/doa_raw_deg (Float32): latest azimuth (deg)

    Publishes:
      - /audio/stt_text (String): plain transcript text (per utterance)
      - /audio/stt_doa_json (String): JSON with {text, azimuth_deg, language, duration, stamp}
    """
    def __init__(self):
        super().__init__("stt_fwhisper_node")

        # --- Parameters (align with your DoA node) ---
        self.declare_parameter("fs_hz", 16000)
        self.declare_parameter("total_channels", 6)
        self.declare_parameter("mic_lanes", [1, 2, 3, 4])   # lanes present in the byte stream
        self.declare_parameter("ref_ch", 0)                 # index into mic_lanes to pick one channel

        # STT configuration

        self.declare_parameter("translate", False)          # True: translate to English
        self.declare_parameter("language", "en")              # force language or leave empty to auto

        # NEW: role-specific model configs
        # Final transcription (heavier) defaults
        self.declare_parameter("stt_model_size", "medium")
        self.declare_parameter("stt_device", "cuda")
        self.declare_parameter("stt_compute_type", "float16")

        # Gating / endpointing (lighter) defaults
        self.declare_parameter("wg_model_size", "medium")
        self.declare_parameter("wg_device", "cuda")
        self.declare_parameter("wg_compute_type", "int8")

        # --- Whisper-driven endpointing params ---
        self.declare_parameter("wg_window_ms", 3000)        # analyze last N ms
        self.declare_parameter("wg_hop_ms", 1000)           # step size between windows
        self.declare_parameter("wg_end_silence_ms", 800)    # finalize after this much trailing silence
        self.declare_parameter("wg_max_utter_ms", 20000)    # hard cap per utterance
        self.declare_parameter("wg_min_chars", 4)           # reject tiny outputs
        self.declare_parameter("wg_min_avg_logprob", -1.0)  # Whisper conf gate (higher = stricter)
        self.declare_parameter("wg_max_no_speech", 0.6)     # Whisper no-speech gate (lower = stricter)

        # --- RViz DoA Marker params ---
        self.declare_parameter("marker_enabled", True)
        self.declare_parameter("marker_topic", "/audio/doa_marker")
        self.declare_parameter("marker_frame_id", "base_link")
        self.declare_parameter("arrow_length", 0.6)
        self.declare_parameter("shaft_diameter", 0.03)
        self.declare_parameter("head_diameter", 0.08)
        self.declare_parameter("color_rgba", [0.1, 0.8, 0.1, 0.9])
        self.declare_parameter("marker_ttl_sec", 1.5)
        self.declare_parameter("prefer_vector_for_marker", True)
        
        
        # --- Runtime toggles ---
        self.declare_parameter("enable_stt", True)   # final ASR (HF pipeline)
        self.declare_parameter("enable_gate", True)  # gate/endpointing (Torch Whisper)

        self.enable_stt  = bool(self.get_parameter("enable_stt").value)
        self.enable_gate = bool(self.get_parameter("enable_gate").value)

        # --- Services to toggle at runtime ---
        from std_srvs.srv import SetBool
        self.create_service(SetBool, "/toggle_stt",  self._srv_toggle_stt)
        self.create_service(SetBool, "/toggle_gate", self._srv_toggle_gate)

        # --- Metrics publisher ---
        
        self.perf_pub = self.create_publisher(String, "/stt_perf", 10)

        # --- Metrics state ---
        self._ema_alpha = 0.2
        self.lat_gate_ms_ema = None
        self.lat_asr_ms_ema  = None
        self.lat_e2e_ms_ema  = None
        self.windows_processed = 0
        self.utterances_finalized = 0
        self._last_perf_publish = time.time()
        self._perf_publish_period = 2.0  # seconds

       
        self.wg_window_ms = int(self.get_parameter("wg_window_ms").value)
        self.wg_hop_ms = int(self.get_parameter("wg_hop_ms").value)
        self.wg_end_silence_ms = int(self.get_parameter("wg_end_silence_ms").value)
        self.wg_max_utter_ms = int(self.get_parameter("wg_max_utter_ms").value)
        self.wg_min_chars = int(self.get_parameter("wg_min_chars").value)
        self.wg_min_avg_logprob = float(self.get_parameter("wg_min_avg_logprob").value)
        self.wg_max_no_speech = float(self.get_parameter("wg_max_no_speech").value)

        # --- Speaker ID / enrollment ---
        self.declare_parameter("spk_profiles_dir", "/tmp/spk_profiles")
        self.declare_parameter("spk_threshold", 0.85)   # tune later
        self.declare_parameter("spk_save_wav", False)

        self.profiles_dir = str(self.get_parameter("spk_profiles_dir").value)
        os.makedirs(self.profiles_dir, exist_ok=True)

        self.spk_threshold = float(self.get_parameter("spk_threshold").value)
        self.spk_save_wav = bool(self.get_parameter("spk_save_wav").value)

        # Embedding model (Resemblyzer)
        self.spk_encoder = VoiceEncoder()

        # State machine: idle | enroll(person) | verify(person)
        self._spk_mode = "verify"
        self._spk_target = "human_a"

        self.pub_spk = self.create_publisher(String, "/audio/speaker_match", 10)

        self.create_service(Trigger, "/speaker/enroll", self._srv_enroll)
        self.create_service(Trigger, "/speaker/verify", self._srv_verify)
        self.create_service(Trigger, "/speaker/cancel", self._srv_cancel)
        self.declare_parameter("speaker_id", "human_a")

        # ---- Annotate STT output with speaker verification ----
        self.declare_parameter("stt_text_annotate_verify", True)
        self.declare_parameter("stt_text_verify_format", "[spk:{id} score={score:.2f} {'OK' if match else 'NO'}] ")
        self.declare_parameter("stt_text_verify_ttl_ms", 10000)

        self.stt_text_annotate_verify = bool(self.get_parameter("stt_text_annotate_verify").value)
        self.stt_text_verify_format = str(self.get_parameter("stt_text_verify_format").value)
        self.stt_text_verify_ttl_ms = int(self.get_parameter("stt_text_verify_ttl_ms").value)

        self._last_spk = None  # {"id":..., "score":..., "match":..., "ts_ns":..., "threshold":...}


        # Load params
        self.fs = int(self.get_parameter("fs_hz").value)
        self.TC = int(self.get_parameter("total_channels").value)
        self.mic_lanes = [int(x) for x in self.get_parameter("mic_lanes").value]
        self.ref_ch = int(self.get_parameter("ref_ch").value)
        assert 0 <= self.ref_ch < len(self.mic_lanes), "ref_ch out of range of mic_lanes"
        self.pick_lane = self.mic_lanes[self.ref_ch]

        self.translate = bool(self.get_parameter("translate").value)
        self.language = str(self.get_parameter("language").value).strip() or None

        self.stt_model_size = (str(self.get_parameter("stt_model_size").value) or self.model_size)
        self.stt_device = (str(self.get_parameter("stt_device").value) or self.device)
        self.stt_compute_type = (str(self.get_parameter("stt_compute_type").value) or self.compute_type)

        self.wg_model_size = (str(self.get_parameter("wg_model_size").value) or self.model_size)
        self.wg_device = (str(self.get_parameter("wg_device").value) or self.device)
        self.wg_compute_type = (str(self.get_parameter("wg_compute_type").value) or self.compute_type)


        self.stt_hf_id = _resolve_hf_id(str(self.get_parameter("stt_model_size").value))
        self.stt_device = str(self.get_parameter("stt_device").value)
        self.declare_parameter("stt_chunk_s", 30.0)
        self.declare_parameter("stt_batch", 8)
        self.stt_chunk_s = float(self.get_parameter("stt_chunk_s").value)
        self.stt_batch = int(self.get_parameter("stt_batch").value)

        self.wg_hf_id = _resolve_hf_id(str(self.get_parameter("wg_model_size").value))
        self.wg_device = str(self.get_parameter("wg_device").value)
        self.declare_parameter("wg_chunk_s", 15.0)
        self.declare_parameter("wg_batch", 4)
        self.wg_chunk_s = float(self.get_parameter("wg_chunk_s").value)
        self.wg_batch = int(self.get_parameter("wg_batch").value)

        self.get_logger().info(
            f"Python={sys.executable} CUDA={torch.cuda.is_available()} "
            f"torch.cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}"
        )

        # Build pipelines

        self.worker_asr, self.worker_gen = _build_hf_asr(
            self.stt_hf_id, self.language, self.stt_device, self.stt_chunk_s, self.stt_batch
        )

        self.gate_processor, self.gate_model, self.gate_forced_ids, self.gate_device = _build_torch_whisper_gate(
            self.wg_hf_id, self.wg_device, self.language, self.translate
        )
        
        '''
        self.gate_model = WhisperModel(
            self.wg_model_size, device=self.wg_device, compute_type=self.wg_compute_type
        )
        '''
        self.declare_parameter("mix_strategy", "mean")  # single|mean|energy_weighted|maxrms
        self.mix_strategy = str(self.get_parameter("mix_strategy").value).strip().lower()


        # Publishers
        self.pub_text = self.create_publisher(String, "/audio/stt_text", 10)
        self.pub_json = self.create_publisher(String, "/audio/stt_doa_json", 10)
        self.pub_partial = self.create_publisher(String, "/audio/stt_partial_json", 10)

        audio_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.sub_bytes = self.create_subscription(
            UInt8MultiArray, "/mic/audio", self.on_audio_bytes, audio_qos
        )
        
        self.sub_doa_vec = self.create_subscription(
            Vector3Stamped, "/audio/doa_raw", self.on_doa_vec, 10
        )
        self.sub_az = self.create_subscription(
            Float32, "/audio/doa_raw_deg", self.on_azimuth, 50
        )

        # Keep ~5 seconds of DoA history (sorted by time)
        self.doa_hist = deque()  # list of (t_sec_float, az_deg)
        self.doa_keep_sec = 6.0
        self.doa_hist_speech = []

        # Track audio time based on sample counts; initialize to node clock "now"
        self.audio_time = self.get_clock().now()   # rclpy Time
        self.samples_accumulated = 0

        # Keep most-recent DoA and stamp
        self.latest_azimuth = 0.0
        self.latest_stamp = now_to_msg(self.get_clock())

        
        # Start workers
        self.out_q = queue.Queue()
        self.worker = HFWhisperWorker(
            self.worker_asr, self.worker_gen, self.translate, self.language, self.out_q, self.get_logger()
        )
        self.worker.start()       
        
        # Timer to drain completed transcripts
        self.create_timer(0.02, self.drain_outputs)

        # Byte reservoir for partial frames
        self._partial_bytes = bytearray()

        # ---- Angle frequency tracking ----
        self.angle_hist = deque(maxlen=1000)  # keep last 1000 readings (~seconds)
        self.create_timer(1.0, self.show_top_angles)  # update every 1s


        # Raw mono ring buffer (int16)
        self._mono_ring = deque(maxlen=self.fs * 60)  # keep ~60s just in case

        # Sliding window scheduler
        self._next_window_at_ns = self.get_clock().now().nanoseconds  # when to run next window
        self._wg_running = True

        # Utterance assembly
        self._utt_active = False
        self._utt_start_time_ns = None
        self._utt_samples = []  # list of np.int16 chunks
        self._last_speech_time_ns = None

        # Background queue/results for window analyses
        self._win_q = queue.Queue()
        self._win_out = queue.Queue()

        # Timer to launch windows and drain results
        self.create_timer(0.05, self._wg_tick)  # 20 Hz

        self.wg = TorchWhisperGateWorker(
            self.gate_processor, self.gate_model, self.gate_forced_ids, self.gate_device,
            self.language, self.translate, self._win_out, self.get_logger(), sample_rate=self.fs
        )
        self.wg.parent_vad_ratio = self._vad_voiced_ratio
        
        self.wg.start()
        
        
        self.add_on_set_parameters_callback(self._on_param_update)
        
        # Marker params
        self.marker_enabled = bool(self.get_parameter("marker_enabled").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.marker_frame = str(self.get_parameter("marker_frame_id").value)
        self.marker_len = float(self.get_parameter("arrow_length").value)
        self.marker_shaft_d = float(self.get_parameter("shaft_diameter").value)
        self.marker_head_d = float(self.get_parameter("head_diameter").value)
        crgba = [float(x) for x in self.get_parameter("color_rgba").value]
        self.marker_r, self.marker_g, self.marker_b, self.marker_a = crgba
        self.marker_ttl = float(self.get_parameter("marker_ttl_sec").value)
        self.prefer_vector_for_marker = bool(self.get_parameter("prefer_vector_for_marker").value)
        
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)
        
        # --- State for RViz marker (vector + timestamp) ---
        self._dir_vec = None    # (vx, vy, vz) normalized
        self._dir_vec_ts = 0.0   # wall time
        self._voice_active = False
        self._last_marker_publish = 0.0


        # ---- TTS busy gating ----
        self.declare_parameter("busy_topic", "/tts_busy")
        self._tts_busy = False
        busy_topic = self.get_parameter("busy_topic").get_parameter_value().string_value
        self.create_subscription(Bool, busy_topic, self._busy_cb, 10)


        self.get_logger().info(
            f"STT node: lane={self.pick_lane} fs={self.fs} TC={self.TC} "
          
        )

    def _srv_enroll(self, req, resp):
        # person id comes from a parameter for simplicity
        # (Trigger has no request fields)
        pid = str(self.get_parameter("speaker_id").value) if self.has_parameter("speaker_id") else "person"
        self._spk_mode = "enroll"
        self._spk_target = pid
        resp.success = True
        resp.message = f"Enrollment armed for '{pid}'. Say a sentence now."
        self.get_logger().info(resp.message)
        return resp

    def _srv_verify(self, req, resp):
        # In the new behavior, "verify" = "identify among all enrolled speakers".
        self._spk_mode = "verify"
        # _spk_target is no longer used for verify; keep it only for enroll naming.
        resp.success = True
        resp.message = "Verification armed: will identify the most similar enrolled speaker."
        self.get_logger().info(resp.message)
        return resp


    def _srv_cancel(self, req, resp):
        self._spk_mode = "idle"
        self._spk_target = None
        resp.success = True
        resp.message = "Speaker operation cancelled."
        self.get_logger().info(resp.message)
        return resp


    def _rebuild_asr(self, model_size=None, device=None, chunk_s=None, batch=None, language=None, translate=None):
        """Recreate the HF pipeline worker safely."""
        try:
            if model_size is not None:
                self.stt_hf_id = _resolve_hf_id(str(model_size))
            if device is not None:
                self.stt_device = str(device)
            if chunk_s is not None:
                self.stt_chunk_s = float(chunk_s)
            if batch is not None:
                self.stt_batch = int(batch)
            if language is not None:
                self.language = (str(language).strip() or None)
            if translate is not None:
                self.translate = bool(translate)

            # stop old worker
            if hasattr(self, "worker"):
                try: self.worker.stop()
                except: pass

            # rebuild pipeline + worker
            self.worker_asr, self.worker_gen = _build_hf_asr(
                self.stt_hf_id, self.language, self.stt_device, self.stt_chunk_s, self.stt_batch
            )
            self.worker = HFWhisperWorker(
                self.worker_asr, self.worker_gen, self.translate, self.language, self.out_q, self.get_logger()
            )
            self.worker.start()
            self.get_logger().info(f"[params] ASR rebuilt: model={self.stt_hf_id} device={self.stt_device} chunk={self.stt_chunk_s}s batch={self.stt_batch}")
        except Exception as e:
            self.get_logger().error(f"[params] Rebuild ASR failed: {e}")
            raise

    def _rebuild_gate(self, model_size=None, device=None, window_ms=None, hop_ms=None,
                      min_chars=None, min_avg_logprob=None, max_no_speech=None, language=None, translate=None):
        """Recreate the Torch gate worker safely."""
        try:
            if model_size is not None:
                self.wg_hf_id = _resolve_hf_id(str(model_size))
            if device is not None:
                self.wg_device = str(device)
            if window_ms is not None:
                self.wg_window_ms = int(window_ms)
            if hop_ms is not None:
                self.wg_hop_ms = int(hop_ms)
            if min_chars is not None:
                self.wg_min_chars = int(min_chars)
            if min_avg_logprob is not None:
                self.wg_min_avg_logprob = float(min_avg_logprob)
            if max_no_speech is not None:
                self.wg_max_no_speech = float(max_no_speech)
            if language is not None:
                self.language = (str(language).strip() or None)
            if translate is not None:
                self.translate = bool(translate)

            # stop old gate
            if hasattr(self, "wg"):
                try: self.wg.stop()
                except: pass

            # rebuild processor/model/forced_ids + worker
            self.gate_processor, self.gate_model, self.gate_forced_ids, self.gate_device = _build_torch_whisper_gate(
                self.wg_hf_id, self.wg_device, self.language, self.translate
            )
            self.wg = TorchWhisperGateWorker(
                self.gate_processor, self.gate_model, self.gate_forced_ids, self.gate_device,
                self.language, self.translate, self._win_out, self.get_logger(), sample_rate=self.fs
            )
            self.wg.parent_vad_ratio = self._vad_voiced_ratio
            self.wg.start()
            self.get_logger().info(f"[params] Gate rebuilt: model={self.wg_hf_id} device={self.wg_device} win={self.wg_window_ms}ms hop={self.wg_hop_ms}ms")
        except Exception as e:
            self.get_logger().error(f"[params] Rebuild Gate failed: {e}")
            raise

    def _on_param_update(self, params):
        """Hot-apply parameter changes: enable flags, model sizes, devices, language,
           and gate/ASR tuning. Rebuilds workers only when needed.
        """
        resp = SetParametersResult(successful=True)
        # Track whether to rebuild each worker and the desired deltas
        asr_kwargs = {}
        gate_kwargs = {}
        do_rebuild_asr = False
        do_rebuild_gate = False

        try:
            for p in params:
                name, val = p.name, p.value

                # Toggles (services still supported)
                if name == "enable_stt":
                    self.enable_stt = bool(val)
                elif name == "enable_gate":
                    self.enable_gate = bool(val)

                # Language/task
                elif name == "language":
                    asr_kwargs["language"] = val
                    gate_kwargs["language"] = val
                    do_rebuild_asr = True
                    do_rebuild_gate = True
                elif name == "translate":
                    asr_kwargs["translate"] = bool(val)
                    gate_kwargs["translate"] = bool(val)
                    do_rebuild_asr = True
                    do_rebuild_gate = True

                # ASR model/device/tuning
                elif name == "stt_model_size":
                    asr_kwargs["model_size"] = val; do_rebuild_asr = True
                elif name == "stt_device":
                    asr_kwargs["device"] = val; do_rebuild_asr = True
                elif name == "stt_chunk_s":
                    asr_kwargs["chunk_s"] = val; do_rebuild_asr = True
                elif name == "stt_batch":
                    asr_kwargs["batch"] = val; do_rebuild_asr = True

                # Gate model/device/tuning
                elif name == "wg_model_size":
                    gate_kwargs["model_size"] = val; do_rebuild_gate = True
                elif name == "wg_device":
                    gate_kwargs["device"] = val; do_rebuild_gate = True
                elif name == "wg_window_ms":
                    gate_kwargs["window_ms"] = val; do_rebuild_gate = True
                elif name == "wg_hop_ms":
                    gate_kwargs["hop_ms"] = val; do_rebuild_gate = True
                elif name == "wg_min_chars":
                    gate_kwargs["min_chars"] = val; do_rebuild_gate = True
                elif name == "wg_min_avg_logprob":
                    gate_kwargs["min_avg_logprob"] = val; do_rebuild_gate = True
                elif name == "wg_max_no_speech":
                    gate_kwargs["max_no_speech"] = val; do_rebuild_gate = True

                # Mic selection / mixing (no rebuild needed)
                elif name == "ref_ch":
                    self.ref_ch = int(val)
                    self.pick_lane = self.mic_lanes[self.ref_ch]
                elif name == "mix_strategy":
                    self.mix_strategy = str(val).strip().lower()

            # Apply rebuilds (stop old threads first inside helpers)
            if do_rebuild_asr:
                self._rebuild_asr(**asr_kwargs)
            if do_rebuild_gate:
                self._rebuild_gate(**gate_kwargs)

        except Exception as e:
            resp.successful = False
            resp.reason = f"param update failed: {e}"
            self.get_logger().error(resp.reason)
        return resp


    def _ema(self, ema_val, x, alpha=0.2):
        if x is None:
            return ema_val
        if ema_val is None:
            return x
        return (1.0 - alpha) * ema_val + alpha * x

    def _srv_toggle_stt(self, req, resp):
        self.enable_stt = bool(req.data)
        resp.success = True
        resp.message = f"STT final ASR {'ENABLED' if self.enable_stt else 'DISABLED'}"
        self.get_logger().info(resp.message)
        return resp

    def _srv_toggle_gate(self, req, resp):
        self.enable_gate = bool(req.data)
        resp.success = True
        resp.message = f"Gate/endpointing {'ENABLED' if self.enable_gate else 'DISABLED'}"
        self.get_logger().info(resp.message)
        return resp

    def _maybe_publish_perf(self, extra=None):
        now = time.time()
        if (now - self._last_perf_publish) < self._perf_publish_period:
            return
        self._last_perf_publish = now

        extra = extra or {}

        # Build latency_ms dict in the order the broker likes.
        # First key should be the "main" metric for audio_asr.
        latency_ms = {}
        if self.lat_asr_ms_ema is not None:
            # main metric -> used by broker for EMA
            latency_ms["utter_infer_mean"] = float(self.lat_asr_ms_ema)
        if self.lat_gate_ms_ema is not None:
            latency_ms["window_infer_mean"] = float(self.lat_gate_ms_ema)
        if self.lat_e2e_ms_ema is not None:
            latency_ms["e2e_mean"] = float(self.lat_e2e_ms_ema)

        payload = {
            # task / model so broker can match task_registry: audio_asr + tiny/small/medium/large-v3
            "task": "audio_asr",
            "model": str(self.get_parameter("stt_model_size").value),

            # the thing broker actually reads
            "latency_ms": latency_ms,

            # extra telemetry (nice to keep, broker mostly ignores)
            "stamp_wall": now,
            "windows_processed":    self.windows_processed,
            "utterances_finalized": self.utterances_finalized,
            "last_gate_lat_ms": extra.get("last_gate_lat_ms"),
            "last_asr_lat_ms":  extra.get("last_asr_lat_ms"),
            "last_e2e_ms":      extra.get("last_e2e_ms"),
            "enable_stt":  self.enable_stt,
            "enable_gate": self.enable_gate,
        }

        self.perf_pub.publish(String(data=json.dumps(payload)))


    def _vad_voiced_ratio(self, audio_f32: np.ndarray, sr: int) -> float:
        """Return fraction of frames flagged as speech by WebRTC VAD."""
        try:
            vad = getattr(self, "vad", None)
            if vad is None:
                self.vad = webrtcvad.Vad(2)  # 0..3, 3=aggressive
                vad = self.vad
            # convert to 16-bit PCM
            x = np.clip(audio_f32, -1.0, 1.0)
            i16 = (x * 32768.0).astype(np.int16)
            frame_len = sr // 50  # 20ms
            n = (len(i16) // frame_len) * frame_len
            if n <= 0:
                return 0.0
            buf = i16[:n].tobytes()
            total = n // frame_len
            voiced = 0
            step = frame_len * 2
            for i in range(0, len(buf), step):
                frame = buf[i:i+step]
                if len(frame) < step:
                    break
                try:
                    if vad.is_speech(frame, sr):
                        voiced += 1
                except Exception:
                    pass
            return (voiced / max(total, 1))
        except Exception:
            return 0.0


    def _select_mono(self, frames: np.ndarray) -> np.ndarray:
        """
        frames shape: [N, TC] int16. Use self.mic_lanes and self.pick_lane.
        Returns int16 mono array according to self.mix_strategy.
        """
        if self.mix_strategy == "single":
            return frames[:, self.pick_lane].astype(np.int16)

        sel = frames[:, self.mic_lanes]
        if self.mix_strategy == "mean":
            mix = sel.astype(np.int32).mean(axis=1)
            return np.clip(mix, -32768, 32767).astype(np.int16)

        if self.mix_strategy == "energy_weighted":
            sel_f = sel.astype(np.float32)
            rms = np.sqrt((sel_f ** 2).mean(axis=0)) + 1e-8
            w = rms / rms.sum()
            mix = (sel_f * w).sum(axis=1)
            return np.clip(mix, -32768, 32767).astype(np.int16)

        if self.mix_strategy == "maxrms":
            sel_f = sel.astype(np.float32)
            rms = np.sqrt((sel_f ** 2).mean(axis=0))
            best = int(np.argmax(rms))
            return sel[:, best].astype(np.int16)

        # Fallback to single
        return frames[:, self.pick_lane].astype(np.int16)



    def _busy_cb(self, msg: Bool):
        self.get_logger().info(
            f"STT node: busy={msg.data}"
          
        )
        was = self._tts_busy
        self._tts_busy = bool(msg.data)
        if self._tts_busy and not was:
            # Robot just started speaking → wipe in-progress state so nothing leaks through
            self._utt_active = False
            self._utt_start_time_ns = None
            self._utt_samples = []
            self._last_speech_time_ns = None
            self.doa_hist_speech = []
            self._mono_ring.clear()
            # also pause RViz updates
            self._dir_vec = None


    # ---------------- RViz Marker helpers ----------------

    def _publish_marker(self, action_add=True, vec=None):
        if not self.marker_enabled:
            return
        m = Marker()
        m.header.frame_id = self.marker_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "doa"
        m.id = 1
        if not action_add:
            m.action = Marker.DELETE
            self.marker_pub.publish(m)
            return

        # Orientation from direction vector (fallback to +X)
        vx, vy, vz = (1.0, 0.0, 0.0) if vec is None else vec
        yaw = math.atan2(vy, vx)
        pitch = math.atan2(-vz, math.sqrt(vx*vx + vy*vy))
        qx, qy, qz, qw = quat_from_yaw_pitch(yaw, pitch)

        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.scale.x = float(self.marker_len)
        m.scale.y = float(self.marker_shaft_d)
        m.scale.z = float(self.marker_head_d)
        m.color.r = float(self.marker_r)
        m.color.g = float(self.marker_g)
        m.color.b = float(self.marker_b)
        m.color.a = float(self.marker_a)
        self.marker_pub.publish(m)
        self._last_marker_publish = time.time()

    def _update_marker_visibility(self):
        # Called from _wg_tick to show/hide marker based on voice activity & staleness
        now = time.time()
        if self._dir_vec is not None:
            self._publish_marker(action_add=True, vec=self._dir_vec)



    def _finalize_utterance(self, t_end_ns: int):
        if not self._utt_active:
            return
        # build int16 audio
        if not self._utt_samples:
            # nothing
            self._utt_active = False
            return
        audio_i16 = np.concatenate(self._utt_samples).astype(np.int16)
        # compute mid timestamp for DoA association
        dur_sec = len(audio_i16) / float(self.fs)
        t_mid_sec = (t_end_ns * 1e-9) - 0.5 * dur_sec
        az_for_utt = self._az_at(t_mid_sec)
        # stamp = t_end
        stamp_msg = self._make_time_msg_from_ns(t_end_ns)
        
        
        angles_only = [dh[1] for dh in self.doa_hist]
        counter_doa = Counter(angles_only)
        latest_angle = counter_doa.most_common(5)
        
        az = math.radians(latest_angle[0][0])                   # NEW

        
        # ---- speaker verify/enroll FIRST so drain_outputs can annotate ----
        self._speaker_process_final(audio_i16, stamp_msg)

        if self.enable_stt:
            self.worker.submit(audio_i16, stamp_msg, latest_angle[0][0])
        else:
            self.get_logger().info("STT disabled: dropping finalized utterance audio")

        # reset
        self._utt_active = False
        self._utt_start_time_ns = None
        self._utt_samples = []
        self.doa_hist_speech = []
        self._last_speech_time_ns = None
        self.utterances_finalized += 1
        self._maybe_publish_perf()


    def _load_all_profiles(self):
        """
        Load all .npz profiles from self.profiles_dir.

        Returns:
            dict[speaker_id -> np.ndarray embedding]
        """
        profiles = {}
        try:
            for fname in os.listdir(self.profiles_dir):
                if not fname.endswith(".npz"):
                    continue
                sid = os.path.splitext(fname)[0]
                path = os.path.join(self.profiles_dir, fname)
                try:
                    data = np.load(path)
                    emb = data["emb"].astype(np.float32)
                    profiles[sid] = emb
                except Exception as e:
                    self.get_logger().warn(f"[spk] Failed to load profile {path}: {e}")
        except Exception as e:
            self.get_logger().warn(f"[spk] Error listing profiles in {self.profiles_dir}: {e}")
        return profiles


    def _speaker_process_final(self, audio_i16: np.ndarray, stamp_msg: TimeMsg):
        if self._spk_mode == "idle":
            return

        pid = self._spk_target
        ts_ns = int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)

        # Save wav (optional)
        wav_path = os.path.join(self.profiles_dir, f"{pid}_{ts_ns}.wav")
        if self.spk_save_wav:
            sf.write(wav_path, audio_i16.astype(np.int16), self.fs, subtype="PCM_16")

        # Compute embedding (Resemblyzer wants float wav at 16k; preprocess_wav can read file)
        # If we saved wav, we can just read it via preprocess_wav; otherwise convert directly.
        if self.spk_save_wav:
            wav = preprocess_wav(wav_path)
        else:
            wav = (audio_i16.astype(np.float32) / 32768.0)
        emb = self.spk_encoder.embed_utterance(wav)

        prof_path = os.path.join(self.profiles_dir, f"{pid}.npz")

        if self._spk_mode == "enroll":
            np.savez(prof_path, emb=emb.astype(np.float32), fs=self.fs, created_ns=ts_ns)
            
            # cache latest speaker status (for annotating text)
            self._last_spk = {
                "id": pid,
                "score": 1.0,
                "match": True,
                "threshold": float(self.spk_threshold),
                "ts_ns": ts_ns,
                "kind": "enroll_done",
            }
            
            payload = {
                "kind": "enroll_done",
                "speaker_id": pid,
                "profile_path": prof_path,
                "wav_path": wav_path if self.spk_save_wav else None,
                "stamp": {"sec": stamp_msg.sec, "nanosec": stamp_msg.nanosec},
            }
            self.pub_spk.publish(String(data=json.dumps(payload)))
            self.get_logger().info(f"[spk] Enrolled '{pid}' -> {prof_path}")

        elif self._spk_mode == "verify":
            # New behavior: identify among ALL enrolled profiles in profiles_dir.
            profiles = self._load_all_profiles()
            if not profiles:
                payload = {
                    "kind": "verify_error",
                    "error": "no_profiles",
                    "speaker_id": None,
                    "profile_dir": self.profiles_dir,
                }
                self.pub_spk.publish(String(data=json.dumps(payload)))
                self.get_logger().warn(f"[spk] No profiles found in {self.profiles_dir}")
            else:
                best_id = None
                best_score = -1.0
                all_scores = []

                for sid, ref in profiles.items():
                    s = _cosine(emb, ref)
                    s_f = float(s)
                    all_scores.append((sid, s_f))
                    if s_f > best_score:
                        best_score = s_f
                        best_id = sid

                is_match = bool(best_score >= self.spk_threshold)

                # Cache latest result for /audio/stt_text annotation
                self._last_spk = {
                    "id": best_id,
                    "score": float(best_score),
                    "match": is_match,
                    "threshold": float(self.spk_threshold),
                    "ts_ns": ts_ns,
                    "kind": "verify_done",
                    # optional: scores per speaker if you want to debug
                    "scores": {sid: float(s) for sid, s in all_scores},
                }

                payload = {
                    "kind": "verify_done",
                    "speaker_id": best_id,
                    "score": float(best_score),
                    "threshold": float(self.spk_threshold),
                    "match": is_match,
                    "scores": all_scores,  # list of [speaker_id, score]
                    "wav_path": wav_path if self.spk_save_wav else None,
                    "stamp": {"sec": stamp_msg.sec, "nanosec": stamp_msg.nanosec},
                }
                self.pub_spk.publish(String(data=json.dumps(payload)))
                self.get_logger().info(
                    f"[spk] Identify -> {best_id} score={best_score:.3f} "
                    f"(profiles={len(profiles)}) match={is_match}"
                )


        # Disarm after one utterance
        if self._spk_mode == "enroll":
            self._spk_mode = "idle"
            self._spk_target = None



    def _wg_tick(self):
    
        if not self.enable_gate:
            # still advance next window schedule so we don't backlog
            self._next_window_at_ns = self.get_clock().now().nanoseconds + int(self.wg_hop_ms * 1e6)
            return

    
        now_ns = self.get_clock().now().nanoseconds
        # 8.1 Launch a new window if due
        if now_ns >= self._next_window_at_ns:
            win_s = int(self.wg_window_ms * self.fs / 1000)
            hop_s = int(self.wg_hop_ms * self.fs / 1000)
            if len(self._mono_ring) >= win_s:
                # take newest window
                mono_np = np.frombuffer(np.array(self._mono_ring, dtype=np.int16)[-win_s:].tobytes(), dtype='<i2')
                # time mapping: window ends at self.audio_time (end of received audio)
                t_end_ns = self.audio_time.nanoseconds
                t_start_ns = t_end_ns - int((win_s / float(self.fs)) * 1e9)
                # submit float32 for gate
                audio_f32 = mono_np.astype(np.float32) / 32768.0
                self.wg.submit({"t_start_ns": t_start_ns, "t_end_ns": t_end_ns, "audio_f32": audio_f32})
            # schedule next window
            self._next_window_at_ns = now_ns + int(self.wg_hop_ms * 1e6)

        # 8.2 Drain gate results and do endpointing
        advanced_any = False

        while True:
            try:
                res = self._win_out.get_nowait()
            except queue.Empty:
                break

            lat_gate_ms = float(res.get("lat_ms", 0.0))
            self.lat_gate_ms_ema = self._ema(self.lat_gate_ms_ema, lat_gate_ms, self._ema_alpha)
            self.windows_processed += 1
            self._maybe_publish_perf({"last_gate_lat_ms": lat_gate_ms})

            text = (res["text"] or "").strip()
            avg_lp = float(res.get("avg_logprob", -99.0))
            max_nsp = float(res.get("max_no_speech", 1.0))
            t_start_ns = int(res["t_start_ns"])
            t_end_ns = int(res["t_end_ns"])
            dur_ms = (t_end_ns - t_start_ns) / 1e6

            is_speech = (
                len(text) >= self.wg_min_chars and
                avg_lp >= self.wg_min_avg_logprob) # and
                #max_nsp <= self.wg_max_no_speech
            #)

            self.get_logger().info(f"BEFORE!!! {avg_lp} {max_nsp} {text} {is_speech}")            

            if is_speech:

                # NEW: summarize angle "so far" and publish a partial item
                mode_deg, mean_deg, top_list = self._doa_summary()
                ts_sec = t_end_ns * 1e-9  # or time.time(), but this lines up with window end

                partial_payload = {
                    "kind": "partial_gate",
                    "t_start_ns": t_start_ns,
                    "t_end_ns":   t_end_ns,
                    "duration_ms": dur_ms,
                    "ts": ts_sec,                     # matches context field `ts`
                    "text": text,
                    "avg_logprob": avg_lp,
                    "max_no_speech": max_nsp,
                    "lat_ms": lat_gate_ms,

                    # 🔽 FLAT FIELDS to match task_registry
                    "doa_angle_mode_deg": mode_deg,
                    "doa_angle_mean_deg": mean_deg,

                    # keep nested object too, if you still want it
                    "doa": {
                        "angle_mode_deg": mode_deg,
                        "angle_mean_deg": mean_deg,
                        "top_angles": top_list,
                    },
                }

                self.pub_partial.publish(
                    String(data=json.dumps(partial_payload, ensure_ascii=False))
                )

            
                # start or continue utterance
                angles_only = [dh[1] for dh in self.doa_hist]
                counter_doa = Counter(angles_only)
                latest_angle = counter_doa.most_common(5)
                numerator = sum(item * count for item, count in counter_doa.items())

                # Denominator: sum of counts (total number of items)
                denominator = sum(counter_doa.values()) # or item_counts.total() in Python 3.10+
                weighted_average = numerator / denominator
                
                az = math.radians(latest_angle[0][0])                   # NEW
                self._dir_vec = (math.cos(az), math.sin(az), 0.0)    # NEW
                self._dir_vec_ts = time.time()           
                self.get_logger().info(f"{self.wg_min_chars} {self.wg_min_avg_logprob} {self.wg_max_no_speech} {avg_lp} {max_nsp} {text} {is_speech} {latest_angle} {weighted_average}")
                
                # Update RViz marker visibility each tick
                self._update_marker_visibility()
                
                
                    
                
                # append raw samples for exact final transcription
                # (pull corresponding raw int16 slice)
                win_samps = int(self.wg_window_ms * self.fs / 1000)
                hop_s = int(self.wg_hop_ms * self.fs / 1000)
                # slice again (safe & simple)
                mono_np = np.frombuffer(np.array(self._mono_ring, dtype=np.int16)[-win_samps:].tobytes(), dtype='<i2')
                
                if not self._utt_active:
                    self._utt_active = True
                    self._utt_start_time_ns = t_start_ns
                    self._utt_samples = []
                    self.doa_hist_speech = []
                    self._utt_samples.append(mono_np.copy())
                else:
                    # Subsequent chunks: only append the NEW part (the hop)
                    hop_np = mono_np[-hop_s:] if hop_s < len(mono_np) else mono_np
                    self._utt_samples.append(hop_np.copy())
                    
                self._last_speech_time_ns = t_end_ns


                self.doa_hist_speech.extend(self.doa_hist)
                # hard stop on max length
                if (t_end_ns - self._utt_start_time_ns) / 1e6 >= self.wg_max_utter_ms:
                    self._finalize_utterance(t_end_ns)
                    advanced_any = True

            else:
                # consider silence: if in-utterance and silence long enough, finalize
                if self._utt_active:
                    if (t_end_ns - (self._last_speech_time_ns or t_end_ns)) / 1e6 >= self.wg_end_silence_ms:
                        self._finalize_utterance(t_end_ns)
                        advanced_any = True

        # nothing else
        #self.get_logger().info(f"advanced_any {advanced_any}")
        return


    def _ns(self, t: Time) -> int:
        return t.nanoseconds

    def _make_time_msg_from_ns(self, ns: int) -> TimeMsg:
        msg = TimeMsg()
        msg.sec = ns // 1_000_000_000
        msg.nanosec = ns % 1_000_000_000
        return msg


    # ADD helper to convert builtin_interfaces/Time or rclpy Time to float seconds
    def _to_sec(self, t):
        if isinstance(t, Time):
            return t.nanoseconds * 1e-9
        # builtin_interfaces/Time
        return float(t.sec) + 1e-9 * float(t.nanosec)

    # --- new callback ---
    def on_doa_vec(self, msg: Vector3Stamped):
    
        # Compute azimuth from vector; 0°=+X, +90°=+Y (matches your DoA node)
        az = math.degrees(math.atan2(msg.vector.y, msg.vector.x))
        t = self._to_sec(msg.header.stamp)
        if int(az) not in [-149,-135,-180,-156,-143]:
            self.doa_hist.append((t, az))

        # drop old
        now_sec = self._to_sec(self.get_clock().now())
        while self.doa_hist and (now_sec - self.doa_hist[0][0] > self.doa_keep_sec):
            self.doa_hist.popleft()

    # --- helper to query nearest azimuth at a target time (seconds) ---
    def _az_at(self, t_target_sec: float) -> float:
        if not self.doa_hist:
            return 0.0
        times = [t for (t, _) in self.doa_hist]
        i = bisect.bisect_left(times, t_target_sec)
        if i <= 0:
            return self.doa_hist[0][1]
        if i >= len(times):
            return self.doa_hist[-1][1]
        # choose closer of neighbors
        t0, a0 = self.doa_hist[i-1]
        t1, a1 = self.doa_hist[i]
        return a0 if (t_target_sec - t0) <= (t1 - t_target_sec) else a1
 
 
    def _doa_summary(self):
        """
        Returns (mode_deg, mean_deg, top_list) from self.doa_hist ([(t, az_deg), ...]).
        If empty, returns (0.0, 0.0, []).
        """
        if not self.doa_hist:
            return 0.0, 0.0, []
        angles_only = [float(az) for (_, az) in self.doa_hist]
        counts = Counter(int(round(a)) for a in angles_only)  # 1° binning for stability
        mode_deg = float(counts.most_common(1)[0][0]) if counts else 0.0
        mean_deg = float(sum(angles_only) / max(len(angles_only), 1))
        top5 = counts.most_common(5)
        # e.g., [[angle_deg, count], ...]
        top_list = [[float(k), int(v)] for (k, v) in top5]
        return mode_deg, mean_deg, top_list
       
        
    # ---- handlers ----
    def on_azimuth(self, msg: Float32):
        
        latest_azimuth = int(msg.data)
        
        if latest_azimuth not in [-149,-135,-180]:
        
            self.latest_azimuth = float(msg.data)
            self.latest_stamp = now_to_msg(self.get_clock())
            #rounded = 5 * round(self.latest_azimuth / 5)
            
            self.angle_hist.append(self.latest_azimuth)

    def show_top_angles(self):
        if not self.angle_hist:
            return
        counts = Counter(self.angle_hist)
        top5 = counts.most_common(5)
        display = " | ".join([f"{ang:.1f}° ({cnt})" for ang, cnt in top5])
        #self.get_logger().info(f"Top5 angles (last {len(self.angle_hist)}): {display}")



    def on_audio_bytes(self, msg: UInt8MultiArray):
        """
        Parse PCM16LE interleaved bytes with TC channels and extract one lane (self.pick_lane).
        Feed mono bytes to VAD; submit utterances to worker with latest DoA.
        """
        
        if self._tts_busy:
            return

        
        b = bytes(msg.data)  # convert list[int] to bytes efficiently
        # combine with any leftover for full frames of all channels
        self._partial_bytes.extend(b)

        bytes_per_frame_all = 2 * self.TC
        usable = len(self._partial_bytes) - (len(self._partial_bytes) % bytes_per_frame_all)
        if usable <= 0:
            return

        blob = self._partial_bytes[:usable]
        del self._partial_bytes[:usable]

        # Interpret as int16, reshape, pick lane
        i16 = np.frombuffer(blob, dtype='<i2')
        if i16.size % self.TC != 0:
            return
        try:
            frames = i16.reshape(-1, self.TC)
        except ValueError:
            return

        
        # --- in on_audio_bytes() after you form `frames` (shape [N, TC]) ---
        n_samples = frames.shape[0]
        # Advance audio_time by n_samples/fs
        dt_ns = int((n_samples / float(self.fs)) * 1e9)
        self.audio_time = Time(nanoseconds=(self.audio_time.nanoseconds + dt_ns))


        mono = self._select_mono(frames)
        mono_bytes = mono.tobytes()

        self._mono_ring.extend(mono.tolist())
        
        '''
        self.get_logger().info(
            f"Voice detected angle: {self.latest_azimuth}"
        )
        '''
        
            
        '''
        # --- still in on_audio_bytes(), where you iterate utter_list ---
        for utt in utter_list:
            audio_i16 = np.frombuffer(utt, dtype='<i2')
            # Utterance timing: end at current audio_time; duration from sample count
            dur_sec = len(audio_i16) / float(self.fs)
            t_end = self.audio_time
            t_mid_sec = self._to_sec(t_end) - 0.5 * dur_sec
            az_for_utt = self._az_at(t_mid_sec)

            angles_only = [dh[1] for dh in self.doa_hist]
            counter_doa = Counter(angles_only)

            self.get_logger().info(f"Voice detected angle: {self.latest_azimuth} time {self._to_sec(t_end) - dur_sec} {t_mid_sec} {t_end} at {counter_doa.most_common(3)} {np.mean(angles_only)} {self.doa_hist}")

            # Build a stamp for publishing/metadata = t_end (or t_mid if you prefer)
            stamp_msg = TimeMsg()
            t_end_ns = t_end.nanoseconds
            stamp_msg.sec = int(t_end_ns // 1_000_000_000)
            stamp_msg.nanosec = int(t_end_ns % 1_000_000_000)

            # Submit to worker with the aligned azimuth
            self.worker.submit(audio_i16, stamp_msg, az_for_utt)
        '''

    def drain_outputs(self):
        while True:
            try:
                item = self.out_q.get_nowait()
            except queue.Empty:
                break
            
            now_wall = time.time()
            # Default stamp to current time if missing
            sec = int(item.get("stamp", {}).get("sec", 0))
            nsec = int(item.get("stamp", {}).get("nanosec", 0))
            t_end = sec + 1e-9 * nsec
            e2e_ms = max(0.0, (now_wall - t_end) * 1000.0) if (sec or nsec) else 0.0

            kind = item.get("kind", "final_asr")
            if kind == "final_asr":
                lat_asr_ms = float(item.get("lat_ms", 0.0))
                self.lat_asr_ms_ema = self._ema(self.lat_asr_ms_ema, lat_asr_ms, self._ema_alpha)
                self.lat_e2e_ms_ema = self._ema(self.lat_e2e_ms_ema, e2e_ms, self._ema_alpha)
                self._maybe_publish_perf({"last_asr_lat_ms": lat_asr_ms, "last_e2e_ms": e2e_ms})

                
            text = (item.get("text") or "").strip()
            az = float(item.get("azimuth_deg", 0.0))
            stamp = item.get("stamp", {"sec": 0, "nanosec": 0})
            lang = item.get("language")
            duration = item.get("duration")

            # -------- /audio/stt_text as JSON --------
            if text:
                payload_text = {
                    "text": text,
                }

                # Attach speaker verification info if recent enough
                spk = getattr(self, "_last_spk", None)
                if spk is not None:
                    self.get_logger().info(
                        f"Hello"
                    )
                    now_ns = self.get_clock().now().nanoseconds
                    age_ms = (now_ns - int(spk.get("ts_ns", 0))) / 1e6
                    if age_ms <= getattr(self, "stt_text_verify_ttl_ms", 100000):
                        payload_text.update({
                            "speaker_id":      spk.get("id", "unknown"),
                            "speaker_score":   float(spk.get("score", 0.0)),
                            "speaker_match":   bool(spk.get("match", False)),
                            "speaker_threshold": float(spk.get("threshold", self.spk_threshold)),
                            "speaker_age_ms":  age_ms,
                        })

                # Publish JSON instead of raw text
                self.pub_text.publish(
                    String(data=json.dumps(payload_text, ensure_ascii=False))
                )


            
            # Publish JSON w/ angle + metadata
            payload = {
                "text": text,
                "azimuth_deg": az,
                "language": lang,
                "duration": duration,
                "stamp": stamp
            }
            self.pub_json.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            

    def destroy_node(self):
        try:
            self.worker.stop()
            self.wg.stop()
            self._publish_marker(action_add=False)
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = STTFasterWhisperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

