#!/usr/bin/env python3
import queue
import threading
import time
import json
import struct
from collections import deque, Counter

import numpy as np
import webrtcvad
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


class FasterWhisperWorker(threading.Thread):
    """
    Background transcriber to avoid blocking ROS callbacks.
    """
    def __init__(self, model, translate, language, out_queue, logger):
        super().__init__(daemon=True)
        self.model = model #WhisperModel(model_size, device=device, compute_type=compute_type)
        self.translate = translate
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
                # Convert to float32 in [-1,1]
                audio_f32 = (audio_int16.astype(np.float32) / 32768.0)
                task = "translate" if self.translate else "transcribe"
                segments, info = self.model.transcribe(
                    audio_f32,
                    language=self.language, task=task,
                    vad_filter=False,  # VAD already handled
                    beam_size=5,
                    best_of=5
                )
                text = "".join(seg.text for seg in segments).strip()
                self.out_q.put({
                    "text": text,
                    "language": info.language if hasattr(info, "language") else None,
                    "duration": getattr(info, "duration", None),
                    "azimuth_deg": az,
                    "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec}
                })
            except Exception as e:
                self.log.warn(f"Transcribe error: {e}")

    def stop(self):
        self.running = False

class WhisperGateWorker(threading.Thread):
    """
    Consumes windows of float32 audio and returns gate metrics.
    Input item: dict{ 't_start_ns', 't_end_ns', 'audio_f32' }
    Output item: dict{ 't_start_ns','t_end_ns','text','avg_logprob','max_no_speech','segments' }
    """
    def __init__(self, model: WhisperModel, language, translate, out_q, logger):
        super().__init__(daemon=True)
        self.model = model
        self.language = language
        self.translate = translate
        self.q = queue.Queue()
        self.out_q = out_q
        self.log = logger
        self.running = True

    def submit(self, item: dict):
        self.q.put(item)

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                task = "translate" if self.translate else "transcribe"
                segments, info = self.model.transcribe(
                    item["audio_f32"],
                    language=self.language, task=task,
                    vad_filter=True,  # <-- Whisper VAD
                    beam_size=5, best_of=5
                )
                text_parts, seg_cnt, sum_lp, max_nsp = [], 0, 0.0, 0.0
                for seg in segments:
                    seg_cnt += 1
                    text_parts.append(seg.text)
                    if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                        sum_lp += float(seg.avg_logprob)
                    if hasattr(seg, "no_speech_prob") and seg.no_speech_prob is not None:
                        max_nsp = max(max_nsp, float(seg.no_speech_prob))
                text = "".join(text_parts).strip()
                avg_lp = (sum_lp / max(seg_cnt, 1)) if seg_cnt else -99.0
                self.out_q.put({
                    "t_start_ns": item["t_start_ns"],
                    "t_end_ns": item["t_end_ns"],
                    "text": text,
                    "avg_logprob": avg_lp,
                    "max_no_speech": max_nsp,
                    "segments": seg_cnt
                })
            except Exception as e:
                self.log.warn(f"WhisperGate window error: {e}")


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
        self.declare_parameter("stt_device", "cpu")
        self.declare_parameter("stt_compute_type", "int8")

        # Gating / endpointing (lighter) defaults
        self.declare_parameter("wg_model_size", "small")
        self.declare_parameter("wg_device", "cpu")
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

        self.wg_window_ms = int(self.get_parameter("wg_window_ms").value)
        self.wg_hop_ms = int(self.get_parameter("wg_hop_ms").value)
        self.wg_end_silence_ms = int(self.get_parameter("wg_end_silence_ms").value)
        self.wg_max_utter_ms = int(self.get_parameter("wg_max_utter_ms").value)
        self.wg_min_chars = int(self.get_parameter("wg_min_chars").value)
        self.wg_min_avg_logprob = float(self.get_parameter("wg_min_avg_logprob").value)
        self.wg_max_no_speech = float(self.get_parameter("wg_max_no_speech").value)


        # Load params
        self.fs = int(self.get_parameter("fs_hz").value)
        self.TC = int(self.get_parameter("total_channels").value)
        self.mic_lanes = [int(x) for x in self.get_parameter("mic_lanes").value]
        self.ref_ch = int(self.get_parameter("ref_ch").value)
        assert 0 <= self.ref_ch < len(self.mic_lanes), "ref_ch out of range of mic_lanes"
        self.pick_lane = self.mic_lanes[self.ref_ch]

        self.model_size = str(self.get_parameter("model_size").value)
        self.device = str(self.get_parameter("device").value)
        self.compute_type = str(self.get_parameter("compute_type").value)
        self.translate = bool(self.get_parameter("translate").value)
        self.language = str(self.get_parameter("language").value).strip() or None

        self.stt_model_size = (str(self.get_parameter("stt_model_size").value) or self.model_size)
        self.stt_device = (str(self.get_parameter("stt_device").value) or self.device)
        self.stt_compute_type = (str(self.get_parameter("stt_compute_type").value) or self.compute_type)

        self.wg_model_size = (str(self.get_parameter("wg_model_size").value) or self.model_size)
        self.wg_device = (str(self.get_parameter("wg_device").value) or self.device)
        self.wg_compute_type = (str(self.get_parameter("wg_compute_type").value) or self.compute_type)

        # Load separate models
        self.gate_model = WhisperModel(
            self.wg_model_size, device=self.wg_device, compute_type=self.wg_compute_type
        )
        self.worker_model = WhisperModel(
            self.stt_model_size, device=self.stt_device, compute_type=self.stt_compute_type
        )



        # Publishers
        self.pub_text = self.create_publisher(String, "/audio/stt_text", 10)
        self.pub_json = self.create_publisher(String, "/audio/stt_doa_json", 10)

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

        
        # Background transcriber
        self.out_q = queue.Queue()
        self.worker = FasterWhisperWorker(
            self.worker_model,
            self.translate, self.language,
            self.out_q, self.get_logger()
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

        # Reuse the same loaded model from self.worker for gating (no 2nd load)
        #model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self.wg = WhisperGateWorker(
            self.gate_model,  # reuse model instance
            self.language, self.translate, self._win_out, self.get_logger()
        )
        self.wg.start()
        
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

        self.get_logger().info(
            f"STT node: lane={self.pick_lane} fs={self.fs} TC={self.TC} model={self.model_size} "
            f"device={self.device}/{self.compute_type} translate={self.translate}"
        )


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

        
        # hand to your existing worker (high-quality transcription)
        self.worker.submit(audio_i16, stamp_msg, latest_angle[0][0])
        
        # reset
        self._utt_active = False
        self._utt_start_time_ns = None
        self._utt_samples = []
        self.doa_hist_speech = []
        self._last_speech_time_ns = None


    def _wg_tick(self):
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

            text = (res["text"] or "").strip()
            avg_lp = float(res.get("avg_logprob", -99.0))
            max_nsp = float(res.get("max_no_speech", 1.0))
            t_start_ns = int(res["t_start_ns"])
            t_end_ns = int(res["t_end_ns"])
            dur_ms = (t_end_ns - t_start_ns) / 1e6

            is_speech = (
                len(text) >= self.wg_min_chars and
                avg_lp >= self.wg_min_avg_logprob and
                max_nsp <= self.wg_max_no_speech
            )

            

            if is_speech:
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


        mono = frames[:, self.pick_lane].astype(np.int16)
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
            text = (item.get("text") or "").strip()
            az = float(item.get("azimuth_deg", 0.0))
            stamp = item.get("stamp", {"sec": 0, "nanosec": 0})
            lang = item.get("language")
            duration = item.get("duration")

            # Publish plain text
            if text:
                self.pub_text.publish(String(data=text))

            
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

