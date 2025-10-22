#!/usr/bin/env python3
import queue, threading, math, json, numpy as np
from collections import deque, Counter
from faster_whisper import WhisperModel

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import UInt8MultiArray, Float32, String
from geometry_msgs.msg import Vector3Stamped
import math
from rclpy.time import Time

class WhisperGateWorker(threading.Thread):
    """Runs Whisper on short windows and reports gate metrics."""
    def __init__(self, model, language, translate, out_q, logger):
        super().__init__(daemon=True)
        self.model = model
        self.language = language
        self.translate = translate
        self.q = queue.Queue()
        self.out_q = out_q
        self.log = logger
        self.running = True

    def submit(self, t_end_ns, audio_f32):
        self.q.put((t_end_ns, audio_f32))

    def run(self):
        while self.running:
            try:
                t_end_ns, audio_f32 = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                segs, info = self.model.transcribe(
                    audio_f32, language=self.language,
                    task="transcribe", vad_filter=True,
                    beam_size=5, best_of=5
                )
                # --- FIX: never call len() on segs (it's a generator) ---
                text_parts = []
                seg_cnt = 0
                sum_lp = 0.0
                max_nsp = 0.0

                for s in segs:  # iterate the generator once
                    seg_cnt += 1
                    text_parts.append(s.text or "")
                    if getattr(s, "avg_logprob", None) is not None:
                        sum_lp += float(s.avg_logprob)
                    if getattr(s, "no_speech_prob", None) is not None:
                        max_nsp = max(max_nsp, float(s.no_speech_prob))

                text = "".join(text_parts).strip()
                avg_lp = (sum_lp / seg_cnt) if seg_cnt > 0 else -99.0

                
                self.out_q.put({
                    "t_end_ns": t_end_ns,
                    "text": text,
                    "avg_lp": avg_lp,
                    "max_nsp": max_nsp,
                    "segments": seg_cnt,
                })
            except Exception as e:
                self.log.warn(f"Whisper gate error: {e}")

    def stop(self): self.running=False


class WhisperSpeechGate(Node):
    """Publishes angle when Whisper detects speech in sliding windows."""
    def __init__(self):
        super().__init__("whisper_speech_gate")

        # --- Parameters ---
        self.declare_parameter("fs_hz", 16000)
        self.declare_parameter("model_size", "medium")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("compute_type", "int8")
        self.declare_parameter("language", "en")

        # window & gate thresholds
        self.declare_parameter("window_ms", 3000)
        self.declare_parameter("hop_ms", 1000)
        self.declare_parameter("wg_min_chars", 4)
        self.declare_parameter("wg_min_avg_logprob", -1.0)
        self.declare_parameter("wg_max_no_speech", 0.6)

        # load params
        self.fs = int(self.get_parameter("fs_hz").value)
        self.win_s = int(self.fs * self.get_parameter("window_ms").value / 1000)
        self.hop_s = int(self.fs * self.get_parameter("hop_ms").value / 1000)
        self.min_chars = int(self.get_parameter("wg_min_chars").value)
        self.min_lp = float(self.get_parameter("wg_min_avg_logprob").value)
        self.max_ns = float(self.get_parameter("wg_max_no_speech").value)

        # publishers/subscribers
        self.pub_gate = self.create_publisher(String, "/audio/whisper_gate", 10)
        self.sub_doa_vec = self.create_subscription(
            Vector3Stamped, "/audio/doa_raw", self.on_doa_vec, 10
        )
        qos_audio = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.sub_audio = self.create_subscription(UInt8MultiArray, "/mic/audio", self.on_audio, qos_audio)

        # Keep ~5 seconds of DoA history (sorted by time)
        self.doa_hist = deque()  # list of (t_sec_float, az_deg)
        self.doa_keep_sec = 20.0

        # gate state

        self.ring = deque(maxlen=self.win_s*2)
        self._next_eval_ns = 0

        # background worker
        self.out_q = queue.Queue()
        self.model = WhisperModel(
            str(self.get_parameter("model_size").value),
            device=str(self.get_parameter("device").value),
            compute_type=str(self.get_parameter("compute_type").value)
        )
        self.worker = WhisperGateWorker(self.model,
            self.get_parameter("language").value, False, self.out_q, self.get_logger())
        self.worker.start()

        self.create_timer(0.05, self.tick)
        self.get_logger().info("Whisper speech gate running.")


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
            
            
    def on_audio(self, msg: UInt8MultiArray):
        i16 = np.frombuffer(bytes(msg.data), dtype="<i2")
        self.ring.extend(i16.tolist())

    def tick(self):
        now_ns = self.get_clock().now().nanoseconds
        # schedule whisper every hop_ms
        if now_ns >= self._next_eval_ns and len(self.ring) >= self.win_s:
            samples = np.array(list(self.ring)[-self.win_s:], dtype=np.int16)
            audio_f32 = samples.astype(np.float32) / 32768.0
            self.worker.submit(now_ns, audio_f32)
            self._next_eval_ns = now_ns + int(self.hop_s / self.fs * 1e9)

        # drain results
        while True:
            try:
                res = self.out_q.get_nowait()
            except queue.Empty:
                break
            text = res["text"]; lp=res["avg_lp"]; ns=res["max_nsp"]
            is_speech = (len(text)>=self.min_chars and lp>=self.min_lp and ns<=self.max_ns)
            
            self.get_logger().info(f"{self.min_chars} {self.min_lp} {self.max_ns} {lp} {ns} {text} {is_speech}")
            
            if is_speech:
            
                angles_only = [dh[1] for dh in self.doa_hist]
                counter_doa = Counter(angles_only)
                latest_angle = counter_doa.most_common(1)[0][0]
                
                msg = {"angle_deg": latest_angle,
                       "text": text, "avg_lp": lp, "no_speech": ns}
                self.pub_gate.publish(String(data=json.dumps(msg)))

                self.get_logger().info(f"Speech detected @ {latest_angle:+.1f}°, conf={lp:.2f}, nsp={ns:.2f}")

    def destroy_node(self):
        try: self.worker.stop()
        except: pass
        super().destroy_node()

def main():
    rclpy.init()
    node = WhisperSpeechGate()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__=="__main__":
    main()

