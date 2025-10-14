#!/usr/bin/env python3
import rclpy, io, time, numpy as np, soundfile as sf
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, String
from faster_whisper import WhisperModel
from collections import deque

RATE = 16000
BYTES_PER_SAMPLE = 2

class SttSubscriber(Node):
    def __init__(self):
        super().__init__('stt_subscriber')

        # Parameters
        self.declare_parameter('input_topic', '/mic/audio')
        self.declare_parameter('output_topic', '/stt/text')
        self.declare_parameter('partial_topic', '/stt/text_partial')
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('compute_type', 'auto')
        self.declare_parameter('language', 'en')
        self.declare_parameter('vad_filter', True)
        self.declare_parameter('beam_size', 1)
        self.declare_parameter('window_sec', 2.0)
        self.declare_parameter('hop_sec', 1.0)

        self.input_topic  = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.partial_topic= self.get_parameter('partial_topic').value
        self.window  = float(self.get_parameter('window_sec').value)
        self.hop     = float(self.get_parameter('hop_sec').value)

        self.model = WhisperModel(
            self.get_parameter('model_size').value,
            compute_type=self.get_parameter('compute_type').value,
            device='cpu'
        )

        self._buf = deque()
        self._last = time.time()
        self._tail = np.zeros(int(self.hop * RATE), dtype=np.int16)

        self.sub = self.create_subscription(UInt8MultiArray, self.input_topic, self.on_audio, 200)
        self.pub_text = self.create_publisher(String, self.output_topic, 10)
        self.pub_partial = self.create_publisher(String, self.partial_topic, 10)

        self.get_logger().info(f'STT listening on {self.input_topic}')

    def on_audio(self, msg: UInt8MultiArray):
        # Append raw bytes
        self._buf.extend(msg.data)
        if (time.time() - self._last) >= self.window:
            self._last = time.time()
            self._flush_and_transcribe()

    def _flush_and_transcribe(self):
        if not self._buf:
            return
        arr = np.frombuffer(bytes(self._buf), dtype=np.uint8)
        self._buf.clear()

        # Ensure 16-bit alignment
        if len(arr) % BYTES_PER_SAMPLE != 0:
            arr = arr[:len(arr) - (len(arr) % BYTES_PER_SAMPLE)]

        # Convert to int16 little-endian PCM
        audio_i16 = arr.view('<i2')

        # Overlap (context)
        audio_i16 = np.concatenate([self._tail, audio_i16])
        self._tail = audio_i16[-int(self.hop * RATE):].copy()

        # Normalize to float32 [-1,1]
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        # Write to in-memory WAV and transcribe
        bio = io.BytesIO()
        sf.write(bio, audio_f32, RATE, format='WAV'); bio.seek(0)

        segments, _ = self.model.transcribe(
            bio,
            language=self.get_parameter('language').value,
            vad_filter=self.get_parameter('vad_filter').value,
            beam_size=int(self.get_parameter('beam_size').value),
        )

        parts = []
        for s in segments:
            t = s.text.strip()
            if t:
                parts.append(t)
                p = String(); p.data = t
                self.pub_partial.publish(p)

        full = " ".join(parts).strip()
        if full:
            m = String(); m.data = full
            self.pub_text.publish(m)
            self.get_logger().info(f'ASR: {full}')

def main():
    rclpy.init()
    n = SttSubscriber()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
