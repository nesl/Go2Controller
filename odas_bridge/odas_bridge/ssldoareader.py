#!/usr/bin/env python3
import math
import socket
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32, UInt8MultiArray
from geometry_msgs.msg import Vector3Stamped

def build_freq_mask(fs, nfft, band, notch_bands):
    """Return boolean mask for rfft bins to KEEP (band-pass minus notches)."""
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = (f >= band[0]) & (f <= band[1])
    if notch_bands:
        for lo, hi in notch_bands:
            keep &= ~((f >= lo) & (f <= hi))
    return keep

def gcc_phat_tdoa_banded(x, y, fs, band=(300,3400), notch_bands=None, max_tau=None):
    """TDOA (s) using GCC-PHAT with band mask and optional notch bands."""
    nfft = 1 << (int(len(x) + len(y) - 1).bit_length())
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y, nfft)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)

    mask = build_freq_mask(fs, nfft, band, notch_bands or [])
    R[~mask] = 0.0

    cc = np.fft.irfft(R, nfft)
    max_shift = nfft // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = int(np.argmax(cc)) - max_shift
    tau = shift / float(fs)
    if max_tau is not None:
        tau = float(np.clip(tau, -max_tau, +max_tau))
    return tau

class GCCPhatTcpDoA(Node):
    """
    DoA via GCC-PHAT from raw PCM16LE over TCP (e.g., `arecord | nc`).

    Publishes:
      - /audio/doa_raw (Vector3Stamped): unit vector (mic_array frame)
      - /audio/doa_raw_deg (Float32): azimuth 0:+X(front), +90:+Y(left)
      - (optional) /mic/audio (UInt8MultiArray): raw bytes passthrough
    """

    def __init__(self):
        super().__init__('gccphat_tcp_doa')

        # ---- TCP input ----
        self.declare_parameter('port', 9004)
        self.declare_parameter('republish_bytes', True)

        # ---- Stream format ----
        self.declare_parameter('fs_hz', 16000)
        self.declare_parameter('total_channels', 6)
        self.declare_parameter('mic_lanes', [1, 2, 3, 4])

        # ---- Geometry / solver ----
        self.declare_parameter('angles_deg', [0, 90, 180, -90])
        #self.declare_parameter('angles_deg', [-45, 45, 135, -135])
        self.declare_parameter('ref_ch', 0)
        self.declare_parameter('radius_m', 0.032)
        self.declare_parameter('frame_ms', 64.0)
        self.declare_parameter('hop_ms', 32.0)
        self.declare_parameter('pre_emph', 0.0)
        self.declare_parameter('min_dbfs', -65.0)
        self.declare_parameter('band_low_hz', 250.0)   # raise low cut (was 300 ok too)
        self.declare_parameter('band_high_hz', 3400.0)
        self.declare_parameter('frame_id', 'mic_array')

        # ---- NEW: noise controls ----
        # from your analyzer output:
        self.declare_parameter('notch_centers', [156.0, 180.0, 258.0, 281.0, 312.5, 336.0, 359.0, 437.5, 539.0, 562.5])
        self.declare_parameter('notch_halfwidth_hz', 8.5)  # can be a scalar or list matching centers
        self.declare_parameter('hp_cut_hz', 180.0)

        # ---- Load params ----
        self.port = int(self.get_parameter('port').value)
        self.pub_bytes = bool(self.get_parameter('republish_bytes').value)
        self.fs = int(self.get_parameter('fs_hz').value)
        self.TC = int(self.get_parameter('total_channels').value)
        self.mic_ix = [int(i) for i in self.get_parameter('mic_lanes').value]
        self.angles = [float(a) for a in self.get_parameter('angles_deg').value]
        self.ref = int(self.get_parameter('ref_ch').value)
        self.r = float(self.get_parameter('radius_m').value)
        self.frame = int(self.fs * float(self.get_parameter('frame_ms').value) / 1000.0)
        self.hop = max(1, int(self.fs * float(self.get_parameter('hop_ms').value) / 1000.0))
        self.pe = float(self.get_parameter('pre_emph').value)
        self.min_db = float(self.get_parameter('min_dbfs').value)
        self.band = (float(self.get_parameter('band_low_hz').value),
                     float(self.get_parameter('band_high_hz').value))
        self.frame_id = str(self.get_parameter('frame_id').value)
        
        # ---- Load params ----
        centers = [float(x) for x in self.get_parameter('notch_centers').value]
        hw_raw = self.get_parameter('notch_halfwidth_hz').value
        if isinstance(hw_raw, (list, tuple)):
            halfwidths = [float(x) for x in hw_raw]
            assert len(halfwidths) == len(centers), "notch_halfwidth_hz list must match notch_centers"
        else:
            halfwidths = [float(hw_raw)] * len(centers)
        
        # build [lo, hi] bands
        self.notch_bands = [(c - h, c + h) for c, h in zip(centers, halfwidths)]
        self.hp_cut = float(self.get_parameter('hp_cut_hz').value)

        # ---- Checks ----
        assert len(self.mic_ix) == 4 and len(self.angles) == 4, "Need 4 mic_lanes and 4 angles."
        assert 0 <= self.ref < 4, "ref_ch must be 0..3"
        assert max(self.mic_ix) < self.TC, "mic_lanes index exceeds total_channels."

        # ---- Geometry ----
        self.c = 343.0
        self.max_tau = (2 * self.r) / self.c
        p = np.stack([self.r * np.cos(np.radians(self.angles)),
                      self.r * np.sin(np.radians(self.angles))], axis=1)  # [4,2]
        self.Prel = p - p[self.ref:self.ref + 1, :]
        self.win = np.hanning(self.frame).astype(np.float32)

        # ---- Simple 1st-order HPF (per-channel state) ----
        # y[n] = a*(y[n-1] + x[n] - x[n-1]);  a = exp(-2π fc / fs)
        self.hp_a = float(np.exp(-2.0 * np.pi * self.hp_cut / self.fs))
        self.hp_x1 = np.zeros(4, dtype=np.float32)
        self.hp_y1 = np.zeros(4, dtype=np.float32)

        # ---- Buffers ----
        self.bytebuf = bytearray()
        self.reservoir = np.zeros((0, 4), dtype=np.float32)
        self.res_cap = self.frame * 10

        # ---- Publishers ----
        if self.pub_bytes:
            qos = QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST
            )
            self.pub_raw = self.create_publisher(UInt8MultiArray, '/mic/audio', qos)
        else:
            self.pub_raw = None
        self.pub_vec = self.create_publisher(Vector3Stamped, '/audio/doa_raw', 20)
        self.pub_deg = self.create_publisher(Float32, '/audio/doa_raw_deg', 20)

        # ---- TCP server ----
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(('0.0.0.0', self.port))
        self.srv.listen(1)
        self.srv.setblocking(False)
        self.conn = None

        # ---- Timer loop ----
        self.timer = self.create_timer(0.005, self.loop)
        self.get_logger().info(
            f"TCP DoA :{self.port} fs={self.fs} TC={self.TC} lanes={self.mic_ix} "
            f"angles={self.angles} ref={self.ref} frame={self.frame} hop={self.hop} "
            f"band={self.band} hp={self.hp_cut}Hz notches={len(self.notch_bands)}"
        )

    # ========= helpers =========
    def _hpf_inplace(self, x: np.ndarray):
        """
        In-place simple 1st-order HPF per channel on [n,4] float32 block.
        """
        a = self.hp_a
        x1 = self.hp_x1
        y1 = self.hp_y1
        for ch in range(4):
            xn = x[:, ch]
            y = np.empty_like(xn)
            prev_x = x1[ch]; prev_y = y1[ch]
            for i in range(len(xn)):
                y[i] = a * (prev_y + xn[i] - prev_x)
                prev_y = y[i]; prev_x = xn[i]
            x[:, ch] = y
            x1[ch] = prev_x; y1[ch] = prev_y

    # ========= main loop =========
    def loop(self):
        try:
            # accept
            if self.conn is None:
                try:
                    self.conn, addr = self.srv.accept()
                    self.conn.setblocking(False)
                    self.get_logger().info(f'Audio: client connected from {addr}')
                    self.bytebuf.clear()
                except BlockingIOError:
                    pass

            # recv
            if self.conn is not None:
                try:
                    data = self.conn.recv(65536)
                    if not data:
                        self.get_logger().info('Audio: client disconnected')
                        self.conn.close(); self.conn = None
                        return
                    if self.pub_raw:
                        for i in range(0, len(data), 4096):
                            msg = UInt8MultiArray(); msg.data = list(data[i:i+4096])
                            self.pub_raw.publish(msg)
                    self.bytebuf.extend(data)
                except BlockingIOError:
                    pass
                except Exception as e:
                    self.get_logger().warn(f'Audio recv error: {e}')
                    try: self.conn.close()
                    except Exception: pass
                    self.conn = None
                    return

            # bytes -> frames -> [n,4] float32
            bytes_per_frame = 2 * self.TC
            usable = len(self.bytebuf) - (len(self.bytebuf) % bytes_per_frame)
            if usable > 0:
                blob = self.bytebuf[:usable]
                del self.bytebuf[:usable]
                i16 = np.frombuffer(blob, dtype='<i2')
                try:
                    fr = i16.reshape(-1, self.TC).astype(np.float32)
                except ValueError:
                    fr = None
                if fr is not None:
                    mic4 = fr[:, self.mic_ix]  # [n,4]
                    # simple HPF on the fly (helps rumble/drive)
                    self._hpf_inplace(mic4)
                    # accumulate
                    if len(self.reservoir) == 0:
                        self.reservoir = mic4.copy()
                    else:
                        self.reservoir = np.vstack((self.reservoir, mic4))
                    if len(self.reservoir) > self.res_cap:
                        self.reservoir = self.reservoir[-self.res_cap:, :]

            # enough?
            if len(self.reservoir) < (self.frame + self.hop):
                return

            windows = 1 + ((len(self.reservoir) - self.frame) // self.hop)
            windows = int(min(windows, 3))
            for _ in range(windows):
                end = len(self.reservoir)
                start = end - self.frame
                block = self.reservoir[start:end, :].copy()
                self.reservoir = self.reservoir[: end - self.hop, :]

                # pre-emphasis + window
                if self.pe != 0.0:
                    block[1:, :] -= self.pe * block[:-1, :]
                block *= self.win[:, None]

                # level gate AFTER HPF (measure what we feed GCC)
                rms = float(np.sqrt(np.mean(block ** 2) + 1e-12)) / 32768.0
                dbfs = 20.0 * math.log10(max(rms, 1e-9))
                if dbfs < self.min_db:
                    continue

                # GCC-PHAT TDOAs vs reference
                x0 = block[:, self.ref]
                taus = np.zeros(4, dtype=np.float32)
                for ch in range(4):
                    if ch == self.ref:
                        taus[ch] = 0.0
                    else:
                        taus[ch] = gcc_phat_tdoa_banded(
                            x0, block[:, ch], self.fs,
                            band=self.band, notch_bands=self.notch_bands, max_tau=self.max_tau
                        )

                tau_rel = taus - taus[self.ref]
                try:
                    s, *_ = np.linalg.lstsq(self.Prel, self.c * tau_rel, rcond=None)
                except Exception:
                    continue

                norm = float(np.linalg.norm(s))
                if norm < 1e-9:
                    continue
                sxy = s / norm
                az = math.degrees(math.atan2(float(sxy[1]), float(sxy[0])))

                v = Vector3Stamped()
                v.header.stamp = self.get_clock().now().to_msg()
                v.header.frame_id = self.frame_id
                v.vector.x = float(sxy[0]); v.vector.y = float(sxy[1]); v.vector.z = 0.0
                self.pub_vec.publish(v)
                self.pub_deg.publish(Float32(data=float(az)))

        except Exception as e:
            self.get_logger().error(f'loop() error: {e}')

    @property
    def frame_id(self):
        return self.frame_id_ if hasattr(self, 'frame_id_') else self.frame_id_
    @frame_id.setter
    def frame_id(self, v):
        self.frame_id_ = v

    def destroy_node(self):
        try:
            if self.conn: self.conn.close()
            self.srv.close()
        finally:
            super().destroy_node()

def main():
    rclpy.init()
    n = GCCPhatTcpDoA()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

