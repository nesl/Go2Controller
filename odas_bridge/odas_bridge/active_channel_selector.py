#!/usr/bin/env python3
import rclpy, json, math, time, numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, Int32, Float32, String, Bool
import webrtcvad
import collections
import pdb

def azimuth_deg(x, y) -> float:
    return math.degrees(math.atan2(y, x))  # [-180, 180)
    
def hann_window(n):
    return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(n)/n)

def spectral_flatness(pxx):
    # geometric / arithmetic mean (avoid zeros)
    pxx = np.maximum(pxx, 1e-12)
    gmean = np.exp(np.mean(np.log(pxx)))
    amean = np.mean(pxx)
    return float(gmean / amean)

def band_energy_ratio(xf, fs, lo=300, hi=3400):
    # xf: power spectrum, one-sided; fs: sample rate
    n = len(xf)
    freqs = np.linspace(0, fs/2, n, endpoint=False)
    m = xf.sum()
    if m <= 0: return 0.0
    mask = (freqs >= lo) & (freqs <= hi)
    return float(xf[mask].sum() / m)


class ActiveChannelSelector(Node):
    def __init__(self):
        super().__init__('active_channel_selector')
        # params
        self.declare_parameter('input_topic', '/mic/audio')
        self.declare_parameter('n_channels', 4)
        self.declare_parameter('hop_samples', 128)
        self.declare_parameter('switch_hysteresis_db', 3.0)
        self.declare_parameter('min_switch_ms', 400)
        self.declare_parameter('az_ttl_sec', 1.5)  # how long to keep last az before considering it stale

        self.declare_parameter('require_voice_active', True)

        self.N    = int(self.get_parameter('n_channels').value)
        self.hop  = int(self.get_parameter('hop_samples').value)
        self.hyst = float(self.get_parameter('switch_hysteresis_db').value)
        self.hold = float(self.get_parameter('min_switch_ms').value) / 1000.0
        self.az_ttl = float(self.get_parameter('az_ttl_sec').value)

        self.require_voice = bool(self.get_parameter('require_voice_active').value)
        self.voice_active = False
        self.sub_vad = self.create_subscription(
            Bool, '/audio/voice_active', lambda m: setattr(self, 'voice_active', bool(m.data)), 10
        )

        # subs & pubs
        self.sub_audio = self.create_subscription(
            UInt8MultiArray, self.get_parameter('input_topic').value, self.on_audio, 200
        )
        self.sub_json  = self.create_subscription(
            String, '/audio/odas/tracked_json', self.on_json, 50
        )

        self.pub_active_bytes = self.create_publisher(UInt8MultiArray, '/mic/audio/active', 50)
        self.pub_active_idx   = self.create_publisher(Int32, '/audio/active_channel', 10)
        self.pub_active_az    = self.create_publisher(Float32, '/audio/active_azimuth_deg', 10)

        # state
        self.current_idx = 0
        self.last_switch = 0.0
        self.latest_az   = None   # degrees
        self.latest_az_ts= 0.0    # seconds epoch


        self.declare_parameter('fs_hz', 16000)
        self.declare_parameter('vad_aggressiveness', 2)  # 0..3
        self.declare_parameter('speech_ratio_min', 0.4) # ERB gate for speech band
        self.declare_parameter('flatness_min', 0.35)      # reject very tonal noise if < this
        self.declare_parameter('voice_hold_ms', 400)     # hangover to avoid choppiness
        

        self.fs = int(self.get_parameter('fs_hz').value)
        self.vad_level = int(self.get_parameter('vad_aggressiveness').value)
        self.vad = webrtcvad.Vad(self.vad_level)
        self.speech_ratio_min = float(self.get_parameter('speech_ratio_min').value)
        self.flatness_min = float(self.get_parameter('flatness_min').value)
        self.voice_hold = float(self.get_parameter('voice_hold_ms').value) / 1000.0

        self.last_voice_ts = 0.0
        # buffers for spectral tests
        self.win_len = 512                      # ~32 ms @16 kHz
        self.win = hann_window(self.win_len).astype(np.float32)
        self.pow_history = collections.deque(maxlen=8)  # ~250 ms
        self.vad_voice_active = False
        self.pub_voice_speech = self.create_publisher(Bool, '/audio/voice_active_speech', 10)

        self.declare_parameter('score_alpha', 0.25)        # EMA smoothing
        self.declare_parameter('speech_thresh', 0.40)      # score threshold
        self.score_alpha    = float(self.get_parameter('score_alpha').value)
        self._score_ema = 0.0    
        self.speech_thresh  = float(self.get_parameter('speech_thresh').value)    
        self._last_gate_on = 0.0
        
        self.vad_buf = bytearray()
        self.declare_parameter('vad_window_ms', 300)
        self.vad_window_ms = int(self.get_parameter('vad_window_ms').value)  # e.g., 300
        
        # Optional: throughput-based fs probe
        self._bytes_in_window = 0
        self._twin_start = time.time()
        self._est_fs = None
        self._fs_probe_secs = 1.0    # size of the window for estimation
        self.byte_reservoir = bytearray()
        
        self.noise_med = None
        self.declare_parameter('speech_lo_hz', 300)
        self.declare_parameter('speech_hi_hz', 3400)
        self.declare_parameter('min_den_lo_hz', 80)      # denominator lower bound
        self.declare_parameter('min_den_hi_hz', 7000)    # denominator upper bound (<= fs/2)
        self.declare_parameter('ratio_min', 0.28)        # gentler default
        self.declare_parameter('flat_min', 0.5)         # gentler default
        self.declare_parameter('excess_min', 0.08)  
        
        self.s_lo      = int(self.get_parameter('speech_lo_hz').value)      # 300
        self.s_hi      = int(self.get_parameter('speech_hi_hz').value)      # 3400
        self.den_lo    = int(self.get_parameter('min_den_lo_hz').value)     # 80
        self.den_hi    = int(self.get_parameter('min_den_hi_hz').value)     # 7000
        self.ratio_min = float(self.get_parameter('ratio_min').value)       # 0.28
        self.flat_min  = float(self.get_parameter('flat_min').value)        # 0.35
        self.excess_min= float(self.get_parameter('excess_min').value)      # 0.08
        self.noise_med = None
        
        self.declare_parameter('snr_min_db', 8.0)         # need >= 6 dB above noise floor
        self.declare_parameter('abs_level_min_db', -50.0) # and absolute level above -55 dBFS
        self.snr_min_db = float(self.get_parameter('snr_min_db').value)
        self.abs_level_min_db = float(self.get_parameter('abs_level_min_db').value)


        # one-time in __init__
        self._prev_pxx = None
        self._flux_hist = collections.deque(maxlen=8)   # ~160 ms @ 20 ms step
        self.declare_parameter('flux_min', 0.015)       # raise if still FP
        self.flux_min = float(self.get_parameter('flux_min').value)
        self._stationary_noise = False
        
        self.get_logger().info(
            f'ActiveChannelSelector: N={self.N}, hysteresis={self.hyst} dB, min_hold={self.hold}s, az_ttl={self.az_ttl}s'
        )

    # --- DoA from ODAS tracked JSON ---
    def on_json(self, msg: String):
        try:
            j = json.loads(msg.data)
            # ODAS variants: "src" or "sources"
            srcs = j.get('src', j.get('sources', []))
            if not isinstance(srcs, list) or len(srcs) == 0:
                # no update; keep last az (sticky)
                return

            # pick the entry with max activity (even if it's 0.0)
            best = max(srcs, key=lambda s: float(s.get('activity', 0.0)))

            if (self.require_voice and not self.voice_active):
                return


            x = float(best.get('x', 0.0))
            y = float(best.get('y', 0.0))
            az = azimuth_deg(x, y)

            # update + publish
            self.latest_az = az
            self.latest_az_ts = time.time()
            self.pub_active_az.publish(Float32(data=float(self.latest_az)))
        except Exception as e:
            # ignore malformed chunks
            pass

    # --- audio: pick active separated channel and publish mono bytes ---
    def on_audio(self, msg: UInt8MultiArray):
        x = np.frombuffer(bytes(msg.data), dtype=np.int16)
        if len(x) < self.N:
            return
        # truncate to full interleaved frames
        rem = len(x) % self.N
        if rem != 0:
            x = x[:len(x) - rem]
        frames = x.reshape(-1, self.N)  # [hop, ch]

        # RMS (dBFS) per channel
        rms = np.sqrt(np.mean(frames.astype(np.float32)**2, axis=0) + 1e-12)
        db  = 20.0 * np.log10(np.maximum(rms / 32768.0, 1e-9))
        loudest = int(np.argmax(db))

        # hysteresis + min-hold to avoid rapid flipping
        now = time.time()
        if loudest != self.current_idx:
            if (db[loudest] - db[self.current_idx]) >= self.hyst and (now - self.last_switch) >= self.hold:
                self.current_idx = loudest
                self.last_switch = now
                self.pub_active_idx.publish(Int32(data=self.current_idx))

        # publish selected mono bytes
        active_bytes = frames[:, self.current_idx].astype(np.int16).tobytes()
        out = UInt8MultiArray(); out.data = list(active_bytes)
        self.pub_active_bytes.publish(out)

        # expire very stale az (optional)
        if self.latest_az is not None and (now - self.latest_az_ts) > self.az_ttl:
            self.latest_az = None  # consumer may treat None as "unknown"
            # (we don't publish anything here; /audio/active_azimuth_deg is event-driven on updates)

        # Append raw bytes to reservoir
        self.byte_reservoir.extend(bytes(msg.data))

        # Optional: estimate fs from throughput (bytes/sec ÷ (2*N))
        self._bytes_in_window += len(msg.data)
        now = time.time()
        if (now - self._twin_start) >= self._fs_probe_secs:
            bytes_per_sec = self._bytes_in_window / max(1e-6, (now - self._twin_start))
            fs_est = bytes_per_sec / (2.0 * max(1, self.N))
            # Accept only plausible rates
            for cand in (8000, 16000, 32000, 48000):
                if abs(fs_est - cand) / cand < 0.12:
                    self._est_fs = cand
                    break
            self._bytes_in_window = 0
            self._twin_start = now
            if self._est_fs and self._est_fs != self.fs:
                self.get_logger().warn(f"Adjusting fs to {self._est_fs} Hz (est from throughput)")
                self.fs = self._est_fs

        # Make sure we have an integer number of FRAME SAMPLES (interleaved)
        bytes_per_sample_all_ch = 2 * self.N  # int16 * N channels
        usable_len = (len(self.byte_reservoir) // bytes_per_sample_all_ch) * bytes_per_sample_all_ch
        if usable_len == 0:
            return

        # Slice off a whole number of interleaved frames and keep the rest in reservoir
        chunk = self.byte_reservoir[:usable_len]
        del self.byte_reservoir[:usable_len]

        # Convert to int16 (explicit LE)
        x = np.frombuffer(chunk, dtype='<i2')  # shape: [samples_all_channels]
        if x.size == 0:
            return

        # Reshape to [T, N] interleaved frames
        frames = x.reshape(-1, self.N)

        vad_ok = False
        speechy = False
        ratio, flat = 0.0, 0.0
        excess_frac = 0.0
        level_ok = False
        if self.voice_active:
        
            # After you pick the active channel index (self.current_idx):
            active_i16 = frames[:, self.current_idx].astype('<i2', copy=False)  # ensure LE
            raw = active_i16.tobytes()

            # Feed rolling buffer
            self.vad_buf.extend(raw)
            max_bytes = int(self.fs * self.vad_window_ms / 1000) * 2
            if len(self.vad_buf) > max_bytes:
                self.vad_buf = self.vad_buf[-max_bytes:]

            # VAD over last ~200 ms
            
            if self.fs in (8000, 16000, 32000, 48000):
                frame_ms = 20
                step = int(self.fs * frame_ms / 1000) * 2  # bytes per 20ms mono frame
                scan_bytes = min(len(self.vad_buf), step * 10)  # 10*20ms = 200ms
                start = len(self.vad_buf) - scan_bytes
                hits = 0
                for i in range(start, len(self.vad_buf) - step + 1, step):
                    if self.vad.is_speech(self.vad_buf[i:i+step], self.fs):
                        hits += 1
                        if hits >= 2: break
                vad_ok = (hits >= 2)   # not just any single hit


            # Need at least win_len samples (= win_len*2 bytes)
            need = self.win_len * 2
            if len(self.vad_buf) >= need:
                # Tail of the rolling mono buffer -> int16 -> float
                mono_tail = np.frombuffer(self.vad_buf[-need:], dtype='<i2').astype(np.float32, copy=False)

                # Hann + pre-emphasis (tilts speech up, reduces low hum)
                seg = mono_tail * self.win
                seg[1:] -= 0.97 * seg[:-1]

                xf = np.fft.rfft(seg, n=self.win_len)
                pxx = (xf.real**2 + xf.imag**2) / self.win_len  # one-sided power

                # median smooth across time (history) for stability
                self.pow_history.append(pxx)
                pxx_med = (np.median(np.stack(self.pow_history, axis=0), axis=0)
                           if len(self.pow_history) >= 3 else pxx)

                # Spectral flux (normalized) between consecutive frames
                if self._prev_pxx is not None:
                    num = np.abs(pxx_med - self._prev_pxx).sum()
                    den = (pxx_med.sum() + self._prev_pxx.sum() + 1e-12)
                    flux = float(num / den)    # 0..1 (steady noise ~0, speech bursts higher)
                    self._flux_hist.append(flux)
                else:
                    flux = 0.0
                self._prev_pxx = pxx_med

                # Stationarity veto: low median flux => steady noise
                flux_med = float(np.median(self._flux_hist)) if self._flux_hist else 0.0
                self._stationary_noise = (flux_med < self.flux_min)

                # ---- band masks ----
                def hz_to_bin(hz, fs, nfft):
                    # rfft bins: 0..nfft/2 inclusive
                    return int(np.clip(round(hz * (nfft/2) / (fs/2)), 0, nfft//2))

                fs = self.fs; nfft = self.win_len
                b_s_lo = hz_to_bin(self.s_lo, fs, nfft)         # e.g., 300 Hz
                b_s_hi = hz_to_bin(self.s_hi, fs, nfft)         # e.g., 3400 Hz
                b_d_lo = hz_to_bin(self.den_lo, fs, nfft)       # e.g., 80 Hz
                b_d_hi = hz_to_bin(min(self.den_hi, fs/2 - 1), fs, nfft)

                # Build masks; exclude DC bin from denominator
                speech_band = np.zeros_like(pxx_med, dtype=bool); speech_band[b_s_lo:b_s_hi+1] = True
                den_band    = np.zeros_like(pxx_med, dtype=bool); den_band[max(1, b_d_lo):b_d_hi+1] = True

                num = float(pxx_med[speech_band].sum())
                den = float(pxx_med[den_band].sum()) + 1e-12
                ratio = num / den

                # pxx_med already computed (one-sided power). Build a smoothed envelope:
                k = 9  # ~ (k/fs) * (nfft/2) smoothing in Hz; k=9 works well at 16 kHz, nfft=512
                ker = np.ones(k, dtype=np.float32) / k
                pxx_env = np.convolve(pxx_med, ker, mode='same')

                # Compute flatness **inside the speech band** on the smoothed envelope
                sb_env = np.maximum(pxx_env[speech_band], 1e-12)
                flat = float(np.exp(np.mean(np.log(sb_env))) / np.mean(sb_env))  # 0..1


                # Noise-adaptive excess energy in speech band
                if (not vad_ok) and (ratio < 0.20):
                    self.noise_med = pxx_med.copy() if getattr(self, 'noise_med', None) is None else (0.98*self.noise_med + 0.02*pxx_med)

                if getattr(self, 'noise_med', None) is not None:
                    noise_sb = self.noise_med[speech_band]
                    excess = (pxx_med[speech_band] - noise_sb); excess[excess < 0.0] = 0.0
                    excess_frac = float(excess.sum()) / (float(pxx_med[speech_band].sum()) + 1e-12)

                ratio_ok  = (ratio  >= self.ratio_min)    # defaults: ratio_min=0.28
                flat_ok   = (flat   >= self.flat_min)     # flat_min=0.35
                excess_ok = (excess_frac >= self.excess_min)  # excess_min=0.08
                speechy = bool(ratio_ok and flat_ok and excess_ok)

            speechy = (ratio >= self.speech_ratio_min) and (flat >= self.flatness_min)
            
            '''
            # 5c) Soft speech score (0..1) + EMA smoothing
            inst_score = 0.6*float(vad_ok) + 0.25*min(1.0, ratio/0.6) + 0.15*min(1.0, flat/0.8)
            self._score_ema = (1.0 - self.score_alpha)*self._score_ema + self.score_alpha*inst_score

            # --- 6) final refined gate: external /audio/voice_active AND (score>thr OR hangover) ---
            # self.voice_active is your external gate (already subscribed elsewhere)
            soft_pass = (self._score_ema >= self.speech_thresh) or ((now - self._last_gate_on) <= self.voice_hold)
            final_gate = bool(soft_pass)
            if final_gate:
                self._last_gate_on = now
            '''
            
            # --- level gate: adaptive noise floor + SNR threshold ---
            # keep a noise floor in dBFS from non-speech segments; require speech to beat it by X dB
            if not hasattr(self, '_noise_floor_db'):
                self._noise_floor_db = -55.0  # start conservative
            if not hasattr(self, '_nf_update_ts'):
                self._nf_update_ts = 0.0

            # instantaneous level of active channel (use your 'db' you already compute per channel)
            lvl_db = float(db[self.current_idx])

            # Update noise floor only when we are clearly NOT in speech
            if not vad_ok and not speechy and not self._stationary_noise:
                # EMA toward current level (slowly), clamp to plausible range
                self._noise_floor_db = max(-90.0, min(0.0, 0.95*self._noise_floor_db + 0.05*lvl_db))

            snr_db = lvl_db - self._noise_floor_db
            level_ok = (snr_db >= self.snr_min_db) and (lvl_db >= self.abs_level_min_db)
            
            
            strict_pass = bool(vad_ok and speechy and level_ok and not self._stationary_noise)  # add 'and level_ok' if you kept the SNR gate

            now = time.time()
            if strict_pass:
                self._last_gate_on = now
            final_gate = strict_pass or ((now - self._last_gate_on) <= self.voice_hold)

        else:
            final_gate = False
        # publish voice_active (used by DoA gate)
        self.pub_voice_speech.publish(Bool(data=final_gate))
        
        
        if False and int(time.time()*10) % 10 == 0:
            self.get_logger().info(
                f"N={self.N} fs={self.fs} vad={vad_ok} score={self._score_ema:.2f} gate={final_gate} "
                f"bytes_reservoir={len(self.byte_reservoir)} vad_buf={len(self.vad_buf)} speechy={speechy} ratio={ratio:.2f} flat={flat:.2f} excess={excess_frac:.2f} level_ok={level_ok}"
            )
    
def main():
    rclpy.init()
    n = ActiveChannelSelector()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
