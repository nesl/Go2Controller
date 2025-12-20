#!/usr/bin/env python3
"""
Qwen VLM ROS2 Server
- Subscribes: /camera/image_raw (sensor_msgs/Image)
- Subscribes: /vlm/prompt (std_msgs/String)  -> sets the current prompt
- Service:   /vlm/run (std_srvs/Trigger)     -> runs VLM on the latest image + current prompt
- Publishes: /vlm/answer (std_msgs/String)   -> full textual answer

Params (ros2 params):
- model_id (str): default "Qwen/Qwen2.5-VL-7B-Instruct"
- int4 (bool):    default True
- bf16 (bool):    default False (set True if your GPU supports bfloat16)
- max_new_tokens (int): default 256
- temperature (float): default 0.1

Notes:
- Requires: torch, torchvision, transformers>=4.46, accelerate, pillow, bitsandbytes (optional for --int4)
- If you don't see images arriving: check camera topic name and QoS compatibility.
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String as RosString
from std_srvs.srv import Trigger

from std_srvs.srv import SetBool
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
import time, json

from PIL import Image
import io
import numpy as np

import re

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# -----------------------------
# Helpers
# -----------------------------
def pil_from_ros_image(msg: RosImage) -> Image.Image:
    """Convert ROS Image (RGB/Mono/BGR) to PIL Image."""
    if not _HAS_CV2:
        # fallback without cv2 (assume RGB8)
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        arr = arr.reshape((msg.height, msg.width, -1))
        # Best effort: assume encoding is RGB8
        if msg.encoding.lower() in ["bgr8", "bgr8; compressed_bgr8"]:
            arr = arr[:, :, ::-1]  # BGR to RGB
        return Image.fromarray(arr, mode="RGB")

    # With OpenCV we can handle more encodings
    dtype = np.uint8
    if "16" in msg.encoding.lower():
        dtype = np.uint16
    arr = np.frombuffer(msg.data, dtype=dtype).copy()
    # channels inference
    if "mono" in msg.encoding.lower():
        arr = arr.reshape((msg.height, msg.width))
        img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        # assume color
        channels = 3 if "8" in msg.encoding or "bgr" in msg.encoding.lower() or "rgb" in msg.encoding.lower() else 3
        arr = arr.reshape((msg.height, msg.width, channels))
        if "bgr" in msg.encoding.lower():
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif "rgb" in msg.encoding.lower():
            pass
        else:
            # best effort
            pass
        img = arr
    return Image.fromarray(img, mode="RGB")

def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def extract_json_from_answer(answer: str, logger=None) -> str:
    """
    Try to extract a strict JSON object/array from a VLM answer.

    - Prefer a ```json ... ``` fenced block.
    - Fallback to the first {...} or [...] blob.
    - Validate with json.loads, then re-dump to normalized JSON.
    - Return "" if nothing valid is found.
    """
    if not answer:
        return ""

    cand = None

    # 1) Prefer ```json fenced block
    m = re.search(
        r"```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```",
        answer,
        flags=re.IGNORECASE,
    )
    if m:
        cand = m.group(1).strip()
    else:
        # 2) Fallback: first {...} or [...] chunk
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", answer)
        if m:
            cand = m.group(1).strip()

    if not cand:
        return ""

    try:
        obj = json.loads(cand)
    except Exception as e:
        if logger is not None:
            logger.warn(f"[vlm] JSON extraction failed: {e} from candidate: {cand[:160]!r}")
        return ""

    # Normalize to compact JSON string
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        if logger is not None:
            logger.warn(f"[vlm] JSON re-dump failed: {e}")
        return ""


# -----------------------------
# VLM Loader (Qwen-friendly)
# -----------------------------
class VLM:
    def __init__(self, model_id: str, int4: bool, bf16: bool):
        self.model_id = model_id
        self.device_map = "cpu" if not torch.cuda.is_available() else "auto"
        self.dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float16

        # Quantization config
        bnb_cfg = None
        if int4:
            if not _HAS_BNB:
                raise RuntimeError("int4=True but bitsandbytes not installed. pip install bitsandbytes or set int4:=false")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.dtype==torch.bfloat16 else torch.float16,
            )
            # dtype handled by bitsandbytes kernels
            dtype = None
        else:
            dtype = self.dtype

        cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        # Heuristic: many modern VLMs are vision-to-seq
        visionish = getattr(cfg, "model_type", "") in {
            "qwen2_5_vl", "internvl2", "fuyu", "mllama", "llava", "git", "phi4multimodal"
        } or "vision" in type(cfg).__name__.lower() or "vl" in type(cfg).__name__.lower()

        try:
            if visionish:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device_map,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device_map,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )
        except Exception:
            # Fallback the other way
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device_map,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )
            except Exception:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device_map,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                )

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        # Log device map once
        devmap = getattr(self.model, "hf_device_map", None)
        print("[VLM] loaded", self.model_id, "device_map=", devmap)

    @torch.inference_mode()
    def infer(self, pil_img: Image.Image, prompt: str, max_new_tokens=256, temperature=0.1, use_beam=False) -> str:
        chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = prompt
        if hasattr(self.processor, "apply_chat_template"):
            try:
                text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        inputs = self.processor(images=pil_img, text=text, return_tensors="pt")
        # Move to model device (single-GPU recommended)
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=(temperature > 0.0),
        )
        if use_beam:
            gen_kwargs.update(num_beams=3, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        input_len = inputs["input_ids"].shape[-1]

        out_ids = self.model.generate(**inputs, **gen_kwargs)

        # Slice off the prompt tokens
        gen_ids = out_ids[0][input_len:]

        if hasattr(self.processor, "decode"):
            text = self.processor.decode(gen_ids, skip_special_tokens=True)
        elif hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
        else:
            tok = self.tokenizer
            text = tok.decode(gen_ids, skip_special_tokens=True)

        return text.strip()


# -----------------------------
# ROS2 Node
# -----------------------------
class QwenVLMServer(Node):
    def __init__(self):
        super().__init__("qwen_vlm_server")

        # Parameters
        self.declare_parameter("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.declare_parameter("int4", False)
        self.declare_parameter("bf16", True)
        self.declare_parameter("max_new_tokens", 256)
        self.declare_parameter("temperature", 0.1)

        # Perf + enable
        self.declare_parameter("perf_topic", "/vlm/perf")
        self.declare_parameter("enabled", True)

        model_id = self.get_parameter("model_id").get_parameter_value().string_value
        int4 = self.get_parameter("int4").get_parameter_value().bool_value
        bf16 = self.get_parameter("bf16").get_parameter_value().bool_value

        self.model_id = model_id
        self.int4 = int4
        self.bf16 = bf16
        self.enabled = self.get_parameter("enabled").get_parameter_value().bool_value
        self.perf_topic = self.get_parameter("perf_topic").get_parameter_value().string_value

        # Latest frame + prompt
        self.latest_pil = None
        self.current_prompt = "Describe the scene."

        # QoS: often camera uses BEST_EFFORT; make subscriber tolerant
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(RosImage, "/camera/image_raw", self.on_image, camera_qos)
        self.create_subscription(RosString, "/vlm/prompt", self.on_prompt, 10)

        self.ans_pub = self.create_publisher(RosString, "/vlm/answer", 10)
        self.srv = self.create_service(Trigger, "/vlm/run", self.on_run)

        self.perf_pub = self.create_publisher(RosString, self.perf_topic, 10)
        self.enable_srv = self.create_service(SetBool, "/vlm/enable", self.on_enable)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        # NEW: generic request channel, symmetric with llm_speech_check
        self.create_subscription(RosString, "/vlm/req", self.on_req, 10)



        self._load_vlm()

    def _load_vlm(self):
        self.get_logger().info(
            f"Loading VLM model: {self.model_id} (int4={self.int4}, bf16={self.bf16}) …"
        )
        try:
            self.vlm = VLM(self.model_id, int4=self.int4, bf16=self.bf16)
            self.get_logger().info("VLM ready.")
        except Exception as e:
            self.get_logger().error(f"Failed to load VLM: {e}")
            raise

    def on_req(self, msg: RosString):
        """
        Handle generic VLM requests coming as JSON on /vlm/req.

        Expected payload:
          {
            "id": <int or str>,
            "prompt": <str>,
            "output_schema": <str>,   # not enforced here, just forwarded
            "tag": <str>,
            "mode": <str>             # optional; ignored here
          }
        """
        raw = msg.data or ""
        self.get_logger().info(f"[vlm] /vlm/req raw payload (len={len(raw)}): {raw[:200]!r}")

        try:
            obj = json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f"[vlm] bad JSON on /vlm/req: {e}")
            return

        client = obj.get("client") or ""
        req_id = obj.get("id")
        prompt = obj.get("prompt") or "Describe the scene."
        tag = obj.get("tag") or "default_vlm"
        mode = obj.get("mode") or "generic"

        self.get_logger().info(
            f"[vlm] parsed /vlm/req id={req_id!r} tag={tag!r} mode={mode!r} "
            f"prompt_len={len(prompt)} prompt_preview={prompt[:120]!r}"
        )

        ok, answer, dt_ms, envelope = self._run_vlm_once(
            prompt=prompt,
            tag=tag,
            phase="run_trigger",   # distinguish from service calls
            req_id=req_id,
            client=client
        )

        if ok:
            self.get_logger().info(
                f"[vlm] req id={req_id} OK tag={tag!r} lat={dt_ms} ms "
                f"answer_len={len(answer)} answer_preview={answer[:120]!r}"
            )
        else:
            self.get_logger().warn(
                f"[vlm] req id={req_id} FAILED tag={tag!r} lat={dt_ms} ms"
            )



    def on_enable(self, req, res):
        self.enabled = bool(req.data)
        res.success = True
        res.message = f"VLM enabled={self.enabled}"
        self.get_logger().info(res.message)
        return res

    def _publish_perf(self, *, lat_ms: int, ok: bool, phase: str,
                      prompt_len: int, answer_len: int):
        payload = {
            "node": "vlm",
            "task": "vlm_inference",
            "model": self.model_id,
            "lat_ms": int(lat_ms),
            "ok": bool(ok),
            "phase": phase,           # e.g., "run_service"
            "prompt_len": int(prompt_len or 0),
            "answer_len": int(answer_len or 0),
            "ts": time.time(),
        }
        try:
            self.perf_pub.publish(RosString(data=json.dumps(payload)))
        except Exception as e:
            self.get_logger().warn(f"[vlm] perf publish failed: {e}")

    def _on_set_parameters(self, params):
        reload_needed = False
        for p in params:
            if p.name == "model_id" and p.type_ == Parameter.Type.STRING:
                self.model_id = p.value
                reload_needed = True
            elif p.name == "int4" and p.type_ == Parameter.Type.BOOL:
                self.int4 = p.value
                reload_needed = True
            elif p.name == "bf16" and p.type_ == Parameter.Type.BOOL:
                self.bf16 = p.value
                reload_needed = True

        if reload_needed:
            try:
                self._load_vlm()
                self.get_logger().info(
                    f"[vlm] reloaded model_id={self.model_id} int4={self.int4} bf16={self.bf16}"
                )
            except Exception as e:
                self.get_logger().error(f"[vlm] failed to reload model: {e}")
                return SetParametersResult(successful=False, reason=str(e))

        return SetParametersResult(successful=True, reason="ok")


    def on_image(self, msg: RosImage):
        try:
            self.latest_pil = pil_from_ros_image(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")

    def on_prompt(self, msg: RosString):
        self.current_prompt = msg.data
        self.get_logger().info(f"Prompt updated: {self.current_prompt!r}")

    def _run_vlm_once(self, *, prompt: str, tag: str = "", phase: str, req_id=None, client: str = ""):

        """
        Shared helper: runs VLM on latest image and publishes JSON envelope to /vlm/answer.

        Returns (ok: bool, answer_text: str, lat_ms: int, envelope: dict).
        """
        tag = tag or ""
        if not self.enabled:
            self.get_logger().warn(
                f"[vlm] request while disabled (phase={phase}, tag={tag!r}, id={req_id!r})"
            )
            return False, "", 0, {
                "id": req_id,
                "client": client,
                "success": False,
                "raw_text": "",
                "json_text": "",
                "model_id": self.model_id,
                "lat_ms": 0,
                "tag": tag,
                "ts": time.time(),
            }

        if self.latest_pil is None:
            self.get_logger().warn(
                f"[vlm] request but no image yet (phase={phase}, tag={tag!r}, id={req_id!r})"
            )
            return False, "", 0, {
                "id": req_id,
                "client": client,
                "success": False,
                "raw_text": "",
                "json_text": "",
                "model_id": self.model_id,
                "lat_ms": 0,
                "tag": tag,
                "ts": time.time(),
            }

        max_new = self.get_parameter("max_new_tokens").get_parameter_value().integer_value
        temp = self.get_parameter("temperature").get_parameter_value().double_value

        # Extra debug about prompt + image
        try:
            w, h = self.latest_pil.size
            img_info = f"{w}x{h}"
        except Exception:
            img_info = "unknown"

        self.get_logger().info(
            f"[vlm] running inference "
            f"(phase={phase}, tag={tag!r}, id={req_id!r}, "
            f"model_id={self.model_id!r}, img={img_info}, "
            f"prompt_len={len(prompt)}, prompt_preview={prompt[:160]!r}, "
            f"max_new_tokens={max_new}, temperature={temp})"
        )

        t0 = time.time()
        ok = False
        answer = ""
        dt_ms = 0
        try:
            answer = self.vlm.infer(
                self.latest_pil,
                prompt,
                max_new_tokens=max_new,
                temperature=temp,
            )
            ok = True
        except Exception as e:
            self.get_logger().error(f"[vlm] error during inference: {e}")
        finally:
            dt_ms = int((time.time() - t0) * 1000)
            self._publish_perf(
                lat_ms=dt_ms,
                ok=ok,
                phase=phase,
                prompt_len=len(prompt or ""),
                answer_len=len(answer or ""),
            )

        self.get_logger().info(
            f"[vlm] inference done (phase={phase}, tag={tag!r}, id={req_id!r}, "
            f"ok={ok}, lat={dt_ms} ms, answer_len={len(answer)}, "
            f"answer_preview={answer[:160]!r})"
        )

        # Try to extract strict JSON (if the prompt asked for it)
        json_blob = ""
        if ok and answer:
            json_blob = extract_json_from_answer(answer, logger=self.get_logger())
            if json_blob:
                self.get_logger().info(
                    f"[vlm] extracted JSON blob len={len(json_blob)} preview={json_blob[:160]!r}"
                )
            else:
                # Optional: only log if answer *looks* like it should contain JSON
                if "```json" in answer or "object_id" in answer:
                    self.get_logger().warn(
                        f"[vlm] no valid JSON extracted from answer (len={len(answer)})"
                    )

        # Build JSON envelope expected by EventLayer/SkillsAgent
        envelope = {
            "id": req_id,
            "client": client,
            "success": bool(ok),
            "raw_text": answer or "",
            "json_text": json_blob,        # ← NOW FILLED WHEN POSSIBLE
            "model_id": self.model_id,
            "lat_ms": dt_ms,
            "tag": tag or "",
            "ts": time.time(),
        }


        try:
            self.ans_pub.publish(RosString(data=json.dumps(envelope)))
        except Exception as e:
            self.get_logger().warn(f"[vlm] answer publish failed: {e}")

        return ok, answer, dt_ms, envelope



    def on_run(self, req, res):
        """
        Legacy service interface for manual testing.

        Uses self.current_prompt (set via /vlm/prompt) and also publishes a
        JSON envelope on /vlm/answer, just like /vlm/req does.
        """
        prompt = self.current_prompt or "Describe the scene."
        tag = "service_run"

        ok, answer, dt_ms, envelope = self._run_vlm_once(
            prompt=prompt,
            tag=tag,
            phase="run_service",
            req_id=None,   # no correlation id for service by default
        )

        if ok:
            trunc = (answer[:180] + "…") if len(answer) > 180 else answer
            res.success = True
            res.message = f"OK: {trunc}"
        else:
            res.success = False
            # envelope already published; describe failure succinctly
            res.message = "Error: VLM inference failed or no image available."

        return res


def main():
    rclpy.init()
    node = QwenVLMServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
