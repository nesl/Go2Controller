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

from PIL import Image
import io
import numpy as np

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

        out_ids = self.model.generate(**inputs, **gen_kwargs)

        if hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        else:
            tok = self.tokenizer
            if tok is None:
                tok = AutoTokenizer.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
            text = tok.decode(out_ids[0], skip_special_tokens=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            print(f"[VLM] peak VRAM: {human_bytes(peak)}")

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

        model_id = self.get_parameter("model_id").get_parameter_value().string_value
        int4 = self.get_parameter("int4").get_parameter_value().bool_value
        bf16 = self.get_parameter("bf16").get_parameter_value().bool_value

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

        self.get_logger().info(f"Loading VLM model: {model_id} (int4={int4}, bf16={bf16}) …")
        try:
            self.vlm = VLM(model_id, int4=int4, bf16=bf16)
            self.get_logger().info("VLM ready.")
        except Exception as e:
            self.get_logger().error(f"Failed to load VLM: {e}")
            raise

    def on_image(self, msg: RosImage):
        try:
            self.latest_pil = pil_from_ros_image(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")

    def on_prompt(self, msg: RosString):
        self.current_prompt = msg.data
        self.get_logger().info(f"Prompt updated: {self.current_prompt!r}")

    def on_run(self, req, res):
        if self.latest_pil is None:
            res.success = False
            res.message = "No image received yet on /camera/image_raw."
            return res

        prompt = self.current_prompt
        max_new = self.get_parameter("max_new_tokens").get_parameter_value().integer_value
        temp = self.get_parameter("temperature").get_parameter_value().double_value

        self.get_logger().info("Running VLM inference…")
        try:
            answer = self.vlm.infer(self.latest_pil, prompt, max_new_tokens=max_new, temperature=temp)
            # publish full text
            self.ans_pub.publish(RosString(data=answer))
            # truncate in service response
            trunc = (answer[:180] + "…") if len(answer) > 180 else answer
            res.success = True
            res.message = f"OK: {trunc}"
        except Exception as e:
            self.get_logger().error(f"VLM error: {e}")
            res.success = False
            res.message = f"Error: {e}"
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
