#!/usr/bin/env python3
import json, time
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
from std_srvs.srv import SetBool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType

from openai import OpenAI


class LlmSpeechCheckNode(Node):
    """
    Pub/sub-based LLM speech checker.

    In:  /llm/speech_check_req  (std_msgs/String, JSON)
    Out: /llm/speech_check_resp (std_msgs/String, JSON)
    Out: /llm/speech_check_perf (std_msgs/String, JSON)

    Services:
      /llm_speech_check/enable         (std_srvs/SetBool)
      /llm_speech_check/set_parameters (rcl_interfaces/SetParameters)
    """

    def __init__(self):
        super().__init__("llm_speech_check")

        # Parameters (runtime-changeable via set_parameters)
        self.declare_parameter("model_id", "gpt-5.1-mini")
        self.declare_parameter("temperature", 0.2)
        self.declare_parameter("max_tokens", 256)

        self.model_id = self.get_parameter("model_id").get_parameter_value().string_value
        self.temperature = float(self.get_parameter("temperature").value)
        self.max_tokens = int(self.get_parameter("max_tokens").value)
        self.enabled = True

        # OpenAI client (expects OPENAI_API_KEY)
        self.client = OpenAI()

        # Pub/Sub
        self.req_sub = self.create_subscription(
            StringMsg,
            "/llm/speech_check_req",
            self._cb_req,
            10,
        )
        self.resp_pub = self.create_publisher(StringMsg, "/llm/speech_check_resp", 10)
        self.perf_pub = self.create_publisher(StringMsg, "/llm/speech_check_perf", 10)

        # Services
        self.enable_srv = self.create_service(SetBool, "/llm_speech_check/enable", self._srv_enable)
        self.set_params_srv = self.create_service(SetParameters, "/llm_speech_check/set_parameters", self._srv_set_parameters)

        self.get_logger().info("llm_speech_check node ready (pub/sub mode).")

    # ────────────────────────── Services ──────────────────────────

    def _srv_enable(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = f"llm_speech_check {'ENABLED' if self.enabled else 'DISABLED'}"
        self.get_logger().info(resp.message)
        return resp

    def _srv_set_parameters(self, req, resp):
        """
        Simple manual SetParameters handler; updates model_id/temperature/max_tokens
        when seen in incoming parameters.
        """
        for p in req.parameters:
            name = p.name
            if name == "model_id" and p.value.type == ParameterType.PARAMETER_STRING:
                self.model_id = p.value.string_value
                self.get_logger().info(f"llm_speech_check model_id -> {self.model_id}")
            elif name == "temperature" and p.value.type in (
                ParameterType.PARAMETER_DOUBLE,
                ParameterType.PARAMETER_INTEGER,
            ):
                self.temperature = float(p.value.double_value or p.value.integer_value)
                self.get_logger().info(f"llm_speech_check temperature -> {self.temperature}")
            elif name == "max_tokens" and p.value.type in (
                ParameterType.PARAMETER_INTEGER,
                ParameterType.PARAMETER_DOUBLE,
            ):
                self.max_tokens = int(p.value.integer_value or p.value.double_value)
                self.get_logger().info(f"llm_speech_check max_tokens -> {self.max_tokens}")

        # echo back what we accepted
        resp.results = []
        return resp

    # ────────────────────────── Helpers ───────────────────────────

    def _build_prompt(self, prompt: str, output_schema: str, text: str, tag: str = "") -> str:
        """
        Generic wrapper:
          - caller provides prompt + (optional) JSON schema
          - we always force STRICT JSON, no prose
        """
        user_prompt = (prompt or "").strip()
        schema_str = (output_schema or "").strip()

        header = "You are a JSON-only tool. Respond with STRICT JSON, no extra text.\n"

        if tag:
            header += f"Task tag: {tag}\n"

        # If caller gave no prompt, provide a very generic one
        if not user_prompt:
            user_prompt = (
                "Analyze the human utterance and return a JSON object summarizing "
                "its intent, entities, and any safety concerns."
            )

        if schema_str:
            schema_block = (
                "The JSON MUST conform to this schema (or a compatible subset):\n"
                f"{schema_str}\n"
            )
        else:
            # ultra-generic fallback schema
            schema_block = (
                "Return a JSON object with at least these fields:\n"
                "{\n"
                '  "intent": string,\n'
                '  "confidence": number,\n'
                '  "notes": string\n'
                "}\n"
            )

        return (
            f"{header}\n"
            f"User utterance:\n{text}\n\n"
            f"Instructions:\n{user_prompt}\n\n"
            f"{schema_block}"
        )


    def _call_llm(self, prompt: str):
    
        self.get_logger().info(
            f"[llm_speech_check] calling LLM model={self.model_id!r} "
            f"prompt_len={len(prompt)} prompt_preview={prompt!r}"
        )
    
        start = time.time()
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON-only tool. Answer with STRICT JSON, no extra text."
                },
                {"role": "user", "content": prompt},
            ],
        )
        lat_ms = (time.time() - start) * 1000.0
        content = resp.choices[0].message.content or ""
        self.get_logger().info(
            f"[llm_speech_check] LLM reply lat={lat_ms:.1f} ms "
            f"content_len={len(content)} content_preview={content!r}"
        )
        
        return content, lat_ms

    # ────────────────────────── Request handler ───────────────────

    def _cb_req(self, msg: StringMsg):
        if not self.enabled:
            return

        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"speech_check_req bad JSON: {e}")
            return

        req_id = obj.get("id")
        if req_id is None:
            self.get_logger().warn("speech_check_req missing 'id'")
            return

        text = (obj.get("text") or "").strip()
        if not text:
            self._publish_resp(req_id, success=False, raw="empty text", json_text="", lat_ms=0.0, tag=obj.get("tag", ""))
            return

        prompt = obj.get("prompt", "")
        output_schema = obj.get("output_schema", "")
        tag = obj.get("tag", "")  # just for logging/echo

        try:
            p = self._build_prompt(prompt, output_schema, text, tag)
            raw, lat_ms = self._call_llm(p)

            json_text = ""
            # same best-effort JSON normalization as before...
            try:
                parsed = json.loads(raw)
                json_text = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
            except Exception:
                stripped = raw.strip().strip("`")
                if stripped.lower().startswith("json"):
                    stripped = stripped[4:].lstrip()
                try:
                    parsed2 = json.loads(stripped)
                    json_text = json.dumps(parsed2, separators=(",", ":"), ensure_ascii=False)
                except Exception:
                    json_text = ""

            self._publish_resp(req_id, True, raw, json_text, lat_ms, tag)

            perf_obj = {
                "model": self.model_id,
                "lat_ms": float(lat_ms),
                "ts": time.time(),
                "ok": True,
                "tag": tag,
            }
            self.perf_pub.publish(StringMsg(data=json.dumps(perf_obj, ensure_ascii=False)))

        except Exception as e:
            self.get_logger().error(f"llm_speech_check error: {e}")
            self._publish_resp(req_id, False, str(e), "", 0.0, tag)

    def _publish_resp(self, req_id, success, raw, json_text, lat_ms, tag=""):
        out = {
            "id": req_id,
            "success": bool(success),
            "raw_text": raw,
            "json_text": json_text,
            "model_id": self.model_id,
            "lat_ms": float(lat_ms),
            "tag": tag,
        }
        self.resp_pub.publish(StringMsg(data=json.dumps(out, ensure_ascii=False)))



def main():
    rclpy.init()
    node = LlmSpeechCheckNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

