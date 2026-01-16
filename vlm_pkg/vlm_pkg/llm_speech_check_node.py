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
        self.declare_parameter("model_id", "gpt-5-mini")
        self.declare_parameter("temperature", 0.2)
        self.declare_parameter("max_tokens", 256)
        # --- LLM name translation (canonical <-> display) ---
        self.declare_parameter("robot_name", "robot")
        self.declare_parameter("human_a_name", "Sam")
        self.declare_parameter("human_b_name", "Jacob")

        robot_name  = self.get_parameter("robot_name").get_parameter_value().string_value
        human_a_name = self.get_parameter("human_a_name").get_parameter_value().string_value
        human_b_name = self.get_parameter("human_b_name").get_parameter_value().string_value

        self.agent_id_to_human_name = {
            "robot": robot_name,
            "human_a": human_a_name,
            "human_b": human_b_name,
        }
        self.human_name_to_agent_id = {v: k for k, v in self.agent_id_to_human_name.items()}



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

    def _swap_str(self, s: str, mapping: dict) -> str:
        if not isinstance(s, str) or not mapping:
            return s
        out = s
        # Longest keys first prevents partial collisions
        for a in sorted(mapping.keys(), key=len, reverse=True):
            out = out.replace(a, mapping[a])
        return out

    def _swap_json(self, obj: Any, mapping: dict) -> Any:
        """
        Recursively replace BOTH dict keys and values (and strings in lists).
        Used ONLY at the LLM boundary.
        """
        if isinstance(obj, dict):
            return {self._swap_str(k, mapping): self._swap_json(v, mapping) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._swap_json(x, mapping) for x in obj]
        if isinstance(obj, str):
            return self._swap_str(obj, mapping)
        return obj


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

    def _build_prompt(
        self,
        prompt: str,
        output_schema: str,
        text: str,
        speaker_id: str,
        tag: str = "",
        history: list[Dict[str, Any]] | None = None,
    ) -> str:

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

        # Add recent conversation history if provided
        history = history or []
        if history:
            header += "Recent dialogue (oldest first):\n"
            for h in history:
                spk = h.get("speaker_id") or "unknown"
                txt = (h.get("text") or "").strip()
                header += f"- [{spk}] {txt}\n"
            header += "\n"


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
            f"Current user utterance (spoken by {speaker_id}):\n{text}\n\n"
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
            self._publish_resp(
                req_id,
                success=False,
                raw="empty text",
                json_text="",
                lat_ms=0.0,
                tag=obj.get("tag", ""),
                original_req=obj,              # ← pass it through
            )
            return

        prompt = obj.get("prompt", "")
        output_schema = obj.get("output_schema", "")
        tag = obj.get("tag", "")  # just for logging/echo
        history = obj.get("history") or []

        # --- LLM boundary translation (canonical -> display) ---
        fwd = dict(getattr(self, "agent_id_to_human_name", {}) or {})
        rev = dict(getattr(self, "human_name_to_agent_id", {}) or {})

        # Translate current utterance text
        text_for_llm = self._swap_str(text, fwd)
        speaker_id = obj.get("speaker_id") or obj.get("speaker") or "unknown"
        speaker_for_llm = self._swap_str(speaker_id, fwd)
        # Translate prompt + schema (they may contain agent ids)
        prompt_for_llm = self._swap_str(prompt, fwd)
        output_schema_for_llm = self._swap_str(output_schema, fwd)

        # Translate history objects (speaker_id fields, and any embedded text)
        history_for_llm = self._swap_json(history, fwd) if isinstance(history, list) else []


        try:
            p = self._build_prompt(
                prompt_for_llm,
                output_schema_for_llm,
                text_for_llm,
                speaker_for_llm,  
                tag,
                history=history_for_llm,
            )

            raw, lat_ms = self._call_llm(p)

            # --- LLM output translation (display -> canonical) ---
            # We will produce:
            #   raw_out: canonicalized raw text (best-effort)
            #   json_text: canonicalized STRICT JSON if parseable, else ""
            json_text = ""
            raw_out = raw

            def _canonicalize_json_obj(obj_display: Any) -> Any:
                # translate display names back to canonical ids
                return self._swap_json(obj_display, rev)

            # Try parsing the raw output as JSON first
            parsed_obj = None
            try:
                parsed_obj = json.loads(raw)
            except Exception:
                stripped = raw.strip().strip("`")
                if stripped.lower().startswith("json"):
                    stripped = stripped[4:].lstrip()
                try:
                    parsed_obj = json.loads(stripped)
                except Exception:
                    parsed_obj = None

            if parsed_obj is not None:
                canon_obj = _canonicalize_json_obj(parsed_obj)
                json_text = json.dumps(canon_obj, separators=(",", ":"), ensure_ascii=False)
                # Make raw_out canonical too (since it *is* JSON)
                raw_out = json_text
            else:
                # Not JSON: still canonicalize any display names in raw text
                raw_out = self._swap_str(raw, rev)


            sanitized_req = dict(obj)
            sanitized_req.pop("prompt", None)
            sanitized_req.pop("output_schema", None)
            sanitized_req.pop("history", None)

            self._publish_resp(
                req_id,
                True,
                raw_out,
                json_text,
                lat_ms,
                tag,
                original_req=sanitized_req,              # ← HERE
            )

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
            self._publish_resp(
                req_id,
                False,
                str(e),
                "",
                0.0,
                tag,
                original_req=obj,              # optional but consistent
            )

    def _publish_resp(self, req_id, success, raw, json_text, lat_ms, tag="", original_req: dict | None = None):
        out = {
            "id": req_id,
            "success": bool(success),
            "raw_text": raw,
            "json_text": json_text,
            "model_id": self.model_id,
            "lat_ms": float(lat_ms),
            "tag": tag,
        }
        # include a reference to the triggering message
        if original_req is not None:
            out["request"] = original_req

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

