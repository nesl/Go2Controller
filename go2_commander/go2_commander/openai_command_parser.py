#!/usr/bin/env python3
import os, re, json, rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI
from rclpy.parameter import Parameter
from jsonschema import validate, ValidationError

# ---------- Command schema ----------
ROBOT_SCHEMA = {
  "name": "RobotCommand",
  "schema": {
    "type":"object","required":["intent"],
    "properties":{
      "intent":{"type":"string","enum":["sense_area","query_results","navigate","handoff","scan_all"]},
      "params":{"type":"object",
        "properties":{
          "ref":{"type":"string","enum":["HERE","THAT","SELF","GLOBAL"]},
          "class_list":{"type":"array","items":{"type":"string"}},
          "radius_m":{"type":"number","minimum":0.3,"maximum":5.0},
          "goal":{"type":"object","properties":{
            "frame":{"type":"string"},"x":{"type":"number"},"y":{"type":"number"},"yaw":{"type":"number"}}},
          "who":{"type":"string","enum":["SPEAKER","POINTED"]},
          "speed_scale":{"type":"number","minimum":0.2,"maximum":1.5}
        },
        "additionalProperties": False
      }
    },
    "additionalProperties": False
  },
  "strict": True
}

SYSTEM_PARSE = (
  "You translate human commands for a mobile robot into STRICT JSON that matches the given schema. "
  "Intents: sense_area, query_results, navigate, handoff, scan_all. "
  "Deictic mapping: HERE=this/this area/come here; THAT=that area/there; "
  "SELF=where you are; GLOBAL=whole area. Choose sensible defaults. "
  "Output ONLY JSON—no prose."
)

FEWSHOTS = [
  ("come sense this area for objects",
   {"intent":"sense_area","params":{"ref":"HERE","class_list":["object"],"radius_m":1.5}}),
  ("what did you find so far", {"intent":"query_results"}),
  ("come here", {"intent":"navigate","params":{"ref":"HERE"}}),
  ("let me put this in your basket", {"intent":"handoff","params":{"who":"SPEAKER"}}),
  ("scan the whole area and report back", {"intent":"scan_all","params":{"ref":"GLOBAL"}}),
  ("go there", {"intent":"navigate","params":{"ref":"THAT"}}),
  ("sense where you are", {"intent":"sense_area","params":{"ref":"SELF","radius_m":1.5}})
]

SYSTEM_SUMMARY = "Write one short sentence (<=25 words) explaining what the robot did and what it found. Be factual and concise."

class OpenAICommandParser(Node):
    def __init__(self):
        super().__init__("openai_command_parser")
        self.declare_parameter("model", "gpt-5-mini")
        self.declare_parameter("trigger_words", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("trigger_word", "bob")
        
        self.client = OpenAI()
        self.model = self.get_parameter("model").get_parameter_value().string_value
        
        
        # Read preferred array first
        tw_param = self.get_parameters_by_prefix("")["trigger_words"]
        trigger_words = []
        if tw_param and tw_param.type_ == rclpy.Parameter.Type.STRING_ARRAY and tw_param.value:
            trigger_words = [w.lower() for w in tw_param.value]
        else:
            # Fall back to single string
            single = self.get_parameter("trigger_word").get_parameter_value().string_value
            trigger_words = [single.lower()] if single else ["bob"]
        
        self._name_regex = re.compile(r"\b(" + "|".join(map(re.escape, trigger_words)) + r")\b", re.IGNORECASE)

        # ROS I/O
        self.sub_stt = self.create_subscription(String, "/llm/request", self.on_text, 20)
        self.pub_cmd = self.create_publisher(String, "/robot/command_json", 10)
        self.sub_status = self.create_subscription(String, "/task/status_struct", self.on_status_struct, 10)
        self.pub_tts = self.create_publisher(String, "/tts/say", 10)

        self.get_logger().info(f"Ready. Trigger words: {trigger_words}")

    # --- helper
    def _chat_json(self, messages, schema, temperature=0.1, max_tokens=150, retries=1):
        for _ in range(retries + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},  # ask for JSON
            )
            content = resp.choices[0].message.content
            try:
                obj = json.loads(content)
                validate(instance=obj, schema=schema["schema"])  # your ROBOT_SCHEMA["schema"]
                return obj
            except (json.JSONDecodeError, ValidationError):
                # tighten instruction and retry once
                messages = messages + [{"role": "system", "content": "Return ONLY valid JSON per the schema. No prose."}]
        # fallback if still bad
        return {"intent": "query_results"}


    def on_text(self, msg: String):
        utter = msg.data.strip()
        if not utter:
            return

        # ---- Gate by name ----
        if not self._name_regex.search(utter):
            # You can log softly or ignore
            self.get_logger().debug(f"Ignored (no trigger): {utter}")
            return

        # Remove the name from the utterance before parsing
        cleaned = self._name_regex.sub("", utter).strip()

        try:
            messages = [{"role":"system","content":SYSTEM_PARSE}]
            for u, a in FEWSHOTS:
                messages += [{"role":"user","content":u},
                 {"role":"assistant","content":json.dumps(a)}]
            messages.append({"role":"user","content":cleaned})

            
            obj = self._chat_json(messages, ROBOT_SCHEMA, temperature=0.1, max_tokens=150, retries=1)
            self.pub_cmd.publish(String(data=json.dumps(obj, ensure_ascii=False)))
            self.get_logger().info(f"Parsed command: {obj}")
        except Exception as e:
            self.get_logger().warn(f"OpenAI parse error: {e}")
            fallback = {"intent":"query_results"}
            self.pub_cmd.publish(String(data=json.dumps(fallback)))

    def on_status_struct(self, msg: String):
        try:
            status = json.loads(msg.data)
        except Exception:
            return
        try:
            messages = [
                {"role":"system","content":SYSTEM_SUMMARY},
                {"role":"user","content":json.dumps(status, ensure_ascii=False)}
            ]
            resp = self.client.responses.create(
                model=self.model,
                messages=messages,
                max_output_tokens=40,
                temperature=0.2
            )
            text = resp.output_text
            if text:
                self.pub_tts.publish(String(data=text))
        except Exception as e:
            self.get_logger().warn(f"OpenAI summary error: {e}")

def main():
    rclpy.init()
    node = OpenAICommandParser()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
