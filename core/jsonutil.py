"""Parse the JSON object a model returns, tolerating code fences and surrounding text."""
import json
import re


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the model reply")
    return json.loads(cleaned[start:end + 1])