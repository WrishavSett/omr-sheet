image_path = "./VLM/reg_no.jpg"
category = "reg_no"

# Read the image file as bytes
try:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
except FileNotFoundError:
    print(f"[ERROR]: Image file not found at {image_path}")
    exit() # Exit if the file doesn't exist

# Convert bytes to base64 encoding (often required for JSON payloads)
import base64
encoded_image = base64.b64encode(image_bytes).decode('utf-8')


import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hyperbolic",
    api_key="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",
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                    Give me ONLY the handwritten data in the format: {{"{category}":"<handwritten data or None>"}}
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}" # Assuming JPEG, adjust if needed
                    }
                }
            ]
        }
    ],
)

print("[INFO]", completion.choices[0].message)


import json

# Assuming 'completion' object is available from your previous code execution
json_output_string = completion.choices[0].message.content

# The model sometimes wraps the JSON output in a markdown code block.
# We need to strip these delimiters if they are present.
if json_output_string.startswith("```json") and json_output_string.endswith("```"):
    json_output_string = json_output_string[len("```json"): -len("```")].strip()
    print("[INFO] Stripped markdown code block delimiters from the output.")

try:
    parsed_json = json.loads(json_output_string)

    print("[OUTPUT] Parsed JSON object:")
    print(json.dumps(parsed_json, indent=2))

except json.JSONDecodeError as e:
    print(f"[ERROR] Decoding JSON: {e}")
    print(f"[ERROR] Raw output received: {json_output_string}")