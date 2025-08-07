import os
import json
from PIL import Image
import torch

# Hugging Face libraries for local model loading
# IMPORTANT CHANGE: Importing AutoModelForVision2Seq instead of AutoModelForCausalLM
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor

# Qwen-VL-specific utility for processing multimodal inputs
from qwen_vl_utils import process_vision_info

# --- Configuration ---
image_path = "" # Make sure this path is correct relative to your script or absolute.
category = ""
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct" # Using the 3B model for CPU compatibility

# --- Step 1: Load the Model and Processor Locally (CPU-focused) ---
print(f"[INFO] Loading model: {MODEL_NAME} for CPU inference using AutoModelForVision2Seq.")
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Load the model using AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained( # <--- CHANGED CLASS HERE
        MODEL_NAME,
        torch_dtype=torch.float32, # Use float32 for CPU for compatibility and stability
        device_map="cpu",          # Explicitly target CPU
    )
    model.eval()
    print("[INFO] Model loaded successfully on CPU using AutoModelForVision2Seq.")

except Exception as e:
    print(f"[ERROR] Failed to load model {MODEL_NAME} on CPU: {e}")
    print("This might indicate a deeper compatibility issue with the current transformers version or model configuration.")
    print("Please ensure your 'transformers' library is the absolute latest from source.")
    exit()

# --- Step 2: Load and prepare the image ---
print(f"[INFO] Loading image from: {image_path}")
try:
    image = Image.open(image_path).convert("RGB")
    print("[INFO] Image loaded successfully.")
except FileNotFoundError:
    print(f"[ERROR]: Image file not found at {image_path}")
    exit()
except Exception as e:
    print(f"[ERROR]: An error occurred while opening the image: {e}")
    exit()

# --- Step 3: Define the user message with image and text ---
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": f"""
                Give me ONLY the handwritten data in the format: {{"{category}":"<handwritten data or None>"}}
                """
            }
        ]
    }
]

# --- Step 4: Process the vision inputs using qwen_vl_utils ---
image_inputs, video_inputs = process_vision_info(messages)

# --- Step 5: Apply the chat template to the text prompt ---
text_for_model = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# --- Step 6: Prepare final model inputs ---
inputs = processor(
    text=[text_for_model],
    images=image_inputs,
    videos=video_inputs,
    padding=False,  # Changed from True to False for efficiency
    return_tensors="pt"
).to("cpu") # Explicitly move inputs to CPU

# --- Step 7: Generate the response ---
print("[INFO] Generating response.")
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=64, # Reduced from 512 to 64 for efficiency
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=processor.tokenizer.eos_token_id
    )

# --- Step 8: Decode the output ---
generated_text = processor.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("[INFO] Raw model output (before JSON parsing):")
print(generated_text)

# --- Step 9: Parse the JSON output ---
try:
    parsed_json = json.loads(generated_text)
    print("\n[OUTPUT] Parsed JSON object:")
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError as e:
    print(f"\n[ERROR] Decoding JSON: {e}")
    print(f"[ERROR] Raw output received (not valid JSON): {generated_text}")
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred during JSON parsing: {e}")
    print(f"[ERROR] Raw output received: {generated_text}")