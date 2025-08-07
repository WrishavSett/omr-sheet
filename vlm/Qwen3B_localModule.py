import os
import json
from PIL import Image
import torch

# Hugging Face libraries for local model loading
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor

# Qwen-VL-specific utility for processing multimodal inputs
# Ensure qwen_vl_utils.py is in the same directory or accessible in your Python path
# You might need to create this file based on Qwen-VL's official examples if not provided
# A simplified qwen_vl_utils.py might look like:
"""
# qwen_vl_utils.py
from PIL import Image
def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for msg in messages:
        if msg["role"] == "user":
            for content_item in msg["content"]:
                if content_item["type"] == "image":
                    image_inputs.append(content_item["image"])
                elif content_item["type"] == "video":
                    video_inputs.append(content_item["video"])
    return image_inputs, video_inputs
"""
from qwen_vl_utils import process_vision_info

# Global variables for model and processor to avoid reloading for each image
# These will be initialized once when qwen_vl_local is first called
_model = None
_processor = None
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

def _load_qwen_model_and_processor():
    """Loads the Qwen-VL model and processor locally if not already loaded."""
    global _model, _processor

    if _model is None or _processor is None:
        print(f"[INFO] Lazily loading model: {MODEL_NAME} for CPU inference using AutoModelForVision2Seq.")
        try:
            _processor = AutoProcessor.from_pretrained(MODEL_NAME)
            _model = AutoModelForVision2Seq.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # Use float32 for CPU for compatibility and stability
                device_map="cpu",           # Explicitly target CPU
            )
            _model.eval()
            print("[INFO] Model loaded successfully on CPU using AutoModelForVision2Seq.")
        except Exception as e:
            print(f"[ERROR] Failed to load model {MODEL_NAME} on CPU: {e}")
            print("This might indicate a deeper compatibility issue with the current transformers version or model configuration.")
            print("Please ensure your 'transformers' library is the absolute latest from source.")
            _model = None  # Ensure these are None if loading fails
            _processor = None
            raise RuntimeError(f"Failed to load Qwen-VL model: {e}")

def qwen_vl_local(image_path: str, category: str):
    """
    Performs local inference using the Qwen-VL-3B-Instruct model
    to extract handwritten data for a specified category from an image.

    Args:
        image_path (str): The path to the input image file.
        category (str): The category of handwritten data to extract (e.g., "reg_no").

    Returns:
        dict or None: A dictionary containing the extracted data in the format
                      {"category": "<handwritten data or None>"}, or None if an error occurs.
    """
    # Ensure model and processor are loaded
    try:
        _load_qwen_model_and_processor()
    except RuntimeError:
        return None # Return None if model loading failed

    # Read and prepare the image
    print(f"[INFO] Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print("[INFO] Image loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR]: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"[ERROR]: An error occurred while opening the image: {e}")
        return None

    # Prepare the message payload for the chat completion
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

    try:
        # Process the vision inputs using qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(messages)

        # Apply the chat template to the text prompt
        text_for_model = _processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare final model inputs
        inputs = _processor(
            text=[text_for_model],
            images=image_inputs,
            videos=video_inputs,
            padding=False, # Changes from True to False for efficiency
            return_tensors="pt"
        ).to("cpu")  # Explicitly move inputs to CPU

        # Generate the response
        print("[INFO] Generating response.")
        with torch.no_grad():
            output_tokens = _model.generate(
                **inputs,
                max_new_tokens=64, # Reduced from 512 to 64 for efficiency
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=_processor.tokenizer.eos_token_id
            )

        # Decode the output
        generated_text = _processor.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print("[INFO] Raw model output (before JSON parsing):")
        print(generated_text)

        # The model sometimes wraps the JSON output in a markdown code block.
        # We need to strip these delimiters if they are present.
        json_output_string = generated_text
        if json_output_string.startswith("```json") and json_output_string.endswith("```"):
            json_output_string = json_output_string[len("```json"): -len("```")].strip()
            print("[INFO] Stripped markdown code block delimiters from the output.")
        elif json_output_string.startswith("```") and json_output_string.endswith("```"):
             # Handle generic code blocks just in case
            json_output_string = json_output_string[len("```"): -len("```")].strip()
            print("[INFO] Stripped generic markdown code block delimiters from the output.")


        # Attempt to parse the JSON string
        parsed_json = json.loads(json_output_string)

        # Validate the expected format: {"category": "data"}
        if category in parsed_json and len(parsed_json) == 1:
            print("[OUTPUT] Parsed JSON object:")
            print(json.dumps(parsed_json, indent=2))
            return parsed_json
        else:
            print(f"[ERROR] Model output format unexpected. Expected {{\"'{category}'\":\"<data>\"}}, got: {parsed_json}")
            return {category: "Error: Unexpected format"}

    except json.JSONDecodeError as e:
        print(f"[ERROR] Decoding JSON: {e}")
        print(f"[ERROR] Raw output received: {generated_text}")
        return {category: "Error: JSONDecodeError"}
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return {category: f"Error: {str(e)}"}


def process_all_images(image_data: list):
    """
    Processes a list of image paths and categories using the qwen_vl_local function.

    Args:
        image_data (list): A list of dictionaries, where each dictionary contains
                           "image_path" and "category" keys.

    Returns:
        dict: A dictionary consolidating all processed results, where keys are
              categories and values are the extracted data or "None" if processing failed.
    """
    congregated_results = {}
    for item in image_data:
        image_path = item.get("image_path")
        category = item.get("category")

        if not image_path or not category:
            print(f"[WARNING] Skipping invalid item: {item}. Both 'image_path' and 'category' are required.")
            continue

        print(f"\n[INFO] Processing image: {image_path} for category: {category}")
        # Call the local Qwen-VL inference function
        result = qwen_vl_local(image_path=image_path, category=category)

        if result:
            # Merge the individual result into the congregated_results
            congregated_results.update(result)
        else:
            # If qwen_vl_local returns None (due to model loading or image issues)
            # or an error within the function results in an error value
            if category not in congregated_results: # Avoid overwriting if result was an error dict
                congregated_results[category] = None
            print(f"[WARNING] Failed to process {image_path} for category {category}. Result set to None or error.")


    print("\n[OUTPUT] Congregated JSON Response:")
    print(json.dumps(congregated_results, indent=2))
    return congregated_results


def main():
    """Main function to define image data and initiate processing."""
    # Ensure these image paths exist and contain handwritten data for the categories
    images_to_process = [
        {"image_path": "./VLM/reg_no.jpg", "category": "reg_no"},
        {"image_path": "./VLM/roll_no.jpg", "category": "roll_no"},
        {"image_path": "./VLM/booklet_no.jpg", "category": "booklet_no"}
        ]
    final_results = process_all_images(images_to_process)
    print("\n##### Final Results #####")
    print(final_results)

if __name__ == "__main__":
    main()