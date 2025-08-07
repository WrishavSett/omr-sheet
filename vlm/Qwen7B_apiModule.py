import os
import base64
import json
from huggingface_hub import InferenceClient

def qwen7b(image_path: str, category: str):
    # Read the image file as bytes
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        print(f"[ERROR]: Image file not found at {image_path}")
        return None # Return None if the file doesn't exist

    # Convert bytes to base64 encoding (often required for JSON payloads)
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # Initialize the InferenceClient
    client = InferenceClient(
        provider="hyperbolic",
        api_key="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", # Placeholder: Use your actual API key
    )

    # Prepare the message payload for the chat completion
    messages = [
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
    ]

    try:
        # Make the API call to the model
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=messages,
        )

        print("[INFO] Model response received.")
        # Extract the content from the completion
        json_output_string = completion.choices[0].message.content

        # The model sometimes wraps the JSON output in a markdown code block.
        # We need to strip these delimiters if they are present.
        if json_output_string.startswith("```json") and json_output_string.endswith("```"):
            json_output_string = json_output_string[len("```json"): -len("```")].strip()
            print("[INFO] Stripped markdown code block delimiters from the output.")

        # Attempt to parse the JSON string
        parsed_json = json.loads(json_output_string)

        print("[OUTPUT] Parsed JSON object:")
        print(json.dumps(parsed_json, indent=2))
        return parsed_json

    except json.JSONDecodeError as e:
        print(f"[ERROR] Decoding JSON: {e}")
        print(f"[ERROR] Raw output received: {json_output_string}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return None


def process_all_images(image_data: list):
    congregated_results = {}
    for item in image_data:
        image_path = item.get("image_path")
        category = item.get("category")

        if not image_path or not category:
            print(f"[WARNING] Skipping invalid item: {item}. Both 'image_path' and 'category' are required.")
            continue

        print(f"\n[INFO] Processing image: {image_path} for category: {category}")
        result = qwen7b(image_path=image_path, category=category)

        if result:
            # Merge the individual result into the congregated_results
            congregated_results.update(result)
        else:
            # If qwen7b returns None, add the category with None value to indicate failure
            congregated_results[category] = None
            print(f"[WARNING] Failed to process {image_path} for category {category}. Result set to None.")

    print("\n[OUTPUT] Congregated JSON Response:")
    print(json.dumps(congregated_results, indent=2))
    return congregated_results


def main():
    images_to_process = [
        {"image_path": "./VLM/reg_no.jpg", "category": "reg_no"},
        {"image_path": "./VLM/roll_no.jpg", "category": "roll_no"},
        {"image_path": "./VLM/booklet_no.jpg", "category": "booklet_no"},
    ]
    final_results = process_all_images(images_to_process)
    print("\n##### Final Results #####")
    print(final_results)

if __name__ == "__main__":
    main()