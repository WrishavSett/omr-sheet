import requests
import base64
import os
import json
import random

def read_labels_from_file(file_path):
    """
    Reads label names from a text file, one label per line.
    """
    labels = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                label = line.strip()
                if label:  # Only add non-empty lines
                    labels.append(label)
        return labels
    except FileNotFoundError:
        print(f"Error: Labels file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error reading labels file: {e}")
        return None

def generate_random_hex_color():
    """
    Generates a random hexadecimal color code.
    """
    return '#%06x' % random.randint(0, 0xFFFFFF)

def generate_label_config_xml(labels):
    """
    Generates the Label Studio XML configuration for object detection
    with bounding boxes, using the provided list of labels and random colors.
    """
    # Muted: "Warning: No labels provided. Generating a basic config without specific labels."
    if not labels:
        return """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="DefaultLabel" background="#FF0000"/>
  </RectangleLabels>
</View>
"""

    labels_xml = ""
    for label in labels:
        color = generate_random_hex_color() # Generate a random color for each label
        labels_xml += f'    <Label value="{label}" background="{color}"/>\n'

    config_xml = f"""
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
{labels_xml.strip()}
  </RectangleLabels>
</View>
"""
    return config_xml

def get_or_create_project(ls_url, api_key, project_name, label_config_xml, project_description=None):
    """
    Checks if a project exists. If it does, updates its configuration.
    If not, creates a new project with the specified configuration.
    Returns the project ID.
    """
    headers = {
        'Authorization': f'Token {api_key}',
        'Content-Type': 'application/json'
    }

    # 1. Check if project exists
    try:
        response = requests.get(f"{ls_url}/api/projects", headers=headers)
        response.raise_for_status()
        projects = response.json()
        
        # Ensure 'projects' is a list, even if API returns a single dictionary or something else
        if not isinstance(projects, list):
            if isinstance(projects, dict) and 'id' in projects and 'title' in projects:
                projects = [projects]
            else:
                # Muted: print(f"Warning: Unexpected response format from {ls_url}/api/projects. Expected a list, got {type(projects).__name__}. Full response: {json.dumps(projects, indent=2)}")
                # Muted: print("Proceeding as if no projects were found or an API error occurred.")
                projects = [] # Treat as empty list if unexpected format
        
        existing_project_id = None
        for project in projects:
            if project['title'] == project_name:
                existing_project_id = project['id']
                break

        if existing_project_id:
            print(f"Project '{project_name}' already exists (ID: {existing_project_id}). Attempting to update its configuration...")
            # Update existing project configuration
            update_payload = {
                'label_config': label_config_xml
            }
            if project_description is not None: # Only add if provided
                update_payload['description'] = project_description
            response = requests.patch(f"{ls_url}/api/projects/{existing_project_id}", headers=headers, data=json.dumps(update_payload))
            response.raise_for_status()
            print(f"Configuration for project '{project_name}' updated successfully.")
            return existing_project_id
        else:
            print(f"Project '{project_name}' not found. Creating a new project...")
            create_payload = {
                'title': project_name,
                'label_config': label_config_xml,
                'project_type': 'image_object_detection' # Optional, but good to specify
            }
            if project_description is not None: # Only add if provided
                create_payload['description'] = project_description
            response = requests.post(f"{ls_url}/api/projects", headers=headers, data=json.dumps(create_payload))
            response.raise_for_status()
            new_project = response.json()
            print(f"Project '{new_project['title']}' created successfully (ID: {new_project['id']}).")
            return new_project['id']

    except requests.exceptions.RequestException as e:
        print(f"Error interacting with Label Studio API for project management: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def upload_image_and_create_task(ls_url, api_key, project_id, image_path):
    """
    Uploads an image to Label Studio and creates a task for it within the specified project.
    """
    headers = {
        'Authorization': f'Token {api_key}'
    }

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Label Studio's import endpoint can take files directly
        # Determine MIME type based on file extension
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        file_extension = os.path.splitext(image_path)[1].lower()
        content_type = mime_type_map.get(file_extension, 'application/octet-stream') # Default if unknown

        files = {'file': (os.path.basename(image_path), image_data, content_type)}

        # Muted: print(f"Uploading image '{os.path.basename(image_path)}' to project ID {project_id}...")
        response = requests.post(f"{ls_url}/api/projects/{project_id}/import", headers=headers, files=files)
        response.raise_for_status()
        
        # The import endpoint usually returns a summary of imported tasks
        # Muted: import_result = response.json()
        # Muted: print(f"Image uploaded successfully. Import result: {import_result}")
        # Muted: if import_result and import_result.get('total_annotations_imported', 0) > 0:
        # Muted:     print("Task(s) created from the uploaded image.")
        # Muted: else:
        # Muted:     print("No new tasks reported by Label Studio after import. Check Label Studio UI.")
        return True

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error uploading image to Label Studio: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during image upload: {e}")
        return False

def main():
    """
    Main function to orchestrate the Label Studio upload process.
    """
    print("--- Label Studio Uploader for Object Detection ---")
    print("Please provide the following details to connect to Label Studio and set up your project:")

    # ls_url = input("Enter your Label Studio URL (e.g., http://localhost:8080): ").strip()
    ls_url = "http://localhost:8080/"  # Default URL for local development
    print(f"Using default Label Studio URL: {ls_url}")
    if not ls_url:
        print("Label Studio URL cannot be empty. Exiting.")
        return

    # api_key = input("Enter your Label Studio API Key (found in your user profile settings): ").strip()
    api_key = "1e0cfed5f935b1d517af3bc9048763a5d7163c2c"  # Placeholder for API key
    print(f"Using default API Key: {api_key}")
    if not api_key:
        print("API Key cannot be empty. Exiting.")
        return

    project_name = input("Enter the desired Label Studio Project Name (will be created if it doesn't exist): ").strip()
    if not project_name:
        print("Project name cannot be empty. Exiting.")
        return
    
    project_description = input("Enter an optional project description (press Enter to skip): ").strip()

    image_path = input("Enter the full path to the image file you want to upload: ").strip()
    if not image_path:
        print("Image path cannot be empty. Exiting.")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at '{image_path}'. Exiting.")
        return

    labels_file_path = input("Enter the full path to the .txt file containing label names (one per line): ").strip()
    if not labels_file_path:
        print("Labels file path cannot be empty. Exiting.")
        return
    if not os.path.exists(labels_file_path):
        print(f"Error: Labels file does not exist at '{labels_file_path}'. Exiting.")
        return

    # 1. Read labels
    labels = read_labels_from_file(labels_file_path)
    if labels is None:
        return # Exit if there was an error reading labels

    # 2. Generate Label Studio XML configuration
    label_config_xml = generate_label_config_xml(labels)

    # 3. Get or create the project
    project_id = get_or_create_project(ls_url, api_key, project_name, label_config_xml, project_description)
    if project_id is None:
        print("Failed to get or create Label Studio project. Exiting.")
        return

    # 4. Upload the image and create a task
    if upload_image_and_create_task(ls_url, api_key, project_id, image_path):
        print("\nProcess completed successfully!")
        print(f"You can now access your project at: {ls_url}/projects/{project_id}/data")
    else:
        print("\nImage upload and task creation failed.")

if __name__ == "__main__":
    main()
