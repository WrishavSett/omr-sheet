import requests
import os
import re # For regular expressions to parse project ID
from urllib.parse import urlparse
import zipfile # To inspect and extract the contents of the zip file
import shutil # To copy files

def export_label_studio_project(ls_url, api_key, project_id, export_type, key_fields_json_path):
    """
    Exports a Label Studio project in the specified format,
    then unzips it to a user-defined location, copies the key_fields.json file, and deletes the zip file.

    Args:
        ls_url (str): The base URL of your Label Studio instance (e.g., http://localhost:8080).
        api_key (str): Your Label Studio API key.
        project_id (int): The ID of the project to export.
        export_type (str): The specific export type string (e.g., "YOLO_WITH_IMAGES").
        key_fields_json_path (str): The file path to the key_fields.json file to be copied.

    Returns:
        bool: True if the export and subsequent operations were successful, False otherwise.
    """
    headers = {
        'Authorization': f'Token {api_key}',
    }

    # Construct the export URL with the selected export_type
    export_url = f"{ls_url}/api/projects/{project_id}/export?exportType={export_type}"
    
    print(f"Attempting to export project ID {project_id} from {ls_url} in {export_type} format...")
    print(f"Export URL being used: {export_url}")

    output_filename = None # Initialize to None
    full_extraction_path = None # Initialize to None
    try:
        # Stream the response content to handle potentially large files
        response = requests.get(export_url, headers=headers, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Determine filename from Content-Disposition header or use a default
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            fname = re.findall('filename="(.+)"', content_disposition)
            if fname:
                output_filename = fname[0]
            else:
                output_filename = f"label_studio_project_{project_id}_{export_type.lower()}_export.zip"
        else:
            output_filename = f"label_studio_project_{project_id}_{export_type.lower()}_export.zip"

        # Save the content to a file
        temp_output_filename = output_filename + ".temp"
        with open(temp_output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Rename to final filename only after successful download
        os.rename(temp_output_filename, output_filename)

        print(f"Project successfully exported to '{output_filename}'")

        # --- Unzip and Delete Logic ---
        if output_filename.lower().endswith('.zip'):
            print("\n--- ZIP File Operations ---")
            extraction_base_path = input("Enter the directory where you want to extract the files (e.g., C:/Exports or /home/user/data): ").strip()
            if not extraction_base_path:
                print("Extraction directory cannot be empty. Skipping extraction and file copy.")
                os.remove(output_filename) # Still delete the zip file
                return False

            extracted_folder_name = input("Enter the name for the folder inside the extraction directory (e.g., my_yolo_dataset): ").strip()
            if not extracted_folder_name:
                print("Extracted folder name cannot be empty. Skipping extraction and file copy.")
                os.remove(output_filename) # Still delete the zip file
                return False

            full_extraction_path = os.path.join(extraction_base_path, extracted_folder_name)

            try:
                os.makedirs(full_extraction_path, exist_ok=True) # Create directory if it doesn't exist
                print(f"Extracting '{output_filename}' to '{full_extraction_path}'...")
                with zipfile.ZipFile(output_filename, 'r') as zip_ref:
                    zip_ref.extractall(full_extraction_path)
                print("Extraction complete.")

                # Delete the original zip file
                os.remove(output_filename)
                print(f"Deleted original ZIP file: '{output_filename}'")

            except FileNotFoundError:
                print(f"Error: ZIP file '{output_filename}' not found for extraction.")
                return False
            except zipfile.BadZipFile:
                print(f"Error: Downloaded file '{output_filename}' is not a valid ZIP file. Cannot extract.")
                os.remove(output_filename) # Delete corrupted zip
                return False
            except Exception as e:
                print(f"An error occurred during extraction or deletion: {e}")
                print(f"The ZIP file '{output_filename}' might still be present.")
                return False
        else:
            print(f"\n--- Note: Exported file '{output_filename}' is not a ZIP file. Skipping extraction and copy. ---")
            return True # Consider this a success even without the zip operations

        # --- Key Fields JSON Copy Logic ---
        print("\n--- Copying key_fields.json ---")
        if not key_fields_json_path:
            print("No path for key_fields.json was provided. Skipping file copy.")
        else:
            try:
                # Get the destination path
                destination_path = os.path.join(full_extraction_path, os.path.basename(key_fields_json_path))
                shutil.copy(key_fields_json_path, destination_path)
                print(f"Successfully copied '{key_fields_json_path}' to '{destination_path}'")
            except FileNotFoundError:
                print(f"Error: Source file '{key_fields_json_path}' not found. Cannot copy.")
            except Exception as e:
                print(f"An unexpected error occurred during the copy operation: {e}")
        # --- End Key Fields JSON Copy Logic ---

        # --- List Directory Contents ---
        print(f"\n--- Contents of the directory '{full_extraction_path}' ---")
        if os.path.exists(full_extraction_path):
            for item in os.listdir(full_extraction_path):
                print(f" - {item}")
        else:
            print("Extraction directory not found.")
        # --- End List Directory Contents ---

        print("\n--- Important Note for Troubleshooting ---")
        print("If the content is not as expected (e.g., missing images for 'WITH_IMAGES' formats), please:")
        print("1. Double-check how the images were originally imported into Label Studio (direct upload vs. URL reference).")
        print("2. Try exporting the project directly from the Label Studio UI (Project Settings -> Export) using the *exact same format* to confirm expected output.")
        print("3. If the UI export works correctly but the script's export doesn't, inspect the browser's Developer Tools (Network tab) during the UI export to identify any subtle differences in the API request (URL, headers, or query parameters).")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error during export: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        # Ensure cleanup if download failed but a partial file exists
        if output_filename and os.path.exists(output_filename + ".temp"):
            os.remove(output_filename + ".temp")
            print(f"Cleaned up partial download file: '{output_filename}.temp'")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Ensure cleanup if download failed but a partial file exists
        if output_filename and os.path.exists(output_filename + ".temp"):
            os.remove(output_filename + ".temp")
            print(f"Cleaned up partial download file: '{output_filename}.temp'")
        return False

def get_project_id_from_url(project_url):
    """
    Extracts the project ID from a Label Studio project URL.
    e.g., http://localhost:8080/projects/104/data?tab=102 -> 104
    """
    parsed_url = urlparse(project_url)
    path_segments = [segment for segment in parsed_url.path.split('/') if segment] # Remove empty strings

    if len(path_segments) >= 2 and path_segments[-2] == 'projects':
        try:
            project_id = int(path_segments[-1])
            return project_id
        except ValueError:
            print(f"Could not parse project ID from URL path segment: '{path_segments[-1]}'")
            return None
    elif len(path_segments) >= 3 and path_segments[-3] == 'projects': # Handles /projects/ID/data
        try:
            project_id = int(path_segments[-2])
            return project_id
        except ValueError:
            print(f"Could not parse project ID from URL path segment: '{path_segments[-2]}'")
            return None
    
    print(f"Could not extract project ID from the provided URL: {project_url}")
    print("Expected URL format like: http://<host>:<port>/projects/<ID>/data")
    return None

def main():
    """
    Main function to orchestrate the Label Studio project export process.
    """
    print("--- Label Studio Project Exporter ---")

    # ls_url = input("Enter your Label Studio base URL (e.g., http://localhost:8080): ").strip()
    ls_url = "http://localhost:8080/"  # Default URL for local development
    print(f"Using default Label Studio URL: {ls_url}")
    if not ls_url:
        print("Label Studio URL cannot be empty. Exiting.")
        return

    # api_key = input("Enter your Label Studio API Key: ").strip()
    api_key = "1e0cfed5f935b1d517af3bc9048763a5d7163c2c"  # Placeholder for API key
    print(f"Using default API Key: {api_key}")
    if not api_key:
        print("API Key cannot be empty. Exiting.")
        return

    project_url = input("Enter the full URL of the Label Studio project (e.g., http://localhost:8080/projects/104/data): ").strip()
    if not project_url:
        print("Project URL cannot be empty. Exiting.")
        return

    project_id = get_project_id_from_url(project_url)
    if project_id is None:
        print("Invalid project URL provided. Exiting.")
        return

    # Add new input for key_fields.json file path
    key_fields_json_path = input("\nEnter the full path to the 'key_fields.json' file (generated by labels.py): ").strip()
    if not key_fields_json_path or not os.path.exists(key_fields_json_path):
        print(f"Warning: 'key_fields.json' file not found at '{key_fields_json_path}'. The file will not be copied.")
        # Proceed, but with a warning.
        key_fields_json_path = None


    # Define available export formats
    export_formats = {
        1: "YOLO",
        2: "YOLO_WITH_IMAGES",
    }

    selected_format_num = 2 # Default to YOLO_WITH_IMAGES for local development
    print(f"\nUsing default export format: {export_formats[selected_format_num]}")
    while selected_format_num not in export_formats:
        try:
            choice = input("Enter the number of your desired export format: ").strip()
            selected_format_num = int(choice)
            if selected_format_num not in export_formats:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    chosen_export_type = export_formats[selected_format_num]

    export_label_studio_project(ls_url, api_key, project_id, chosen_export_type, key_fields_json_path)

if __name__ == "__main__":
    main()