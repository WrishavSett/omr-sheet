import os
import re
import json
from utils.utils import process_omr_data, process_key_coordinates, update_key_field_dimensions

# base_folder,
# omr_template_name,
# date
# batch_name

omr_template_name = "HSOMR"  # Default template name for demonstration
json_file_path = "D:/OMR_DEV/abhigyan/BATCH018 copy.json"
field_mappings_path = "D:/OMR_DEV/abhigyan/field_mappings.json"
key_fields_path = "D:/OMR_DEV/abhigyan/key_fields.json"
output_folder = "D:/OMR_DEV/abhigyan/"
coordinates_output_path = os.path.join(output_folder, "key_field_coordinates.json")


output_file_path = process_omr_data(
    json_file_path=json_file_path, 
    key_fields_path=key_fields_path, 
    template_name=omr_template_name, 
    output_file_path=json_file_path)
print(f"Debug: Output File Path: {output_file_path}")

coordinates_output_path = process_key_coordinates(
    field_mappings_path=field_mappings_path, 
    key_fields_path=key_fields_path, 
    output_path=coordinates_output_path)
print(f"Debug: Coordinates Output Path: {coordinates_output_path}")

key_fields_path = "D:/OMR_DEV/abhigyan/key_fields.json"

# NEW: Call the function to update the dimensions in the output JSON
path = update_key_field_dimensions(
    output_json_path=output_file_path,
    key_field_coordinates_path=coordinates_output_path,
    key_fields_path=key_fields_path
)
print(f"Debug: Fully Updated Output File Path: {path}")