# OMR Sheet Processing System

This repository contains a comprehensive system for processing Optical Mark Recognition (OMR) sheets. The pipeline is designed to perform a series of tasks, including anchor detection, field mapping, marked option detection, and Intelligent Character Recognition (ICR) for handwritten fields. The system is built with a modular structure, allowing for easy customization and extension.

***

### Project Structure

OMR Sheet Processing System
This repository contains a comprehensive system for processing Optical Mark Recognition (OMR) sheets. The pipeline is designed to perform a series of tasks, including anchor detection, field mapping, marked option detection, and Intelligent Character Recognition (ICR) for handwritten fields. The system is built with a modular structure, allowing for easy customization and extension.

Project Structure
```
.
├── abhigyan/
├── api/
│   ├── qwen3b.py
│   └── request.py
├── dataset/
│   ├── ASSAM/
│   ├── HS/
│   └── TEST/
├── icr/
├── labels/
├── notebooks/
│   └── OMR.ipynb
├── omr/
│   ├── ASSAM/
│   ├── SAROJ/
│   └── TEST/
├── redundant/
│   ├── ls_export.py
│   └── main.py
├── src/
│   ├── batch_process.py
│   ├── ls_export.py
│   └── ls_import.py
├── temp/
├── utils/
│   ├── barcodeScanner.py
│   ├── labels.py
│   ├── temp.py
│   └── utils.py
└── README.md
```

***

### Core Functionality

The project's main pipeline is orchestrated by `main.py` and consists of the following key steps:

1.  **Anchor Detection:** The system first identifies fixed anchor points on the OMR sheet to correct for rotation and perspective, ensuring all sheets are aligned consistently.
2.  **Field Mapping:** It then maps predefined fields (e.g., questions, roll numbers) based on the aligned anchors.
3.  **Marked Option Detection:** The system analyzes the mapped fields to detect marked bubbles, using configurable thresholds to determine "full" and "partial" marks.
4.  **Intelligent Character Recognition (ICR):** For handwritten fields like roll numbers or registration numbers, the system uses a Vision Language Model (VLM) via a local API (`api/qwen3b.py`) to recognize and extract the text.
5.  **Data Consolidation:** Finally, all the extracted information is consolidated and restructured into a final JSON output.

***

### Module Breakdown

#### `main.py`

This is the **main orchestrator** of the entire OMR processing pipeline. It loads a configuration from `config.json`, parses command-line arguments, and sequentially calls functions from other modules to perform all the necessary steps.

* **Functionality**:
    * Loads `config.json` for base paths.
    * Parses command-line arguments for `omr_template_name`, `date`, `batch_name`, and various optional flags (`--save-anchor`, `--save-mapped`, `--draw-bboxes`, `--full-mark`, `--partial-mark`).
    * Calls `process_batch` for anchor detection and image alignment.
    * Calls `process_field_mapping` to identify field regions.
    * Calls `process_marked_options` to detect filled bubbles.
    * Calls `process_icr_requests` to send specific fields to the ICR API.
    * Calls `json_restructure` to consolidate all results into a final JSON format.

* **Inputs**: Command-line arguments and `config.json`.
* **Outputs**: A series of processed images and a final JSON output file located in the batch's output directory.

---

### Batch Process Module: `src/batch_process.py`

The `batch_process.py` script is a crucial component of the OMR processing pipeline. It is responsible for the initial processing of a batch of OMR sheets, specifically focusing on anchor detection and image alignment.

#### Functionality and Working

The script contains a single primary function, `process_batch`, which orchestrates the entire workflow for a given set of OMR sheets. The process works as follows:

1.  **Configuration and Pathing:** The function first constructs the necessary file paths for the input images, the output directory, and the template's configuration files (e.g., `key_field_coordinates.json`) based on the provided `base_folder`, `omr_template_name`, `date`, and `batch_name`.

2.  **Image Iteration:** It then iterates through all `.jpg` files found in the designated batch folder. This loop processes each OMR sheet individually.

3.  **Anchor Detection:** For each image, the script attempts to detect anchor points. These are predefined, fixed markers on the OMR sheet that serve as a reference for alignment. The success of this step is critical; if anchors are not found, the image cannot be properly aligned, and an error is logged.

4.  **Image Alignment:** If the anchor detection is successful, the script uses the coordinates of these detected points to perform a geometric transformation on the image. This corrects for any rotation, perspective distortion, or skew, aligning the sheet to a consistent, standardized orientation.

5.  **Output Saving:** If the `save_anchor_images` flag is set to `True`, the script saves a copy of the newly aligned image to the output directory. This is useful for visual inspection and debugging.

6.  **JSON Generation:** The script updates a JSON file (named after the `batch_name`) with the results of the processing. This file contains metadata about each image, including its name, processing status (`success` or `failure`), and the path to the aligned image. This output JSON becomes the input for subsequent stages of the pipeline, such as field mapping and marked option detection.

This robust process ensures that all sheets in a batch are consistently oriented and ready for the next steps of the pipeline, preventing potential errors from misalignment.

#### Functions

* `process_batch(base_folder, omr_template_name, date, batch_name, save_anchor_images)`: This is the main and only function within the script. It orchestrates the entire batch processing workflow, from reading inputs and detecting anchors to aligning images and saving the final JSON output.

#### Input and Output Parameters/Arguments

* **Inputs (for `process_batch` function):**
    * `base_folder` (string): The root directory of the entire project, used as a reference for all other paths.
    * `omr_template_name` (string): The name of the specific OMR sheet template being used (e.g., "HSOMR").
    * `date` (string): The date associated with the batch of sheets.
    * `batch_name` (string): The specific name of the batch being processed.
    * `save_anchor_images` (boolean): An optional flag that, if `True`, saves the aligned images to the output directory.

* **Output (from `process_batch` function):**
    * A `results` dictionary that contains a count of successfully processed images.
    * A JSON file named after the `batch_name` (e.g., `BATCH01.json`) is created in the batch's output directory. This file serves as the main output, containing a list of all images in the batch and their processing status, as well as paths to the aligned images.

---

### API Directory (`api`)

#### `qwen3b.py`

This script serves as a **local Flask web API** for the ICR service. It uses the Qwen2.5-VL-3B-Instruct model to read handwritten text from images.

* **Functionality**:
    * Loads the Qwen VLM and its processor using **4-bit quantization** to reduce memory usage.
    * Sets up a `/predict` endpoint that listens for HTTP `POST` requests.
    * Processes an uploaded image (`image`) and a category (`category`) to generate a text response.
    * Parses the VLM's output and returns the extracted text as a JSON object.
* **Inputs**: An image file and a category string via a `POST` request to `/predict`.
* **Outputs**: A JSON object containing the extracted text (e.g., `{"roll_no": "12345"}`).

#### `request.py`

This script is the **client for the ICR API**. It automates the process of sending multiple images to the `qwen3b.py` server and collecting the results.

* **Functionality**:
    * Configured with the API URL and the directory containing ICR images.
    * Scans predefined subdirectories (`barcode_no`, `booklet_no`, etc.) for images.
    * Iterates through each image, sending it to the API along with its category.
    * Handles successful and failed API responses, logging errors and storing results.
    * Saves all consolidated results into a single output JSON file.
* **Inputs**: The `BASE_IMAGE_DIR` containing subdirectories of images and a configured `API_URL`.
* **Outputs**: A JSON file (`ICR_Output.json`) containing the ICR results for all processed images.

---

### Source Directory (`src`)

#### `ls_export.py`

This script is a utility for exporting image data and annotations to be used with **Label Studio**. It reads a batch's JSON file and converts it into a format that can be easily imported into the labeling tool.

* **Functionality**:
    * Reads the `batch_name.json` output file.
    * Iterates through the images and their detected key fields.
    * Generates a new JSON file (`ls_import.json`) with pre-populated tasks, including image paths and any existing annotations, for import into Label Studio.
* **Inputs**: `base_folder` (string), `omr_template_name` (string), `date` (string), and `batch_name` (string).
* **Outputs**: A file named `ls_import.json` formatted for Label Studio.

#### `ls_import.py`

This script is the counterpart to `ls_export.py`. It imports annotated data from a Label Studio-formatted JSON file and merges it back into the project's output JSON.

* **Functionality**:
    * Reads the original `batch_name.json` file and a `labelstudio_output.json` file.
    * Iterates through the new annotations from Label Studio.
    * Updates the corresponding entries in `batch_name.json` with the new or corrected bounding box coordinates and text values.
* **Inputs**: `base_folder` (string), `omr_template_name` (string), `date` (string), `batch_name` (string), and the path to the Label Studio output file.
* **Outputs**: The original `batch_name.json` file is overwritten with the merged data.

---

### Utility Directory (`utils`)

#### `barcodeScanner.py`

A standalone function for **decoding barcodes** from an image.

* **Functionality**:
    * Uses `cv2` to read an image and preprocess it with grayscale conversion, thresholding, and resizing.
    * Uses `pyzbar` to decode any barcodes present in the preprocessed image.
* **Inputs**: `image_path` (string).
* **Outputs**: A string of the decoded barcode data.

#### `labels.py`

An **interactive command-line tool** for generating the various configuration files needed by the OMR pipeline.

* **Functionality**:
    * Prompts the user for the number of anchors, key fields, and questions.
    * Allows the user to name key fields and define options for each question.
    * Generates `key_fields.json`, `field_mappings.json`, `question_mappings.json`, and `classes.txt`.
* **Inputs**: User input via the command line.
* **Outputs**: Multiple JSON and text files in the project's root directory that define the labels and structure of the OMR sheets.

#### `temp.py`

This script, located in the `redundant` directory, is a **temporary test script** used for debugging the data processing workflow.

* **Functionality**:
    * Contains hardcoded paths to a specific OMR template.
    * Sequentially calls functions from `utils.py` to test the processing and updating of JSON files, such as `process_omr_data` and `update_key_field_dimensions`.
* **Inputs**: Hardcoded file paths.
* **Outputs**: Modified JSON files and console output for debugging.

#### `utils.py`

This file contains a collection of **general-purpose utility functions** for data processing, primarily focused on the JSON configuration files.

* **Functionality**:
    * `process_omr_data`: Validates and restructures a JSON file based on a template.
    * `process_key_coordinates`: Calculates and generates the bounding box coordinates for key fields based on `field_mappings.json`.
    * `update_key_field_dimensions`: Merges new coordinate data into an existing JSON file.
    * `update_field_names_for_omr`: Updates field names for better readability.
* **Inputs**: Varies per function, but typically includes paths to JSON files (`json_file_path`, `key_fields_path`, `field_mappings_path`).
* **Outputs**: Modified or newly created JSON files.