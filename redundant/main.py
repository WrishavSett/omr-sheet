import os
import re
import cv2
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any

# --- Global Configurations / Constants ---
TEMP_DIR = "D:\\OMR_DEV\\temp"
ANCHOR_NAME = "anchor_1"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif') # Common image extensions

class OMRProcessor:
    """
    A class to process OMR sheets, including template creation and test sheet evaluation.
    It identifies and extracts information from OMR bubbles based on a template.
    """

    def __init__(self, temp_directory: str = "D:\\OMR_DEV\\temp"):
        """
        Initializes the OMRProcessor.

        Args:
            temp_directory (str): The path to the temporary directory for saving intermediate files.
        """
        self.temp_directory = temp_directory
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)
            print(f"[INFO] Created temporary directory: {self.temp_directory}")
        else:
            print(f"[INFO] Temporary directory already exists: {self.temp_directory}")

        # Template related attributes
        self.template_image = None
        self.template_name = None
        self.labels = []
        self.classes = []
        self.anchor_boxes = {}
        self.object_centers = {}
        self.object_boxes = {}
        self.anchor_center = None
        self.template_data = None # Stores the loaded template JSON data

        # Test image processing attributes (will be updated per image in batch processing)
        self.test_image = None
        self.test_name = None
        self.transformed_center = None
        self.transformed_bbox = None
        self.result_image = None # Image with drawn detections (for visualization)
        self.detected_answers = {} # Stores results for the current single image being processed

    def _parse_class_name(self, name: str) -> Tuple[Optional[str], Any]:
        """
        Parses a class name to determine its type and identifier.

        Args:
            name (str): The class name from the YOLO label.

        Returns:
            Tuple[Optional[str], Any]: A tuple containing the kind of object (e.g., "question", "option",
                                       "reg_no_char") and its relevant identifier (e.g., question number,
                                       full name for characters).
        """
        # Questions and options
        if re.match(r'^question_\d+$', name):
            return "question", int(name.split('_')[1])
        elif re.match(r'^\d+[A-D]$', name):
            return "option", int(re.match(r'^(\d+)', name).group(1))

        # Registration number characters
        elif name.startswith("reg_no") and name != "reg_no":
            return "reg_no_char", name
        elif name == "reg_no":
            return "reg_no_main", name

        # Roll number characters
        elif name.startswith("roll_no") and name != "roll_no":
            return "roll_no_char", name
        elif name == "roll_no":
            return "roll_no_main", name

        # Booklet number characters
        elif name.startswith("booklet_no") and name != "booklet_no":
            return "booklet_no_char", name
        elif name == "booklet_no":
            return "booklet_no_main", name
        
        elif "anchor" in name:
            return "anchor", name

        else:
            return None, None

    def _get_mean_intensity(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> float:
        """
        Calculates the mean pixel intensity within a given bounding box in an image.

        Args:
            image (np.ndarray): The input image (OpenCV format).
            bbox (Tuple[float, float, float, float]): Bounding box coordinates (x1, y1, x2, y2).

        Returns:
            float: The mean intensity of the ROI, or 0 if the bounding box is invalid.
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1: # Check for invalid bounding box
            print(f"[WARNING] Invalid bounding box coordinates: {bbox}. Returning 0 mean intensity.")
            return 0.0

        roi = image[y1:y2, x1:x2]
        # For grayscale images, mean() works directly. For color, convert to grayscale first.
        if len(roi.shape) == 3: # If it's a color image
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        if roi.size == 0: # Check if ROI is empty after clamping
            print(f"[WARNING] Empty ROI after clamping for bbox: {bbox}. Returning 0 mean intensity.")
            return 0.0

        return np.mean(roi)

    def _detect_marked_bubble(self, image: np.ndarray, bubble_group_data: Dict[str, Any],
                             anchor_x: float, anchor_y: float) -> List[str]:
        """
        Detects marked bubbles within a group using dynamic thresholding.

        Args:
            image (np.ndarray): The image to analyze (e.g., test_image).
            bubble_group_data (Dict[str, Any]): A dictionary containing relative bounding box
                                                data for a group of bubbles (e.g., options for a question,
                                                digits for a column).
            anchor_x (float): The x-coordinate of the transformed anchor_1 center in the test image.
            anchor_y (float): The y-coordinate of the transformed anchor_1 center in the test image.

        Returns:
            List[str]: A list of names of the detected marked bubbles.
        """
        intensities = {}
        relative_intensities = {}
        all_relative_intensity_values = []
        marked_bubbles = []

        # Calculate absolute bounding boxes and mean intensities for each bubble
        for bubble_name, bubble_rel_data in bubble_group_data.items():
            bbox_rel = bubble_rel_data["bbox"]
            x1_abs = int(anchor_x + bbox_rel["x1"])
            y1_abs = int(anchor_y + bbox_rel["y1"])
            x2_abs = int(anchor_x + bbox_rel["x2"])
            y2_abs = int(anchor_y + bbox_rel["y2"])

            current_bbox = (x1_abs, y1_abs, x2_abs, y2_abs)
            mean_intensity = self._get_mean_intensity(image, current_bbox)
            intensities[bubble_name] = mean_intensity

        total_intensity = sum(intensities.values())

        # Determine relative intensities and collect them
        if total_intensity > 0:
            for bubble_name, mean_intensity in intensities.items():
                relative_intensity = (mean_intensity / total_intensity) * 100
                relative_intensities[bubble_name] = relative_intensity
                all_relative_intensity_values.append(relative_intensity)
                # print(f"    [INFO] {bubble_name}: Mean Intensity = {mean_intensity:.2f}, Relative Intensity = {relative_intensity:.2f}%")

            # --- DYNAMIC THRESHOLD CALCULATION AND MARKING ---
            if len(all_relative_intensity_values) > 1: # Need at least two values for min and max
                min_val = min(all_relative_intensity_values)
                max_val = max(all_relative_intensity_values)
                
                # Check for uniform values to avoid division by zero or nonsensical threshold
                if max_val == min_val:
                    print(f"[WARNING] All relative intensities are the same ({max_val:.2f}%). Cannot calculate dynamic threshold for this group.")
                    # Fallback or specific logic for uniform intensities
                    # For OMR bubbles, uniform intensity usually means no clear mark.
                    return [] 
                
                dynamic_threshold = (((min_val + max_val) / 2) - 2)
                # print(f"    [INFO] Dynamic Threshold for this group: ((({min_val:.2f} + {max_val:.2f}) / 2) - 2) = {dynamic_threshold:.2f}")

                # Identify marked bubbles based on dynamic threshold (for multiple elements)
                for bubble_name, relative_intensity in relative_intensities.items():
                    if relative_intensity < dynamic_threshold: # Compare with dynamic threshold
                        marked_bubbles.append(bubble_name)
            else: # This branch covers len(all_relative_intensity_values) is 0 or 1
                if len(all_relative_intensity_values) == 1:
                    # If there's only one element, use absolute mean intensity threshold
                    single_bubble_name = list(intensities.keys())[0]
                    single_bubble_mean_intensity = intensities[single_bubble_name]
                    print(f"    [INFO] Only one element in group '{single_bubble_name}'. Checking actual mean intensity against threshold 190.")
                    if single_bubble_mean_intensity < 190: # Directly check against the 190 threshold
                        marked_bubbles.append(single_bubble_name)
                        print(f"    [INFO] {single_bubble_name}: Marked as < 190 threshold.")
                    else:
                        print(f"    [INFO] {single_bubble_name}: Not marked as >= 190 threshold.")
                else: # len(all_relative_intensity_values) == 0 (no bubbles in group)
                    print(f"    [WARNING] No elements to process within this group.")
        else:
            print(f"    [WARNING] Total intensity is zero for this group, no bubbles can be marked.")

        return marked_bubbles

    # def _detect_marked_bubble(self, image: np.ndarray, bubble_group_data: Dict[str, Any],
    #                          anchor_x: float, anchor_y: float) -> List[str]:
    #     """
    #     Detects marked bubbles within a group using a fixed mean intensity threshold.

    #     Args:
    #         image (np.ndarray): The image to analyze (e.g., test_image).
    #         bubble_group_data (Dict[str, Any]): A dictionary containing relative bounding box
    #                                             data for a group of bubbles (e.g., options for a question,
    #                                             digits for a column).
    #         anchor_x (float): The x-coordinate of the transformed anchor_1 center in the test image.
    #         anchor_y (float): The y-coordinate of the transformed anchor_1 center in the test image.

    #     Returns:
    #         List[str]: A list of names of the detected marked bubbles.
    #     """
    #     marked_bubbles = []
    #     FIXED_INTENSITY_THRESHOLD = 190 # Set the fixed threshold here

    #     # Calculate absolute bounding boxes and mean intensities for each bubble
    #     for bubble_name, bubble_rel_data in bubble_group_data.items():
    #         bbox_rel = bubble_rel_data["bbox"]
    #         x1_abs = int(anchor_x + bbox_rel["x1"])
    #         y1_abs = int(anchor_y + bbox_rel["y1"])
    #         x2_abs = int(anchor_x + bbox_rel["x2"])
    #         y2_abs = int(anchor_y + bbox_rel["y2"])

    #         current_bbox = (x1_abs, y1_abs, x2_abs, y2_abs)
    #         mean_intensity = self._get_mean_intensity(image, current_bbox)
            
    #         # Print the mean intensity for each bubble
    #         print(f"    [DEBUG] {bubble_name}: Mean Intensity = {mean_intensity:.2f}")

    #         # Directly compare mean intensity with the fixed threshold
    #         # A lower mean intensity indicates a darker (marked) bubble
    #         if mean_intensity < FIXED_INTENSITY_THRESHOLD:
    #             marked_bubbles.append(bubble_name)
    #             # print(f"    [INFO] {bubble_name}: Mean Intensity = {mean_intensity:.2f} (MARKED)")
    #         # else:
    #             # print(f"    [INFO] {bubble_name}: Mean Intensity = {mean_intensity:.2f} (NOT MARKED)")

    #     return marked_bubbles

    def create_template(self, base_template_dir: str) -> str:
        """
        Creates a template JSON file by automatically finding the image, label, and class files
        within the given base directory structure.

        Args:
            base_template_dir (str): The base directory for the template (e.g., "D:/OMR_DEV/dataset/HS/").
                                     It expects 'images' and 'labels' subdirectories, and a 'classes.txt'
                                     directly in the base_template_dir.

        Returns:
            str: The path to the created template JSON file.
        """
        # --- Locate Template Image ---
        images_dir = os.path.join(base_template_dir, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"[ERROR] 'images' directory not found in base template path: {images_dir}")
        
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(images_dir, f))]
        
        if not image_files:
            raise FileNotFoundError(f"[ERROR] No image files found in: {images_dir}")
        if len(image_files) > 1:
            print(f"[WARNING] More than one image found in {images_dir}. Using the first one: {image_files[0]}")
        
        template_image_path = os.path.join(images_dir, image_files[0])
        print(f"[INFO] Detected template image path: {template_image_path}")


        # --- Locate Label File ---
        labels_dir = os.path.join(base_template_dir, "labels")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"[ERROR] 'labels' directory not found in base template path: {labels_dir}")

        label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt') and os.path.isfile(os.path.join(labels_dir, f))]
        
        if not label_files:
            raise FileNotFoundError(f"[ERROR] No label (.txt) files found in: {labels_dir}")
        if len(label_files) > 1:
            print(f"[WARNING] More than one label (.txt) file found in {labels_dir}. Using the first one: {label_files[0]}")
        
        label_file_path = os.path.join(labels_dir, label_files[0])
        print(f"[INFO] Detected label file path: {label_file_path}")


        # --- Locate Class File ---
        class_files = [f for f in os.listdir(base_template_dir) if f.lower() == 'classes.txt' and os.path.isfile(os.path.join(base_template_dir, f))]
        
        if not class_files:
            raise FileNotFoundError(f"[ERROR] 'classes.txt' file not found directly in base template path: {base_template_dir}")
        if len(class_files) > 1:
            print(f"[WARNING] More than one 'classes.txt' file found in {base_template_dir}. Using the first one: {class_files[0]}")
        
        class_file_path = os.path.join(base_template_dir, class_files[0])
        print(f"[INFO] Detected class file path: {class_file_path}")

        # --- Proceed with existing template creation logic using the discovered paths ---
        self.template_image = cv2.imread(template_image_path)
        if self.template_image is None:
            raise FileNotFoundError(f"[ERROR] Template image could not be loaded from: {template_image_path}")
        print("[INFO] Template image loaded successfully.")

        self.template_name = os.path.basename(template_image_path)
        self.template_name = os.path.splitext(self.template_name)[0]
        print(f"[INFO] Template image name: {self.template_name}")

        # Load the labels (assuming YOLO format)
        try:
            with open(label_file_path, 'r') as f:
                for line in f:
                    self.labels.append(line.strip().split())
            print("[INFO] Labels loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"[ERROR] Label file not found at: {label_file_path}")

        # Load the class names
        try:
            with open(class_file_path, 'r') as f:
                for line in f:
                    self.classes.append(line.strip())
            print("[INFO] Classes loaded successfully:", self.classes)
            print("[INFO] Total number of classes:", len(self.classes))
        except FileNotFoundError:
            raise FileNotFoundError(f"[ERROR] Class file not found at: {class_file_path}")

        # Identify anchor boxes from the labels and extract their coordinates
        image_height, image_width = self.template_image.shape[:2]

        for label in self.labels:
            class_id_str, x_center_norm_str, y_center_norm_str, width_norm_str, height_norm_str = label
            class_id = int(class_id_str)
            
            if class_id >= len(self.classes) or class_id < 0:
                print(f"[WARNING] Class ID {class_id} out of bounds for classes list. Skipping label: {label}")
                continue

            class_name = self.classes[class_id]

            # Convert normalized coordinates to pixel coordinates
            x_center = float(x_center_norm_str) * image_width
            y_center = float(y_center_norm_str) * image_height
            width = float(width_norm_str) * image_width
            height = float(height_norm_str) * image_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            center_x = int(x_center)
            center_y = int(y_center)
            
            # Store all object centers and boxes regardless of being an anchor
            self.object_centers[class_name] = (center_x, center_y)
            self.object_boxes[class_name] = (x1, y1, x2, y2)

            if ANCHOR_NAME in class_name: # Check if it's any anchor, then assign to generic anchor_boxes structure
                self.anchor_boxes[class_name] = {
                    "bounding_box": (x1, y1, x2, y2),
                    "center": (center_x, center_y)
                }
                print(f"[INFO] Detected anchor '{class_name}'. Bounding Box: ({x1}, {y1}, {x2}, {y2}), Center Point: ({center_x}, {center_y})")


        # Draw the anchor boxes and their center points on the image (for visualization during template creation)
        image_with_anchors = self.template_image.copy()
        for class_name, coords in self.anchor_boxes.items():
            x1, y1, x2, y2 = coords["bounding_box"]
            center_x, center_y = coords["center"]

            cv2.rectangle(image_with_anchors, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_anchors, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image_with_anchors, (center_x, center_y), 5, (0, 0, 255), -1)

            # print(f"[INFO] {class_name} Bounding Box: ({x1}, {y1}, {x2}, {y2})") # Already printed above
            # print(f"[INFO] {class_name} Center Point: ({center_x}, {center_y})") # Already printed above

        # Uncomment the following line if running in Google Colab for visualization
        # from google.colab.patches import cv2_imshow
        # cv2_imshow(image_with_anchors)
        # print("[INFO] Displaying template image with detected anchor boxes.")


        # Step 3: Extract anchor_1 as reference
        if ANCHOR_NAME not in self.object_centers:
            raise ValueError(f"[ERROR] Required anchor '{ANCHOR_NAME}' not found in labels. Cannot create template.")
        
        self.anchor_center = self.object_centers[ANCHOR_NAME]
        print(f"[INFO] Reference anchor '{ANCHOR_NAME}' center: {self.anchor_center}")

        # Step 4: Build relative data structure
        json_data = {
            "questions": {},
            "reg_no": {},
            "roll_no": {},
            "booklet_no": {}
        }

        for name in self.object_centers:
            # Skip the main anchor for relative calculations as everything is relative to it
            if name == ANCHOR_NAME:
                continue

            kind, identifier = self._parse_class_name(name)
            if kind is None or kind == "anchor": # Skip other anchors like anchor_2, anchor_3
                continue

            cx, cy = self.object_centers[name]
            x1, y1, x2, y2 = self.object_boxes[name]

            rel = {
                "center": {
                    "dx": cx - self.anchor_center[0],
                    "dy": cy - self.anchor_center[1]
                },
                "bbox": {
                    "x1": x1 - self.anchor_center[0],
                    "y1": y1 - self.anchor_center[1],
                    "x2": x2 - self.anchor_center[0],
                    "y2": y2 - self.anchor_center[1]
                }
            }

            if kind == "question":
                qnum = identifier
                if qnum not in json_data["questions"]:
                    json_data["questions"][qnum] = { "question": {}, "options": {} }
                json_data["questions"][qnum]["question"] = rel
            elif kind == "option":
                qnum = identifier
                if qnum not in json_data["questions"]:
                    json_data["questions"][qnum] = { "question": {}, "options": {} }
                json_data["questions"][qnum]["options"][name] = rel
            elif kind in ["reg_no_char", "reg_no_main"]:
                json_data["reg_no"][identifier] = rel
            elif kind in ["roll_no_char", "roll_no_main"]:
                json_data["roll_no"][identifier] = rel
            elif kind in ["booklet_no_char", "booklet_no_main"]:
                json_data["booklet_no"][identifier] = rel
        
        self.template_data = json_data # Store the created template data

        # Step 5: Save relative structure to JSON
        output_json_path = os.path.join(self.temp_directory, f"{self.template_name}_template.json")
        with open(output_json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"[INFO] Saved relative template data to: {output_json_path}")
        return output_json_path

    def process_test_image(self, test_image_path: str, template_json_path: str) -> Dict[str, Any]:
        """
        Processes a test OMR image using a pre-generated template JSON.

        Args:
            test_image_path (str): Path to the test OMR image.
            template_json_path (str): Path to the template JSON file created by create_template.

        Returns:
            Dict[str, Any]: A dictionary containing the detected answers and numbers.
        """
        # Load the template data
        try:
            with open(template_json_path, "r") as f:
                self.template_data = json.load(f)
            print(f"[INFO] Loaded template data from: {template_json_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"[ERROR] Template JSON file not found at: {template_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"[ERROR] Could not decode JSON from file: {template_json_path}")

        # Test image details
        self.test_image = cv2.imread(test_image_path)
        if self.test_image is None:
            raise FileNotFoundError(f"[ERROR] Test image not found at: {test_image_path}")
        print("[INFO] Test image loaded successfully.")

        self.test_name = os.path.basename(test_image_path)
        self.test_name = os.path.splitext(self.test_name)[0]
        print(f"[INFO] Test image name: {self.test_name}")

        # Step 2: Find anchor_1 using ORB + Homography
        orb = cv2.ORB_create(5000)
        
        if self.template_image is None:
             raise ValueError("[ERROR] Template image not loaded. Run create_template first or ensure it's available.")
        
        kp1, des1 = orb.detectAndCompute(self.template_image, None)
        kp2, des2 = orb.detectAndCompute(self.test_image, None)

        if des1 is None or des2 is None:
            print("[WARNING] Could not detect descriptors from one or both images. This might indicate poor image quality or lack of features.")
            if des1 is None and des2 is None:
                raise ValueError("[ERROR] No descriptors detected from template and test images. Homography not possible.")
            elif des1 is None:
                raise ValueError("[ERROR] No descriptors detected from template image. Homography not possible.")
            else: # des2 is None
                raise ValueError("[ERROR] No descriptors detected from test image. Homography not possible.")


        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Ensure there are enough descriptors for matching
        if len(kp1) < 2 or len(kp2) < 2:
            print(f"[WARNING] Not enough keypoints detected (Template: {len(kp1)}, Test: {len(kp2)}) in one or both images for robust matching.")
            raise Exception("[ERROR] Insufficient keypoints for homography calculation.")

        # Handle cases where knnMatch might fail if des1 or des2 are empty arrays
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except cv2.error as e:
            print(f"[ERROR] OpenCV matching error: {e}. This often happens if descriptors are empty or malformed.")
            raise Exception("[ERROR] Failed to perform feature matching. Check image content or descriptor validity.")


        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print(f"[INFO] Found {len(good)} good matches for homography.")

        # Homography
        if len(good) > 10: # A common heuristic for reliable homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                raise Exception("[ERROR] Homography matrix could not be computed. Insufficient good matches or highly distorted images.")

            # Ensure object_centers has the anchor_name before attempting transformation
            if ANCHOR_NAME not in self.object_centers:
                 raise ValueError(f"[ERROR] '{ANCHOR_NAME}' center not available. Template might not have been created correctly or anchor is missing in labels.")

            anchor_pt = np.array([[self.object_centers[ANCHOR_NAME]]], dtype=np.float32)
            transformed_anchor = cv2.perspectiveTransform(anchor_pt, M)
            self.transformed_center = tuple(map(int, transformed_anchor[0][0]))
            print(f"[INFO] Transformed '{ANCHOR_NAME}' center in test image: {self.transformed_center}")

            # Calculate the bounding box in the test image based on the transformed center
            # Using the relative offsets from the template anchor box
            template_anchor_bbox = self.object_boxes[ANCHOR_NAME]
            template_anchor_center = self.object_centers[ANCHOR_NAME]

            dx1 = template_anchor_bbox[0] - template_anchor_center[0]
            dy1 = template_anchor_bbox[1] - template_anchor_center[1]
            dx2 = template_anchor_bbox[2] - template_anchor_center[0]
            dy2 = template_anchor_bbox[3] - template_anchor_center[1]

            transformed_x1 = int(self.transformed_center[0] + dx1)
            transformed_y1 = int(self.transformed_center[1] + dy1)
            transformed_x2 = int(self.transformed_center[0] + dx2)
            transformed_y2 = int(self.transformed_center[1] + dy2)

            self.transformed_bbox = (transformed_x1, transformed_y1, transformed_x2, transformed_y2)
            print(f"[INFO] Transformed '{ANCHOR_NAME}' Bounding Box in test image: {self.transformed_bbox}")

            # Draw the bounding box and center on the test image for visualization
            self.result_image = self.test_image.copy()
            cv2.rectangle(self.result_image, (transformed_x1, transformed_y1), (transformed_x2, transformed_y2), (0, 255, 0), 2)
            cv2.circle(self.result_image, self.transformed_center, 5, (0, 0, 255), -1)
            cv2.putText(self.result_image, ANCHOR_NAME, (transformed_x1, transformed_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Uncomment the following line if running in Google Colab for visualization
            # from google.colab.patches import cv2_imshow
            # cv2_imshow(self.result_image)
            # print(f"[INFO] Displaying test image with transformed '{ANCHOR_NAME}'.")

        else:
            raise Exception(f"[ERROR] Not enough good matches found ({len(good)}) for homography. Minimum 10 required to proceed with OMR detection.")

        anchor_x, anchor_y = self.transformed_center

        # Process and detect marked bubbles for different groups
        self.detected_answers = {}

        # Questions
        print("\n[INFO] Processing Questions:")
        for qnum_str in sorted(self.template_data["questions"].keys(), key=int):
            qdata = self.template_data["questions"][qnum_str]
            if "options" in qdata and qdata["options"]:
                print(f"  [INFO] Processing Question {qnum_str} Options:")
                marked_options = self._detect_marked_bubble(self.result_image, qdata["options"], anchor_x, anchor_y)
                if marked_options:
                    print(f"  [OUTPUT] Question {qnum_str} Answer(s): {', '.join(marked_options)}")
                    self.detected_answers[f"question_{qnum_str}"] = marked_options
                else:
                    print(f"  [INFO] Question {qnum_str}: No clear marked option detected.")
                    self.detected_answers[f"question_{qnum_str}"] = None

        # Registration Number
        reg_no_chars_groups = {}
        for name, data in self.template_data["reg_no"].items():
            if name.startswith("reg_no_") and name != "reg_no":
                match = re.match(r'reg_no_(\d+)_(\d+)', name)
                if match:
                    group_index = int(match.group(1))
                    if group_index not in reg_no_chars_groups:
                        reg_no_chars_groups[group_index] = {}
                    reg_no_chars_groups[group_index][name] = data

        print("\n[INFO] Processing Registration Number:")
        detected_reg_no_list = []
        # Ensure processing happens in column order
        for group_index in sorted(reg_no_chars_groups.keys()):
            print(f"  [INFO] Processing reg_no_column_{group_index}:")
            marked_chars = self._detect_marked_bubble(self.result_image, reg_no_chars_groups[group_index], anchor_x, anchor_y)

            column_digits = []
            if marked_chars:
                # Sort marked_chars to ensure digits are in order if multiple are marked
                for marked_char in sorted(marked_chars):
                    digit_match = re.search(r'\_(\d+)$', marked_char)
                    if digit_match:
                        print(f"    [INFO] Detected digit: {digit_match.group(1)}")
                        column_digits.append(digit_match.group(1))
                    else:
                        print(f"[WARNING] Could not extract digit from marked char name: {marked_char}. Appending '?'")
                        column_digits.append("?")

            if column_digits:
                detected_reg_no_list.append("".join(column_digits))
            else:
                detected_reg_no_list.append("-")

        final_reg_no = "".join(detected_reg_no_list)
        if all(char == '-' for char in final_reg_no) or not final_reg_no: # Also handle empty string if no columns
            self.detected_answers["reg_no"] = None
        else:
            self.detected_answers["reg_no"] = final_reg_no
        print(f"[OUTPUT] Detected Registration Number: {self.detected_answers['reg_no']}")


        # Roll Number
        roll_no_chars_groups = {}
        for name, data in self.template_data["roll_no"].items():
            if name.startswith("roll_no_") and name != "roll_no":
                match = re.match(r'roll_no_(\d+)_(\d+)', name)
                if match:
                    group_index = int(match.group(1))
                    if group_index not in roll_no_chars_groups:
                        roll_no_chars_groups[group_index] = {}
                    roll_no_chars_groups[group_index][name] = data

        print("\n[INFO] Processing Roll Number:")
        detected_roll_no_list = []
        for group_index in sorted(roll_no_chars_groups.keys()):
            print(f"  [INFO] Processing roll_no_column_{group_index}:")
            marked_chars = self._detect_marked_bubble(self.result_image, roll_no_chars_groups[group_index], anchor_x, anchor_y)

            column_digits = []
            if marked_chars:
                for marked_char in sorted(marked_chars):
                    digit_match = re.search(r'\_(\d+)$', marked_char)
                    if digit_match:
                        print(f"    [INFO] Detected digit: {digit_match.group(1)}")
                        column_digits.append(digit_match.group(1))
                    else:
                        print(f"[WARNING] Could not extract digit from marked char name: {marked_char}. Appending '?'")
                        column_digits.append("?")

            if column_digits:
                detected_roll_no_list.append("".join(column_digits))
            else:
                detected_roll_no_list.append("-")

        final_roll_no = "".join(detected_roll_no_list)
        if all(char == '-' for char in final_roll_no) or not final_roll_no:
            self.detected_answers["roll_no"] = None
        else:
            self.detected_answers["roll_no"] = final_roll_no
        print(f"[OUTPUT] Detected Roll Number: {self.detected_answers['roll_no']}")

        # Booklet Number
        booklet_no_chars_groups = {}
        for name, data in self.template_data["booklet_no"].items():
            if name.startswith("booklet_no_") and name != "booklet_no":
                match = re.match(r'booklet_no_(\d+)_(\d+)', name)
                if match:
                    group_index = int(match.group(1))
                    if group_index not in booklet_no_chars_groups:
                        booklet_no_chars_groups[group_index] = {}
                    booklet_no_chars_groups[group_index][name] = data

        print("\n[INFO] Processing Booklet Number:")
        detected_booklet_no_list = []
        for group_index in sorted(booklet_no_chars_groups.keys()):
            print(f"  [INFO] Processing booklet_no_column_{group_index}:")
            marked_chars = self._detect_marked_bubble(self.result_image, booklet_no_chars_groups[group_index], anchor_x, anchor_y)

            column_digits = []
            if marked_chars:
                for marked_char in sorted(marked_chars):
                    digit_match = re.search(r'\_(\d+)$', marked_char)
                    if digit_match:
                        print(f"    [INFO] Detected digit: {digit_match.group(1)}")
                        column_digits.append(digit_match.group(1))
                    else:
                        print(f"[WARNING] Could not extract digit from marked char name: {marked_char}. Appending '?'")
                        column_digits.append("?")

            if column_digits:
                detected_booklet_no_list.append("".join(column_digits))
            else:
                detected_booklet_no_list.append("-")

        final_booklet_no = "".join(detected_booklet_no_list)
        if all(char == '-' for char in final_booklet_no) or not final_booklet_no:
            self.detected_answers["booklet_no"] = None
        else:
            self.detected_answers["booklet_no"] = final_booklet_no
        print(f"[OUTPUT] Detected Booklet Number: {self.detected_answers['booklet_no']}")

        print("\n--- [OUTPUT] Detected Answers Summary ---")
        for key, value in self.detected_answers.items():
            print(f"[OUTPUT] {key}: {value}")

        # Save detected answers to a JSON file
        output_answers_json = os.path.join(self.temp_directory, f"{self.test_name}_ouput.json")
        with open(output_answers_json, "w") as f:
            json.dump(self.detected_answers, f, indent=4)
        print(f"[INFO] Detected answers saved to: {output_answers_json}")

        return self.detected_answers

# --- Example Usage (similar to original flow, but now using the class) ---
if __name__ == "__main__":
    omr_processor = OMRProcessor(temp_directory=TEMP_DIR)

    # --- Step 1: Create Template ---
    # Prompt user for the base directory of the template
    BASE_TEMPLATE_DIR_INPUT = input("Enter the base directory for the template (e.g., D:/OMR_DEV/dataset/HS/): ")
    # Ensure consistent path separators for different OS
    BASE_TEMPLATE_DIR_INPUT = BASE_TEMPLATE_DIR_INPUT.replace('\\', '/')
    if not BASE_TEMPLATE_DIR_INPUT.endswith('/'):
        BASE_TEMPLATE_DIR_INPUT += '/'

    try:
        template_output_json = omr_processor.create_template(
            base_template_dir=BASE_TEMPLATE_DIR_INPUT
        )
        print(f"\n[INFO] Template creation complete. Template saved at: {template_output_json}")
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"[ERROR] Template creation failed: {e}")
        exit() # Exit if template creation fails, as subsequent steps depend on it

    # --- Step 2: Process a Test Image ---
    TEST_IMAGE_PATH_INPUT = input("Enter test image path: ")
    # Ensure consistent path separators for different OS
    TEST_IMAGE_PATH_INPUT = TEST_IMAGE_PATH_INPUT.replace('\\', '/')

    try:
        detected_results = omr_processor.process_test_image(
            test_image_path=TEST_IMAGE_PATH_INPUT,
            template_json_path=template_output_json
        )
        print("\n[INFO] Test image processing complete.")
        print(f"[OUTPUT] Final detected answers: {detected_results}")
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"[ERROR] Test image processing failed: {e}")

    # Optional: Display the final result image (if not in Colab, use cv2.imshow etc.)
    # if omr_processor.result_image is not None:
    #     cv2_imshow(omr_processor.result_image)
    #     print("[INFO] Displaying final result image with detected marks.")