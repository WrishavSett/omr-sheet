import os
import re
import cv2
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any

# --- Global Configurations / Constants ---
TEMP_DIR = "D:\\Wrishav\\omr-sheet\\temp"
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
                
                dynamic_threshold = (((min_val + max_val) / 2) - 1)
                # print(f"    [INFO] Dynamic Threshold for this group: ((({min_val:.2f} + {max_val:.2f}) / 2) - 1) = {dynamic_threshold:.2f}")

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

    def _create_template(self, base_template_dir: str) -> str:
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
            "template_metadata": {
                "base_template_dir": base_template_dir,
                "template_image_path": template_image_path,
                "label_file_path": label_file_path,
                "class_file_path": class_file_path
            },
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

    def _process_single_image_detection(self, test_image_path: str) -> Dict[str, Any]:
        """
        Processes a single test OMR image using the pre-loaded template data.
        This is a helper function for batch processing.

        Args:
            test_image_path (str): Path to the test OMR image.

        Returns:
            Dict[str, Any]: A dictionary containing the detected answers and numbers for this image.
        """
        self.test_image = cv2.imread(test_image_path)
        if self.test_image is None:
            print(f"[WARNING] Test image not found or could not be loaded at: {test_image_path}. Skipping this image.")
            return {"error": f"Image not found or loaded: {os.path.basename(test_image_path)}"}
        print(f"[INFO] Test image loaded successfully: {os.path.basename(test_image_path)}")

        self.test_name = os.path.basename(test_image_path)
        self.test_name = os.path.splitext(self.test_name)[0]
        # print(f"[INFO] Processing test image: {self.test_name}") # Moved to batch loop

        # Step 2: Find anchor_1 using ORB + Homography
        orb = cv2.ORB_create(5000)
        
        if self.template_image is None:
             raise ValueError("[ERROR] Template image not loaded. Run _create_template first or ensure it's available.")
        
        kp1, des1 = orb.detectAndCompute(self.template_image, None)
        kp2, des2 = orb.detectAndCompute(self.test_image, None)

        if des1 is None or des2 is None:
            print(f"[WARNING] Could not detect descriptors from image {os.path.basename(test_image_path)}. This might indicate poor image quality or lack of features. Skipping.")
            return {"error": f"No descriptors for {os.path.basename(test_image_path)}"}

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        if len(kp1) < 2 or len(kp2) < 2:
            print(f"[WARNING] Not enough keypoints detected (Template: {len(kp1)}, Test: {len(kp2)}) in image {os.path.basename(test_image_path)} for robust matching. Skipping.")
            return {"error": f"Insufficient keypoints for {os.path.basename(test_image_path)}"}

        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except cv2.error as e:
            print(f"[ERROR] OpenCV matching error for {os.path.basename(test_image_path)}: {e}. Skipping.")
            return {"error": f"Matching failed for {os.path.basename(test_image_path)}"}


        # Ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print(f"[INFO] Found {len(good)} good matches for homography in {os.path.basename(test_image_path)}.")

        if len(good) > 10: # A common heuristic for reliable homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                print(f"[WARNING] Homography matrix could not be computed for {os.path.basename(test_image_path)}. Skipping.")
                return {"error": f"Homography failed for {os.path.basename(test_image_path)}"}

            if ANCHOR_NAME not in self.object_centers:
                 raise ValueError(f"[ERROR] '{ANCHOR_NAME}' center not available. Template might not have been created correctly or anchor is missing in labels.")

            anchor_pt = np.array([[self.object_centers[ANCHOR_NAME]]], dtype=np.float32)
            transformed_anchor = cv2.perspectiveTransform(anchor_pt, M)
            self.transformed_center = tuple(map(int, transformed_anchor[0][0]))
            print(f"[INFO] Transformed '{ANCHOR_NAME}' center in {os.path.basename(test_image_path)}: {self.transformed_center}")

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
            print(f"[INFO] Transformed '{ANCHOR_NAME}' Bounding Box in {os.path.basename(test_image_path)}: {self.transformed_bbox}")

            self.result_image = self.test_image.copy()
            cv2.rectangle(self.result_image, (transformed_x1, transformed_y1), (transformed_x2, transformed_y2), (0, 255, 0), 2)
            cv2.circle(self.result_image, self.transformed_center, 5, (0, 0, 255), -1)
            cv2.putText(self.result_image, ANCHOR_NAME, (transformed_x1, transformed_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            print(f"[WARNING] Not enough good matches found ({len(good)}) for homography in {os.path.basename(test_image_path)}. Skipping OMR detection for this image.")
            return {"error": f"Homography failed for {os.path.basename(test_image_path)}"}

        anchor_x, anchor_y = self.transformed_center
        current_image_detected_answers = {} # Store results for this specific image

        # Process and detect marked bubbles for different groups
        # Questions
        # print("\n[INFO] Processing Questions:")
        for qnum_str in sorted(self.template_data["questions"].keys(), key=int):
            qdata = self.template_data["questions"][qnum_str]
            if "options" in qdata and qdata["options"]:
                # print(f"  [INFO] Processing Question {qnum_str} Options:")
                marked_options = self._detect_marked_bubble(self.result_image, qdata["options"], anchor_x, anchor_y)
                if marked_options:
                    # print(f"  [OUTPUT] Question {qnum_str} Answer(s): {', '.join(marked_options)}")
                    current_image_detected_answers[f"question_{qnum_str}"] = marked_options
                else:
                    # print(f"  [INFO] Question {qnum_str}: No clear marked option detected.")
                    current_image_detected_answers[f"question_{qnum_str}"] = None

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

        # print("\n[INFO] Processing Registration Number:")
        detected_reg_no_list = []
        for group_index in sorted(reg_no_chars_groups.keys()):
            # print(f"  [INFO] Processing reg_no_column_{group_index}:")
            marked_chars = self._detect_marked_bubble(self.result_image, reg_no_chars_groups[group_index], anchor_x, anchor_y)

            column_digits = []
            if marked_chars:
                for marked_char in sorted(marked_chars):
                    digit_match = re.search(r'\_(\d+)$', marked_char)
                    if digit_match:
                        # print(f"    [INFO] Detected digit: {digit_match.group(1)}")
                        column_digits.append(digit_match.group(1))
                    else:
                        # print(f"[WARNING] Could not extract digit from marked char name: {marked_char}. Appending '?'")
                        column_digits.append("?")

            if column_digits:
                detected_reg_no_list.append("".join(column_digits))
            else:
                detected_reg_no_list.append("-")

        final_reg_no = "".join(detected_reg_no_list)
        if all(char == '-' for char in final_reg_no) or not final_reg_no:
            current_image_detected_answers["reg_no"] = None
        else:
            current_image_detected_answers["reg_no"] = final_reg_no
        # print(f"[OUTPUT] Detected Registration Number: {current_image_detected_answers['reg_no']}")


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

        # print("\n[INFO] Processing Roll Number:")
        detected_roll_no_list = []
        for group_index in sorted(roll_no_chars_groups.keys()):
            # print(f"  [INFO] Processing roll_no_column_{group_index}:")
            marked_chars = self._detect_marked_bubble(self.result_image, roll_no_chars_groups[group_index], anchor_x, anchor_y)

            column_digits = []
            if marked_chars:
                for marked_char in sorted(marked_chars):
                    digit_match = re.search(r'\_(\d+)$', marked_char)
                    if digit_match:
                        # print(f"    [INFO] Detected digit: {digit_match.group(1)}")
                        column_digits.append(digit_match.group(1))
                    else:
                        # print(f"[WARNING] Could not extract digit from marked char name: {marked_char}. Appending '?'")
                        column_digits.append("?")

            if column_digits:
                detected_roll_no_list.append("".join(column_digits))
            else:
                detected_roll_no_list.append("-")

        final_roll_no = "".join(detected_roll_no_list)
        if all(char == '-' for char in final_roll_no) or not final_roll_no:
            current_image_detected_answers["roll_no"] = None
        else:
            current_image_detected_answers["roll_no"] = final_roll_no
        # print(f"[OUTPUT] Detected Roll Number: {current_image_detected_answers['roll_no']}")

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

        # print("\n[INFO] Processing Booklet Number:")
        detected_booklet_no_list = []
        for group_index in sorted(booklet_no_chars_groups.keys()):
            # print(f"  [INFO] Processing booklet_no_column_{group_index}:")
            marked_chars = self._detect_marked_bubble(self.result_image, booklet_no_chars_groups[group_index], anchor_x, anchor_y)

            column_digits = []
            if marked_chars:
                for marked_char in sorted(marked_chars):
                    digit_match = re.search(r'\_(\d+)$', marked_char)
                    if digit_match:
                        # print(f"    [INFO] Detected digit: {digit_match.group(1)}")
                        column_digits.append(digit_match.group(1))
                    else:
                        # print(f"[WARNING] Could not extract digit from marked char name: {marked_char}. Appending '?'")
                        column_digits.append("?")

            if column_digits:
                detected_booklet_no_list.append("".join(column_digits))
            else:
                detected_booklet_no_list.append("-")

        final_booklet_no = "".join(detected_booklet_no_list)
        if all(char == '-' for char in final_booklet_no) or not final_booklet_no:
            current_image_detected_answers["booklet_no"] = None
        else:
            current_image_detected_answers["booklet_no"] = final_booklet_no
        # print(f"[OUTPUT] Detected Booklet Number: {current_image_detected_answers['booklet_no']}")

        return current_image_detected_answers

    def _process_image_batch(self, batch_dir: str, template_json_path: str) -> Dict[str, Any]:
        """
        Processes all OMR images within a specified batch directory using a pre-generated template JSON.

        Args:
            batch_dir (str): The path to the directory containing test OMR images.
            template_json_path (str): Path to the template JSON file created by _create_template.

        Returns:
            Dict[str, Any]: A dictionary containing the detected answers for the entire batch.
                            Structure: {"batch_name": "...", "processed_images": {"image_name": {...}, ...}}
        """
        # Load the template data once for the entire batch
        try:
            with open(template_json_path, "r") as f:
                self.template_data = json.load(f)
            print(f"[INFO] Loaded template data from: {template_json_path} for batch processing.")
        except FileNotFoundError:
            raise FileNotFoundError(f"[ERROR] Template JSON file not found at: {template_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"[ERROR] Could not decode JSON from file: {template_json_path}")

        # Ensure the template image is loaded (it should be from _create_template, but check)
        # This is important for homography in _process_single_image_detection
        if self.template_image is None:
            if "template_metadata" in self.template_data and "template_image_path" in self.template_data["template_metadata"]:
                template_img_path_from_meta = self.template_data["template_metadata"]["template_image_path"]
                self.template_image = cv2.imread(template_img_path_from_meta)
                if self.template_image is None:
                    raise FileNotFoundError(f"[ERROR] Template image from metadata could not be loaded: {template_img_path_from_meta}")
                print(f"[INFO] Template image re-loaded from metadata: {template_img_path_from_meta}")
            else:
                raise ValueError("[ERROR] Template image not available and its path not found in template metadata. Cannot perform homography.")


        if not os.path.isdir(batch_dir):
            raise FileNotFoundError(f"[ERROR] Batch directory not found: {batch_dir}")
        
        # --- FIX: Ensure batch_dir does not have a trailing slash for correct basename extraction ---
        # os.path.normpath will remove trailing separators and normalize the path
        cleaned_batch_dir = os.path.normpath(batch_dir)
        batch_name = os.path.basename(cleaned_batch_dir)
        # --- END FIX ---

        batch_results = {
            "batch_name": batch_name,
            "batch_full_path": cleaned_batch_dir,
            "processed_images": {}
        }
        print(f"\n[INFO] Starting batch processing for directory: {batch_dir} (Batch Name: {batch_name})")

        image_files_in_batch = [f for f in os.listdir(batch_dir) if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(batch_dir, f))]
        
        if not image_files_in_batch:
            print(f"[WARNING] No image files found in the batch directory: {batch_dir}")
            return batch_results

        for i, image_file in enumerate(sorted(image_files_in_batch)): # Sort for consistent order
            full_image_path = os.path.join(batch_dir, image_file)
            # image_base_name = os.path.splitext(image_file)[0] # This line is no longer needed for key
            
            print(f"\n--- [INFO] Processing image {i+1}/{len(image_files_in_batch)}: {image_file} ---")
            try:
                # Call the helper function for single image detection
                single_image_results = self._process_single_image_detection(full_image_path)
                batch_results["processed_images"][image_file] = single_image_results # MODIFIED: Use image_file (with extension) as key
                print(f"[OUTPUT] Summary for {image_file}: Q1: {single_image_results.get('question_1', 'N/A')}, RegNo: {single_image_results.get('reg_no', 'N/A')}, RollNo: {single_image_results.get('roll_no', 'N/A')}, BookletNo: {single_image_results.get('booklet_no', 'N/A')}")
            except Exception as e:
                print(f"[ERROR] Failed to process {image_file}: {e}")
                batch_results["processed_images"][image_file] = {"error": str(e)} # MODIFIED: Use image_file (with extension) as key

        # --- MODIFIED: Changed the output JSON file name ---
        output_batch_json_path = os.path.join(self.temp_directory, f"{batch_name}_output.json")
        # --- END MODIFIED ---
        with open(output_batch_json_path, "w") as f:
            json.dump(batch_results, f, indent=4)
        print(f"\n[INFO] Batch processing complete. All results saved to: {output_batch_json_path}")
        
        return batch_results

    # NEW FUNCTION: Method to generate a CSV report from batch results
    def _generate_csv_report(self, batch_results: Dict[str, Any], output_dir: str = "./"):
        """
        Generates a CSV file from the batch processing results.

        The CSV will have headers: batch name, image name, q1, q2, ..., qn,
        reg_no, roll_no, booklet_no.

        Args:
            batch_results (Dict[str, Any]): The dictionary containing the batch processing results.
            output_dir (str): The directory where the output CSV file will be saved.
                              Defaults to the temporary directory of the OMRProcessor.
        """
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f"[INFO] Created output directory: {output_dir}")

        batch_name = batch_results.get("batch_name", "UnknownBatch")
        processed_images = batch_results.get("processed_images", {})

        if not processed_images:
            print(f"[WARNING] No processed images found in the batch results for CSV generation.")
            return

        # Dynamically determine all possible question headers (q1, q2, ...)
        all_question_numbers = set()
        for image_results in processed_images.values():
            if isinstance(image_results, dict): # Ensure it's a dict, not an error string
                for key in image_results.keys():
                    if key.startswith("question_"):
                        try:
                            q_num = int(key.split('_')[1])
                            all_question_numbers.add(q_num)
                        except ValueError:
                            print(f"[WARNING] Skipping malformed question key: {key}")

        sorted_question_headers = [f"q{q_num}" for q_num in sorted(list(all_question_numbers))]

        # Define the fixed headers
        fixed_headers = ["batch name", "image name"]
        # Define the suffix headers
        suffix_headers = ["reg_no", "roll_no", "booklet_no"]

        # Combine all headers in the desired order
        csv_headers = fixed_headers + sorted_question_headers + suffix_headers

        # Prepare data for DataFrame
        rows_data = []
        for image_name, image_results in processed_images.items():
            row = {header: None for header in csv_headers} # Initialize row with None values

            row["batch name"] = batch_name
            row["image name"] = image_name

            if "error" in image_results:
                # If there was an error processing the image, mark all relevant fields as ERROR
                for header in sorted_question_headers + suffix_headers:
                    row[header] = "ERROR"
                row["image_error"] = image_results["error"] # Add specific error message
                print(f"[WARNING] Image '{image_name}' had an error: {image_results['error']}")
            else:
                # Populate question answers
                for q_num in sorted(list(all_question_numbers)):
                    q_key = f"question_{q_num}"
                    q_header = f"q{q_num}"
                    answer = image_results.get(q_key)
                    if isinstance(answer, list):
                        row[q_header] = ", ".join(answer) # Join multiple answers with comma
                    elif answer is None:
                        row[q_header] = "N/A" # Or "" for empty string
                    else:
                        row[q_header] = str(answer) # In case it's not a list but some other value

                # Populate registration, roll, and booklet numbers
                row["reg_no"] = image_results.get("reg_no", "N/A")
                row["roll_no"] = image_results.get("roll_no", "N/A")
                row["booklet_no"] = image_results.get("booklet_no", "N/A")
            
            rows_data.append(row)

        # Create DataFrame
        # Ensure columns are in the correct order as per csv_headers
        df = pd.DataFrame(rows_data, columns=csv_headers + (["image_error"] if any("error" in res for res in processed_images.values()) else []))

        # Define output CSV file path
        output_csv_filename = f"{batch_name}_results.csv"
        output_csv_path = os.path.join(output_dir, output_csv_filename)

        # Save DataFrame to CSV
        try:
            df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"[OUTPUT] Successfully generated CSV file: {output_csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV file to {output_csv_path}: {e}")

# --- Example Usage ---
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
        template_output_json_path = omr_processor._create_template(
            base_template_dir=BASE_TEMPLATE_DIR_INPUT
        )
        print(f"\n[INFO] Template creation complete. Template saved at: {template_output_json_path}")
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"[ERROR] Template creation failed: {e}")
        exit() # Exit if template creation fails, as subsequent steps depend on it

    # --- Step 2: Process a Batch of Test Images ---
    BATCH_DIR_INPUT = input("Enter the directory containing test images (batch folder, e.g., D:/OMR_DEV/test_batch/): ")
    # Ensure consistent path separators for different OS
    BATCH_DIR_INPUT = BATCH_DIR_INPUT.replace('\\', '/')
    # No longer force trailing slash for batch_dir, as normpath handles it internally for basename
    # if not BATCH_DIR_INPUT.endswith('/'):
    #     BATCH_DIR_INPUT += '/'

    try:
        all_batch_results = omr_processor._process_image_batch(
            batch_dir=BATCH_DIR_INPUT,
            template_json_path=template_output_json_path
        )
        print("\n[INFO] All test images in the batch have been processed.")
        
        # NEW CALL: Generate CSV report after batch processing
        # The output directory for the CSV will be the same as the TEMP_DIR
        omr_processor._generate_csv_report(all_batch_results, output_dir=omr_processor.temp_directory)
        # END NEW CALL

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"[ERROR] Batch image processing failed: {e}")

    # Optional: Display the final result image (if not in Colab, use cv2.imshow etc.)
    # Note: When processing a batch, self.result_image will only hold the last processed image's visualization.
    # If you need to see all, you'd save them during _process_single_image_detection.
    # if omr_processor.result_image is not None:
    #     cv2_imshow(omr_processor.result_image)
    #     print("[INFO] Displaying the last processed image with detected marks.")