from modules.anchorDetection import process_batch
from modules.fieldMapping import process_field_mapping
from modules.markedOption import process_marked_options
from modules.runRequest import process_icr_requests
from modules.jsonUtils import json_restructure
import os, re, sys, json
import time

def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    
    start_time = time.time()

    # Require at least 3 positional args
    if len(sys.argv) < 4:
        print("Usage: python main.py <omr_template_name> <date> <batch_name> [--save-anchor] [--save-mapped] "
              "[--draw-bboxes] [--full-mark <value>] [--partial-mark <value>]")
        sys.exit(1)

    # Optional flags
    save_anchor_images = "--save-anchor" in sys.argv
    save_mapped_images = "--save-mapped" in sys.argv
    draw_bboxes = "--draw-bboxes" in sys.argv
    full_mark_threshold_pct = None
    partial_mark_threshold_pct = None

    if "--full-mark" in sys.argv:
        full_mark_threshold_pct = float(sys.argv[sys.argv.index("--full-mark") + 1])
    if "--partial-mark" in sys.argv:
        partial_mark_threshold_pct = float(sys.argv[sys.argv.index("--partial-mark") + 1])

    # first 3 arguments always required
    omr_template_name, date, batch_name = sys.argv[1:4]

    config = load_config()
    base_folder = config["base_folder"]

    # Anchor Detection for the batch
    # base_folder = "D:\\Projects\\OMR\\new_abhigyan\\Restructure"
    # key_fields.json = os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")
    results = process_batch(base_folder, omr_template_name, date, batch_name, save_anchor_images=save_anchor_images)
    print(f"|INFO| Batch processed. Images processed: {len(results)}")
    
    # Process field mapping for the batch
    # field_mapping.json = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "annotate_" + "batch_name", field_mappings.json)
    field_mapping_results = process_field_mapping(base_folder, omr_template_name, date, batch_name, save_mapped_images=save_mapped_images)
    print(f"|INFO| Field mapping completed. Mappings found: {len(field_mapping_results)}")

    # Process marked options for the batch
    marked_stats = process_marked_options(base_folder, omr_template_name, date, batch_name, draw_bboxes=draw_bboxes, 
                                          full_mark_threshold_pct=full_mark_threshold_pct, partial_mark_threshold_pct=partial_mark_threshold_pct)
    print(f"|INFO| Marked options processed. "
        f"Images: {marked_stats['processed_images']}, "
        f"Detected fields: {marked_stats['total_detected_fields']}")
    
    print(f"|INFO| Execution time before ICR: {time.time() - start_time:.2f} seconds")

    # Process ICR requests for the batch
    # batch_name.json = os.path.join(base_folder, "Images", omr_template_name, date, "Output", {batch_name}, f"{batch_name}.json")
    icr_stats = process_icr_requests(base_folder, omr_template_name, date, batch_name)
    print(f"|INFO| ICR processed images: {icr_stats['processed_images']}")
    
    
    # ----- Wrishav's utils.py usage -----
    
    json_restructure(base_folder, omr_template_name, date, batch_name)
    print(f"|INFO| JSON restructuring completed for {batch_name}.")

    print(f"|INFO| Total execution time: {time.time() - start_time:.2f} seconds")