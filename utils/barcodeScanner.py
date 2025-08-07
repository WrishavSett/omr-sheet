import cv2
from pyzbar.pyzbar import decode

def read_barcode_from_image(image_path):
    """
    Reads and returns barcode(s) from the given image after preprocessing.

    Parameters:
    image_path (str): Path to the image containing the barcode.

    Returns:
    list: A list of decoded barcode strings (could be empty if none found).
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return []

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to increase contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionally resize (enlarging helps for small barcodes)
    scale_percent = 200  # increase size by 200%
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)

    # Decode barcodes
    decoded_objects = decode(resized)

    if not decoded_objects:
        print("Info: No barcodes found in the image.")
        return []

    barcodes = []
    for obj in decoded_objects:
        barcode_data = obj.data.decode('utf-8')
        barcode_type = obj.type
        print(f"Output: Found {barcode_type} barcode: {barcode_data}")
        barcodes.append(barcode_data)

    return barcodes[0]

if __name__ == "__main__":
    base_path = "./ICR/barcode_no"
    image_list = ["BATCH05003.jpg", "BATCH05005.jpg", "BATCH05009.jpg"]
    for image_name in image_list:
        image_path = f"{base_path}/{image_name}"
        print(f"Image Path: {image_path}")
        barcodes = read_barcode_from_image(image_path)
        print(f"Barcodes found in {image_name}:", barcodes)