# Preprocess wrist x-ray DICOM file

There are 4 images each for one fracture, Pre-reduction AP, Pre-reduction Lateral, Post-reduction AP, Post-reduction Lateral

Each view is located in their respective folder "Pre_AP", "Pre_LAT", "Post_AP", and "Post_LAT"

Bounding boxes annotated by two fellowship-trained orthopedic trauma surgeons was used to crop the image for training.

**Steps for image processing**
1. Read dicom file listed in a csv file using pydicom
2. Get pixel array, window width and length, rescale intercept and slope
3. Rescale the pixel array with the rescale intercept and slope
4. Resize pixel array to desired pixel spacing, which is 0.2 for both x and y axis
5. get bounding box from XML file in PASCAL VOC format (x_min, x_max_ y_min)
6. Calculate width of cropped image as x_max - x_min
7. Using different cropping algorithm for AP and Lateral images
8. Cropped to target size, pad zero if crop area is out of bound
9. save as new DICOM images in target folder for training


```python
import os
import cv2
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

def resize_pixel_array(pixel_array, pixel_spacing, target_spacing=0.2):
    """
    Resize the pixel array to achieve the target pixel spacing using cv2.
    """
    row_scale = pixel_spacing[0] / target_spacing
    col_scale = pixel_spacing[1] / target_spacing
    new_dimensions = (
        int(pixel_array.shape[1] * col_scale),  # New width
        int(pixel_array.shape[0] * row_scale),  # New height
    )
    resized_array = cv2.resize(pixel_array, new_dimensions, interpolation=cv2.INTER_LINEAR)
    return resized_array

def pad_and_crop_image(pixel_array, y_min, x_min, x_max, width, crop_type="default"):
    """
    Crop the image according to the adjusted bounding box logic.
    Pad with zeros if the crop exceeds the image boundaries.
    
    Parameters:
    - pixel_array: Input pixel array.
    - y_min, x_min, x_max: Bounding box coordinates.
    - width: Calculated width of the bounding box.
    - crop_type: "default" or "lat" for different cropping logic.
    
    Returns:
    - Cropped image.
    """
    if crop_type == "lat":  # For Pre_LAT and Post_LAT
        new_y_min = int(y_min)
        new_y_max = int(y_min + 1.3 * width)
        new_x_min = int(x_min - 0.15 * width)
        new_x_max = int(x_max + 0.15 * width)
    else:  # Default cropping for Pre_AP and Post_AP
        new_y_min = int(y_min)
        new_y_max = int(y_min + width)
        new_x_min = int(x_min)
        new_x_max = int(x_max)

    # Pad the array to ensure bounds are within limits
    height, width_orig = pixel_array.shape
    pad_top = max(0, -new_y_min)
    pad_bottom = max(0, new_y_max - height)
    pad_left = max(0, -new_x_min)
    pad_right = max(0, new_x_max - width_orig)

    # Apply padding
    padded_array = np.pad(
        pixel_array,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    # Adjust bounds based on padding
    new_y_min += pad_top
    new_y_max += pad_top
    new_x_min += pad_left
    new_x_max += pad_left

    # Perform cropping
    cropped_array = padded_array[new_y_min:new_y_max, new_x_min:new_x_max]

    return cropped_array

def process_dicom_and_save(row, input_folder, xml_folder, output_folder, subfolders):
    """
    Process each DICOM file, apply resizing, cropping, and save the result.
    """
    file_name = row["FileName"]
    id_date = row["id_date"]

    for subfolder in subfolders:
        dicom_path = os.path.join(input_folder, subfolder, file_name)
        xml_path = os.path.join(xml_folder, subfolder, file_name.replace(".dcm", ".xml"))
        output_path = os.path.join(output_folder, subfolder, f"{id_date}.dcm")

        try:
            # Read DICOM file
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array.astype(np.float64)
            pixel_spacing = ds.PixelSpacing
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            window_center = ds.WindowCenter
            window_width = ds.WindowWidth

            # Apply rescale slope and intercept
            pixel_array = pixel_array * slope

            # Resize pixel array
            resized_array = resize_pixel_array(pixel_array, pixel_spacing)

            # Add intercept to align with window center (keep float64 temporarily)
            resized_array += intercept

            # Adjust pixel array to ensure min value is 0
            array_min = resized_array.min()
            if array_min < 0:
                resized_array -= array_min
                if isinstance(window_center, (list, tuple)):
                    window_center = [wc - array_min for wc in window_center]
                else:
                    window_center -= array_min

            # Round and convert to uint16
            resized_array = np.round(resized_array).astype(np.uint16)

            # Read and process XML file for cropping
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bbox = root.find("object/bndbox")
            x_min = int(float(bbox.find("xmin").text))
            y_min = int(float(bbox.find("ymin").text))
            x_max = int(float(bbox.find("xmax").text))

            # Calculate crop width
            width = x_max - x_min

            # Determine crop type based on folder
            crop_type = "lat" if subfolder in ["Pre_LAT", "Post_LAT"] else "default"

            # Pad and crop
            cropped_array = pad_and_crop_image(resized_array, y_min, x_min, x_max, width, crop_type)

            # Update DICOM metadata
            ds.PixelData = cropped_array.tobytes()
            ds.Rows, ds.Columns = cropped_array.shape
            ds.PixelSpacing = [0.2, 0.2]
            ds.RescaleIntercept = 0  # Set intercept to 0 after adjustment
            ds.WindowCenter = window_center
            ds.WindowWidth = window_width

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save the modified DICOM
            ds.save_as(output_path)
            print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")


# Main logic
csv_path = "Placeholder.csv"
input_folder = "Placeholder_input_folder"
xml_folder = "Placeholder_xml_folder"
output_folder = "Placeholder_output_folder"
subfolders = ["Pre_AP", "Pre_LAT", "Post_AP", "Post_LAT"]

# Read CSV file
df = pd.read_csv(csv_path)

# Process each file
for _, row in df.iterrows():
    process_dicom_and_save(row, input_folder, xml_folder, output_folder, subfolders)
```
