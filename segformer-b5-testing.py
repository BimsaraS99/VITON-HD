from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import os
import time

# Load segmentation pipeline (SegFormer model fine-tuned on human parsing)
pipe = pipeline("image-segmentation", model="segformer-b5", device=-1)

start_time = time.time()

# Load input image (change filename if uploaded image has different name)
image_path = "00891_00.jpg"
image = Image.open(image_path).convert("RGB")

# Create output directory
output_dir = "masks_output"
os.makedirs(output_dir, exist_ok=True)

# Run segmentation
results = pipe(image)

# Save binary masks
for i, item in enumerate(results):
    label = item['label']
    mask = item['mask']  # PIL image

    # Convert to binary numpy array
    mask_np = np.array(mask.convert("L"))
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Save as PNG
    mask_filename = f"{i}_{label.replace(' ', '_')}.png"
    mask_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(mask_path, binary_mask)
    print(f"Saved mask: {mask_path}")

print(f"Segmentation completed in {time.time() - start_time:.2f} seconds.")
