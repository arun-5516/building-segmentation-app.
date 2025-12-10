# predict_service.py
import os
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

# model path inside container (set to your repo path or env var)
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n-seg.pt")

# Load model once at import time
print("Loading YOLO model from:", MODEL_PATH)
MODEL = YOLO(MODEL_PATH)
print("Model loaded.")

def masks_to_combined_uint8(masks_tensor):
    # same utility we used earlier
    if masks_tensor is None:
        return None
    arr = getattr(masks_tensor, "data", masks_tensor)
    arr = np.array(arr)
    if arr.ndim == 2:
        combined = (arr > 0.5).astype(np.uint8)
    else:
        combined = np.zeros(arr.shape[1:], dtype=np.uint8)
        for m in arr:
            combined = np.maximum(combined, (m > 0.5).astype(np.uint8))
    return (combined * 255).astype(np.uint8)

def predict_image(image_path: str, output_dir: str, base_name: str,
                  conf=0.25, meters_per_pixel=0.03, min_area_m2=0.05):
    """
    Runs inference + postprocess, writes files into output_dir, returns list of file paths.
    Implement postprocessing using your existing code (contours->polygons->geojson).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = MODEL.predict(source=image_path, conf=conf, device="cpu")
    if not results:
        raise RuntimeError("No results from model")
    r = results[0]

    mask_uint8 = None
    if hasattr(r, "masks") and r.masks is not None:
        mask_uint8 = masks_to_combined_uint8(r.masks.data if hasattr(r.masks, "data") else r.masks)

    if mask_uint8 is None:
        raise RuntimeError("No mask produced")

    # Save mask PNG
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask_uint8)

    # Call your postprocessing (you can import functions if located elsewhere)
    # Example: produce geojson and overlay bytes and write them out
    from shapely.geometry import Polygon, mapping
    # ... reuse your postprocess_mask_individual function to create overlay & geojson bytes
    # For simplicity, reuse code from earlier in conversation or import that function.
    # Here assume we have a function postprocess_mask_individual that returns (geojson_bytes, overlay_bytes, debug_bytes)
    from predict_and_postprocess_final import postprocess_mask_individual  # adapt path
    geojson_bytes, overlay_bytes, debug_bytes = postprocess_mask_individual(mask_uint8, image_path)

    geojson_path = os.path.join(output_dir, f"{base_name}.geojson")
    with open(geojson_path, "wb") as f:
        f.write(geojson_bytes)

    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    with open(overlay_path, "wb") as f:
        f.write(overlay_bytes)

    debug_path = os.path.join(output_dir, f"{base_name}_mask_debug.png")
    with open(debug_path, "wb") as f:
        f.write(debug_bytes)

    return [mask_path, overlay_path, geojson_path, debug_path]
