"""
predict_and_postprocess_final.py
Full end-to-end script (YOLOv8 seg -> binary mask -> postprocess -> GeoJSON + outline overlay).
Windows-safe (forward-slashes), tuned for building footprint extraction.

UPDATED FOR REVIEW:
- Uses 'best_colab.pt' model
- Enables retina_masks=True (Sharp edges)
- Increases polygon smoothing (cleaner lines)
"""
import os
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from PIL import Image, ImageDraw

# --------------------------- USER CONFIG (edit if needed) ---------------------------
BASE_DIR = "C:/Users/Arun/PycharmProjects/building_seg"   # use forward slashes (safe)

# ✅ FIX 1: Point to the model you downloaded from Colab
model_path = os.path.join(BASE_DIR, "best_colab.pt")

# Inputs: either a single image path or a folder (set one)
# ✅ CHECK THIS PATH matches your actual image location
input_image = "C:/Users/Arun/PycharmProjects/building_seg/test_images/map2.jpg"
input_folder = None

# Output folder
output_dir = os.path.join(BASE_DIR, "output_results")
os.makedirs(output_dir, exist_ok=True)

# Overlay settings
use_orig_for_overlay = True

# --------------------------- TUNED PARAMETERS ---------------------------
meters_per_pixel = 0.03      # set correctly for your orthomosaic (e.g., 0.03 = 3cm)
min_area_m2 = 0.05           # keep tiny roofs
conf_thres = 0.25            # confidence threshold

# ✅ FIX 2: Increased smoothing to fix "wobbly" lines
epsilon_px = 3.0             # approxPolyDP epsilon (Higher = straighter lines. Try 3.0 or 5.0)
simplify_tol_px = 0.5        # polygon simplify tolerance

do_union = False             # keep individual polygons
kernel_small = np.ones((3, 3), np.uint8)
outline_width = 2
device = "cpu"               # CPU is fine for inference
# ------------------------------------------------------------------------------------

print("Config:")
print(" BASE_DIR:", BASE_DIR)
print(" model_path:", model_path)
print(" input_image:", input_image)
print(" Smoothing (epsilon):", epsilon_px)

# ------------------- Derived & safety checks -------------------
if input_image:
    image_paths = [input_image]
else:
    image_paths = sorted(glob.glob(os.path.join(input_folder, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

if not image_paths:
    raise SystemExit(f"No input images found! Check your path.")

# Load model
if not os.path.exists(model_path):
    raise SystemExit(f"Model not found at: {model_path}. Did you move 'best.pt' to your project folder and rename it?")
model = YOLO(model_path)
print("✅ Loaded YOLO model.")

# ------------------- Utility: combine YOLO masks -------------------
def masks_to_combined_uint8(masks_tensor):
    if hasattr(masks_tensor, "detach"):
        arr = masks_tensor.detach().cpu().numpy()
    else:
        arr = np.array(masks_tensor)

    if arr.ndim == 2:
        combined = (arr > 0.5).astype(np.uint8)
    elif arr.ndim == 3:
        combined = np.zeros(arr.shape[1:], dtype=np.uint8)
        for m in arr:
            combined = np.maximum(combined, (m > 0.5).astype(np.uint8))
    else:
        return np.zeros((100,100), dtype=np.uint8) # fallback

    return (combined * 255).astype(np.uint8)


# ------------------- Postprocessing -------------------
def postprocess_mask_individual(mask_uint8, image_path, base_outname):
    # Ensure binary
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Clean up noise
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Fill holes
    h, w = opened.shape
    mask_floodfill = cv2.bitwise_not(opened)
    mask_temp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_floodfill, mask_temp, (0, 0), 255)
    filled = cv2.bitwise_not(mask_floodfill)
    binary_clean = cv2.bitwise_or(opened, filled)

    # Find contours
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px * (meters_per_pixel ** 2) < min_area_m2:
            continue

        # Polygon Approximation (The "Straight Lines" trick)
        approx = cv2.approxPolyDP(cnt, epsilon_px, True)
        if approx is None or len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2)
        poly = Polygon(pts).buffer(0)

        if poly.is_valid and not poly.is_empty:
             polygons.append(poly)

    if not polygons:
        print(f"No polygons found for {base_outname}")
        return None, None

    # Simplify
    simplified_parts = []
    for p in polygons:
        p2 = p.simplify(simplify_tol_px, preserve_topology=True).buffer(0)
        if p2.is_valid and not p2.is_empty:
            if p2.geom_type == "MultiPolygon":
                simplified_parts.extend([g for g in p2.geoms if g.is_valid])
            else:
                simplified_parts.append(p2)

    # Write GeoJSON
    features = []
    for i, poly in enumerate(simplified_parts):
        features.append({
            "type": "Feature",
            "properties": {"id": i + 1},
            "geometry": mapping(poly)
        })
    geojson_path = os.path.join(output_dir, f"clean_buildings_{base_outname}.geojson")
    with open(geojson_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    # Create Overlay
    if image_path and os.path.exists(image_path) and use_orig_for_overlay:
        orig_img = Image.open(image_path).convert("RGBA")
    else:
        orig_img = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", orig_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    outline_color = (255, 0, 0, 255)  # Solid Red

    for poly in simplified_parts:
        if poly.exterior:
            coords = list(poly.exterior.coords)
            draw.line(coords, fill=outline_color, width=outline_width)
        for interior in poly.interiors:
            coords = list(interior.coords)
            draw.line(coords, fill=outline_color, width=outline_width)

    combined = Image.alpha_composite(orig_img, overlay)
    overlay_path = os.path.join(output_dir, f"final_result_{base_outname}.png")
    combined.convert("RGB").save(overlay_path)

    return geojson_path, overlay_path


# ------------------------ MAIN LOOP ------------------------
for img_path in image_paths:
    name = Path(img_path).stem
    print(f"\n🚀 Processing: {name}")

    # ✅ FIX 3: High Quality Inference Settings
    # This generates sharp masks at original resolution
    results = model.predict(
        img_path,
        conf=conf_thres,
        imgsz=640,          # Match training size
        retina_masks=True,  # <--- MAGIC SWITCH for sharp edges
        device=device,
        verbose=False
    )
    r = results[0]

    mask_uint8 = None
    if r.masks is not None:
        mask_uint8 = masks_to_combined_uint8(r.masks.data)

    if mask_uint8 is not None:
        # Save raw mask debug
        cv2.imwrite(os.path.join(output_dir, f"{name}_raw_mask.png"), mask_uint8)

        # Run Post-processing
        out_geo, out_img = postprocess_mask_individual(mask_uint8, img_path, name)

        if out_img:
            print(f"✅ Success! Saved overlay: {out_img}")
    else:
        print("❌ No buildings detected.")

print("\nDONE. Check your output_results folder.")