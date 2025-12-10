"""
predict_and_postprocess_final.py
Postprocessing: YOLOv8 mask -> cleaned polygons -> geojson -> overlay.
Safe for Render deployment (no absolute Windows paths).
"""

import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from PIL import Image, ImageDraw

# Default parameters (can be overridden by caller)
DEFAULT_METERS_PER_PIXEL = 0.03
DEFAULT_MIN_AREA_M2 = 0.05
DEFAULT_EPSILON_PX = 0.5
DEFAULT_SIMPLIFY_PX = 0.2
DEFAULT_OUTLINE_WIDTH = 2
DEFAULT_DO_UNION = False

kernel_small = np.ones((3, 3), np.uint8)

def masks_to_combined_uint8(masks_tensor):
    """Combine YOLO masks into single uint8 binary mask."""
    arr = getattr(masks_tensor, "data", masks_tensor)
    arr = np.array(arr)

    if arr.ndim == 2:
        combined = (arr > 0.5).astype(np.uint8)
    else:
        combined = np.zeros(arr.shape[1:], dtype=np.uint8)
        for m in arr:
            combined = np.maximum(combined, (m > 0.5).astype(np.uint8))

    return (combined * 255).astype(np.uint8)


def postprocess_mask_individual(
    mask_uint8,
    image_path,
    base_outname="result",
    output_dir="output_results",
    meters_per_pixel=DEFAULT_METERS_PER_PIXEL,
    min_area_m2=DEFAULT_MIN_AREA_M2,
    epsilon_px=DEFAULT_EPSILON_PX,
    simplify_tol_px=DEFAULT_SIMPLIFY_PX,
    do_union=DEFAULT_DO_UNION,
    outline_width=DEFAULT_OUTLINE_WIDTH,
):
    """Render-safe postprocessor: creates geojson + overlay bytes."""

    os.makedirs(output_dir, exist_ok=True)

    # Ensure binary
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Cleaning
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

    inv = cv2.bitwise_not(opened)
    h, w = inv.shape
    mask_floodfill = inv.copy()
    temp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_floodfill, temp, (0,0), 255)
    filled = cv2.bitwise_not(mask_floodfill)
    clean = cv2.bitwise_or(opened, filled)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px * (meters_per_pixel**2) < min_area_m2:
            continue

        approx = cv2.approxPolyDP(c, epsilon_px, True)
        if approx is None or len(approx) < 3:
            continue

        pts = approx.reshape(-1,2)
        poly = Polygon(pts).buffer(0)
        if poly.is_empty:
            continue

        polygons.append(poly)

    if not polygons:
        # return debug mask only
        return None, None, _arr_to_png_bytes(clean)

    # Optional union
    if do_union:
        merged = unary_union(polygons)
        polygons = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]

    # Simplify
    final_polys = []
    for p in polygons:
        p2 = p.simplify(simplify_tol_px, preserve_topology=True).buffer(0)
        if p2.is_empty:
            continue
        if p2.geom_type == "MultiPolygon":
            final_polys.extend(list(p2.geoms))
        else:
            final_polys.append(p2)

    if not final_polys:
        return None, None, _arr_to_png_bytes(clean)

    # Build GeoJSON bytes
    features = []
    for i, poly in enumerate(final_polys):
        features.append({
            "type": "Feature",
            "properties": {"id": i+1},
            "geometry": mapping(poly)
        })

    geojson_bytes = json.dumps({
        "type": "FeatureCollection",
        "features": features
    }).encode("utf-8")

    # Build overlay
    overlay_bytes = _make_overlay(final_polys, image_path, (w,h), outline_width)

    # Debug mask if needed
    debug_bytes = _arr_to_png_bytes(clean)

    return geojson_bytes, overlay_bytes, debug_bytes


def _make_overlay(polygons, image_path, size, outline_width):
    w, h = size
    if image_path and os.path.exists(image_path):
        base = Image.open(image_path).convert("RGBA")
    else:
        base = Image.new("RGBA", (w, h), (255,255,255,255))

    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    for poly in polygons:
        coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
        if len(coords) >= 2:
            draw.line(coords + [coords[0]], fill=(255,0,0,200), width=outline_width)

    combined = Image.alpha_composite(base, overlay)
    buf = io.BytesIO()
    combined.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _arr_to_png_bytes(arr):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
