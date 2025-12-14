"""
Streamlit app: YOLOv8 segmentation -> mask -> polygon postprocess -> download ZIP
UPDATED:
- Fixed NumPy/PyTorch warnings
- Fixed Streamlit 'use_column_width' deprecation warning
- Uses 'best_colab.pt' with High Quality settings (retina_masks)
"""
import streamlit as st
import os
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

# ------------------------ USER SETTINGS ------------------------
# ✅ FIX 1: Point to your trained model
MODEL_PATH = "best_colab.pt"

DEFAULT_CONF = 0.25
DEFAULT_MIN_AREA_M2 = 0.05
METERS_PER_PIXEL_DEFAULT = 0.03
USE_ORIG_FOR_OVERLAY = True

# ✅ FIX 2: Default smoothing set to 3.0 for straighter lines
DEFAULT_EPSILON = 3.0
# ---------------------------------------------------------------

st.set_page_config(page_title="Building Segmentation Demo", layout="wide")

st.title("🏙️ Building Segmentation Demo")
st.markdown(
    f"""
    **Model:** `{MODEL_PATH}`  
    **Status:** Ready for Review  
    Upload an aerial map image to detect building footprints and generate GeoJSONs.
    """
)

# ------------------------ Helper utilities ------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: **{model_path}**")
        st.error("Please ensure 'best_colab.pt' is in your GitHub repository.")
        st.stop()
    model = YOLO(model_path)
    return model

def masks_to_combined_uint8(masks_tensor):
    """
    Accepts either a tensor-like or array-like masks object from ultralytics Results.
    Returns a single uint8 binary mask (0/255) where all instance masks are OR-ed.
    """
    if masks_tensor is None:
        return None
    
    # ✅ FIX 3: Robust Tensor conversion to avoid NumPy warnings
    if hasattr(masks_tensor, "data"):
        masks_tensor = masks_tensor.data
    
    if hasattr(masks_tensor, "cpu"):
        arr = masks_tensor.cpu().numpy()
    else:
        # Fallback for other array types
        arr = np.array(masks_tensor)

    if arr.ndim == 2:
        combined = (arr > 0.5).astype(np.uint8)
    elif arr.ndim == 3:
        combined = np.zeros(arr.shape[1:], dtype=np.uint8)
        for m in arr:
            combined = np.maximum(combined, (m > 0.5).astype(np.uint8))
    else:
        # Fallback empty mask if shape is weird
        return np.zeros((100, 100), dtype=np.uint8)

    return (combined * 255).astype(np.uint8)

def postprocess_mask_individual(mask_uint8, image_path=None,
                                meters_per_pixel=METERS_PER_PIXEL_DEFAULT,
                                min_area_m2=DEFAULT_MIN_AREA_M2,
                                epsilon_px=DEFAULT_EPSILON, simplify_tol_px=0.2,
                                do_union=False, outline_width=2,
                                use_orig_for_overlay=USE_ORIG_FOR_OVERLAY):
    """
    From binary mask (uint8 0/255) produce:
      - geojson_bytes, overlay_png_bytes, debug_mask_png_bytes
    """
    # Ensure binary
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape

    # small morphology to clean noise
    kernel_small = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # fill holes inside buildings
    inv = cv2.bitwise_not(opened)
    mask_floodfill = inv.copy()
    mask_temp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_floodfill, mask_temp, (0, 0), 255)
    filled = cv2.bitwise_not(mask_floodfill)
    binary_clean = cv2.bitwise_or(opened, filled)

    # find contours
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        area_m2 = area_px * (meters_per_pixel ** 2)
        if area_m2 < min_area_m2:
            continue
        
        # Polygon Approximation (Straightens wobbly lines)
        approx = cv2.approxPolyDP(cnt, epsilon_px, True)
        if approx is None or len(approx) < 3:
            continue
            
        pts = approx.reshape(-1, 2)
        poly = Polygon(pts).buffer(0)
        
        if not poly.is_valid or poly.is_empty:
            continue
            
        polygons.append(poly)

    if not polygons:
        debug_bytes = _image_to_bytes(binary_clean)
        return None, None, debug_bytes

    # optionally union polygons
    if do_union:
        unioned = unary_union(polygons)
        if unioned.geom_type == "Polygon":
            polygons = [unioned]
        elif unioned.geom_type == "MultiPolygon":
            polygons = list(unioned.geoms)

    # simplify
    simplified_parts = []
    for p in polygons:
        p2 = p.simplify(simplify_tol_px, preserve_topology=True).buffer(0)
        if p2.is_valid and not p2.is_empty:
            if p2.geom_type == "MultiPolygon":
                simplified_parts.extend([g for g in p2.geoms if g.is_valid])
            else:
                simplified_parts.append(p2)

    if not simplified_parts:
        debug_bytes = _image_to_bytes(binary_clean)
        return None, None, debug_bytes

    # build geojson
    features = []
    for i, poly in enumerate(simplified_parts):
        features.append({
            "type": "Feature",
            "properties": {"id": i + 1},
            "geometry": mapping(poly)
        })
    fc = {"type": "FeatureCollection", "features": features}
    geojson_bytes = json.dumps(fc, indent=2).encode("utf-8")

    # create overlay PNG
    overlay_bytes = _create_overlay_png(simplified_parts, image_path, (w, h), outline_width, use_orig_for_overlay)
    debug_bytes = _image_to_bytes(binary_clean)

    return geojson_bytes, overlay_bytes, debug_bytes

def _image_to_bytes(img_array):
    if img_array.ndim == 2:
        pil = Image.fromarray(img_array).convert("L")
    else:
        pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def _create_overlay_png(polygons, image_path, size, outline_width, use_orig_for_overlay):
    w, h = size
    if image_path and os.path.exists(image_path) and use_orig_for_overlay:
        orig_img = Image.open(image_path).convert("RGBA")
        base_img = orig_img if orig_img.size == (w, h) else orig_img
    else:
        base_img = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    outline_color = (255, 0, 0, 255) # Solid Red

    for poly in polygons:
        exterior_coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
        if len(exterior_coords) >= 2:
            draw.line(exterior_coords + [exterior_coords[0]], fill=outline_color, width=outline_width)
        for interior in poly.interiors:
            interior_coords = [(float(x), float(y)) for x, y in interior.coords]
            if len(interior_coords) >= 2:
                draw.line(interior_coords + [interior_coords[0]], fill=outline_color, width=max(1, outline_width-1))

    combined = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    buf = io.BytesIO()
    combined.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# ------------------------ Streamlit UI ------------------------
st.sidebar.header("Processing Settings")
conf_thres = st.sidebar.slider("Confidence", 0.0, 1.0, float(DEFAULT_CONF), 0.01)
meters_per_pixel = st.sidebar.number_input("Scale (meters/pixel)", value=float(METERS_PER_PIXEL_DEFAULT), format="%.4f")
eps_px = st.sidebar.number_input("Smoothing (Epsilon)", value=DEFAULT_EPSILON, help="Higher = Straighter lines. Lower = More detail.")
simplify_px = st.sidebar.number_input("Simplify Tolerance", value=0.2)
outline_width = st.sidebar.slider("Line Width", 1, 5, 2)
use_orig_overlay = st.sidebar.checkbox("Overlay on Original Image", value=USE_ORIG_FOR_OVERLAY)

# Load model once
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Upload
uploaded_files = st.file_uploader("Drop images here...", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if st.button("🚀 Run Segmentation"):
    if not uploaded_files:
        st.warning("Please upload an image first.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_files = []  
        
        # Tabs for multiple images
        tabs = st.tabs([up.name for up in uploaded_files])

        for idx, up in enumerate(uploaded_files):
            with tabs[idx]:
                st.write(f"**Processing:** {up.name}")

                # Save temp image
                tmp_img_path = os.path.join(tmpdir, up.name)
                with open(tmp_img_path, "wb") as f:
                    f.write(up.getbuffer())

                # ✅ FIX 4: Run YOLOv8 Inference with Retina Masks
                with st.spinner("Detecting buildings..."):
                    results = model.predict(
                        source=tmp_img_path, 
                        conf=conf_thres, 
                        imgsz=640,          # Match training size
                        retina_masks=True,  # <--- CRITICAL for sharp edges
                        device="cpu"
                    )

                if not results:
                    st.warning("No results.")
                    continue
                r = results[0]

                # Extract masks
                mask_uint8 = None
                if r.masks is not None:
                    mask_uint8 = masks_to_combined_uint8(r.masks)

                if mask_uint8 is None:
                    st.warning("No buildings detected.")
                    continue

                # Process
                geojson_bytes, overlay_bytes, debug_mask_bytes = postprocess_mask_individual(
                    mask_uint8,
                    image_path=tmp_img_path,
                    meters_per_pixel=meters_per_pixel,
                    epsilon_px=eps_px,
                    simplify_tol_px=simplify_px,
                    outline_width=outline_width,
                    use_orig_for_overlay=use_orig_overlay
                )

                # Display Results
                col1, col2 = st.columns(2)
                with col1:
                    # ✅ FIX 5: Use 'use_container_width' instead of 'use_column_width'
                    st.image(mask_uint8, caption="Raw Mask", use_container_width=True, channels="GRAY")
                with col2:
                    if overlay_bytes:
                        st.image(overlay_bytes, caption="Final Result", use_container_width=True)
                    else:
                        st.warning("No valid polygons found.")

                # Add to ZIP
                base = Path(up.name).stem
                if overlay_bytes: output_files.append((f"{base}_overlay.png", overlay_bytes))
                if geojson_bytes: output_files.append((f"{base}.geojson", geojson_bytes))

        # Create ZIP
        if output_files:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in output_files:
                    zf.writestr(fname, data)
            zip_buf.seek(0)

            st.success("Processing Complete!")
            st.download_button(
                label="📥 Download Results (ZIP)", 
                data=zip_buf.getvalue(), 
                file_name="building_results.zip", 
                mime="application/zip"
            )
