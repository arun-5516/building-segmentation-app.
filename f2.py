# streamlit_frontend.py
import streamlit as st
import requests
from pathlib import Path
import io
import time

st.set_page_config(page_title="Building Seg - Frontend", layout="wide")
st.title("Frontend — YOLOv8 Building Segmentation (test backend)")

st.sidebar.header("Settings")
backend_url = st.sidebar.text_input("Backend base URL (no trailing slash)", value="https://buildingsegmentation.onrender.com")
api_endpoint = st.sidebar.text_input("API endpoint (relative to backend)", value="/api/process")
full_endpoint = backend_url.rstrip("/") + api_endpoint

st.sidebar.markdown("---")
st.sidebar.markdown("Examples:")
st.sidebar.markdown("`https://your-backend.onrender.com` with endpoint `/api/process`")

st.markdown("Upload one or more images. The backend will run segmentation + postprocess and return a ZIP and file URLs.")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg","jpeg","png","tif","tiff"])
conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
# Optional fields to pass additional params
meters_per_pixel = st.sidebar.number_input("Meters per pixel (m/px)", value=0.03, format="%.6f")
min_area_m2 = st.sidebar.number_input("Min polygon area (m²)", value=0.05, format="%.6f")

if not uploaded_files:
    st.info("No images selected. Upload one or more images to enable processing.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Files to process")
    for f in uploaded_files:
        st.write(f"- {f.name}  ({int(len(f.getbuffer())/1024)} KB)")
    send = st.button("Send to backend & process")
with col2:
    st.markdown("### Backend response / previews")
    status_area = st.empty()
    results_area = st.empty()

if send:
    status_area.info(f"Uploading {len(uploaded_files)} file(s) to {full_endpoint} ...")
    # Build multipart form
    files = []
    for f in uploaded_files:
        # form field 'images' expected by backend
        files.append(("images", (f.name, f.getbuffer(), "image/jpeg")))

    # You can also send form fields for conf/min_area etc.
    data = {
        "conf": str(conf),
        "meters_per_pixel": str(meters_per_pixel),
        "min_area_m2": str(min_area_m2)
    }

    try:
        # timeout long enough for model processing
        resp = requests.post(full_endpoint, files=files, data=data, timeout=600)
    except requests.exceptions.RequestException as e:
        status_area.error(f"Request failed: {e}")
        st.stop()

    if resp.status_code != 200:
        status_area.error(f"Server returned status {resp.status_code}: {resp.text}")
        st.stop()

    try:
        payload = resp.json()
    except Exception as e:
        status_area.error(f"Failed to parse JSON response: {e}\n\nResponse text:\n{resp.text}")
        st.stop()

    status_area.success("Processing finished. Displaying results below.")
    # Normalize to list
    results = payload if isinstance(payload, list) else [payload]

    # Show each result (if multiple inputs)
    for i, res in enumerate(results):
        with results_area.container():
            st.markdown(f"---\n#### Result {i+1}")
            zip_url = res.get("zip_url") or res.get("zip") or res.get("zip_url_full")
            # If zip_url is a relative path like "/outputs/xxx.zip", make full
            if zip_url:
                if zip_url.startswith("http"):
                    full_zip = zip_url
                else:
                    full_zip = backend_url.rstrip("/") + zip_url
                st.markdown(f"[Download ZIP]({full_zip})")
                # also offer to open in new tab
                if st.button(f"Open ZIP {i+1}", key=f"openzip{i}"):
                    st.experimental_set_query_params(_open=full_zip)
                    # simple JS open via markdown link
                    st.markdown(f"<a href='{full_zip}' target='_blank'>Open ZIP in new tab</a>", unsafe_allow_html=True)

            files_list = res.get("files") or []
            # some backends return named fields
            for k in ("mask_url", "overlay_url", "geojson_url", "csv_url"):
                if k in res and isinstance(res[k], str):
                    files_list.append(res[k])

            if not files_list:
                st.info("No individual files returned. Use the ZIP link above to download outputs.")
                continue

            # Display thumbnails and links
            thumbs_cols = st.columns(min(3, len(files_list)))
            for idx, file_url in enumerate(files_list):
                c = thumbs_cols[idx % len(thumbs_cols)]
                if not file_url:
                    continue
                if file_url.startswith("http"):
                    full_url = file_url
                else:
                    full_url = backend_url.rstrip("/") + file_url
                fname = Path(full_url).name
                # show image inline if image file
                if any(fname.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
                    try:
                        c.image(full_url, caption=fname)
                    except Exception:
                        # fallback: fetch bytes and display
                        try:
                            r = requests.get(full_url, timeout=60)
                            if r.status_code == 200:
                                c.image(r.content, caption=fname)
                            else:
                                c.write(fname)
                                c.write(f"Cannot fetch ({r.status_code})")
                        except Exception as e:
                            c.write(fname)
                            c.write(f"Error fetching: {e}")
                else:
                    c.markdown(f"[{fname}]({full_url})")

    st.balloons()
    status_area.info("Done. You can upload more images or change settings.")

st.caption("Tip: if you deploy backend on Render, set backend base URL to the Render service URL and keep API endpoint `/api/process`.")
