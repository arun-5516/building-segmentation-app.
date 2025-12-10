# app.py
import os
import io
import zipfile
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict_service import predict_image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

@app.route("/api/process", methods=["POST"])
def api_process():
    if "images" not in request.files:
        return "No files uploaded", 400
    files = request.files.getlist("images")
    job_files = []
    for f in files:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_base = os.path.splitext(f.filename)[0] + "_" + ts
        in_path = os.path.join(OUTPUT_DIR, f"input_{safe_base}{os.path.splitext(f.filename)[1]}")
        f.save(in_path)

        try:
            produced = predict_image(in_path, OUTPUT_DIR, safe_base, conf=float(request.form.get("conf", 0.25)))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        job_files.extend(produced)

    # create single zip per request
    zip_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in job_files:
            zf.write(p, arcname=os.path.basename(p))

    # return zip download path
    return jsonify({
        "zip_url": f"/outputs/{os.path.basename(zip_path)}",
        "files": [f"/outputs/{os.path.basename(p)}" for p in job_files]
    })

@app.route("/outputs/<path:fname>")
def serve_outputs(fname):
    return send_from_directory(OUTPUT_DIR, fname)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
