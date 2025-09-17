# app.py
import os
import io
import json
import uuid
import math
import subprocess
from datetime import timedelta

from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template, abort
)
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np

# Optional but recommended: pip install librosa soundfile
# librosa can decode many formats via soundfile/audioread (ffmpeg recommended for .webm)
import librosa

# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT   = os.getenv("DATA_ROOT", os.getcwd())
AUDIO_DIR   = os.path.join(DATA_ROOT, "audio_files")
META_CSV    = os.path.join(DATA_ROOT, "meta.csv")
UPLOAD_DIR  = os.path.join(DATA_ROOT, "uploads")
MODEL_NAME  = os.getenv("MODEL_PREFIX", "final_model")  # your saved prefix
DATA_DIR = os.path.join(DATA_ROOT, "data")
REFERENCE_DIR  = os.path.abspath(os.path.join(DATA_ROOT, "release_in_the_wild"))
REFERENCE_META = os.path.join(REFERENCE_DIR, "meta.csv")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
DEFAULT_SR  = 16000  # only used if you need to re-encode with ffmpeg



# -----------------------------
# OPTIONAL: integrate your pipeline
# -----------------------------
PIPELINE = None
PIPELINE_ERROR = None

def _load_pipeline():
    global PIPELINE, PIPELINE_ERROR
    try:
        import torch, os
        from config import Config
        from pipeline import DeepfakeDetectionPipeline

        cfg = Config()
        cfg.device = torch.device("cpu")  # you said you're on CPU
        cfg.data_root = DATA_ROOT

        # IMPORTANT: these two lines fix your error
        cfg.train_data_path = REFERENCE_DIR               # AudioDataset will look for meta.csv here
        cfg.vector_db_path  = os.path.join(DATA_DIR, "vector_db")  # where faiss_index.bin lives

        PIPELINE = DeepfakeDetectionPipeline(cfg)
        PIPELINE.load_models("final_model")
        PIPELINE.vector_db.load()

        # Remap Colab /content paths to your server dataset root
        PIPELINE.vector_db.vector_paths = [
            os.path.join(REFERENCE_DIR, os.path.basename(p))
            for p in PIPELINE.vector_db.vector_paths
        ]

        # Sanity prints:
        print("REFERENCE_DIR:", REFERENCE_DIR)
        print("Has meta.csv:", os.path.exists(REFERENCE_META))
        print("FAISS ntotal:", getattr(PIPELINE.vector_db.index, "ntotal", 0))
        print("Sample paths:", PIPELINE.vector_db.vector_paths[:3])

        PIPELINE_ERROR = None
    except Exception as e:
        PIPELINE = None
        PIPELINE_ERROR = str(e)

_load_pipeline()

# -----------------------------
# DATA CATALOG
# -----------------------------
def _label_to_str(y):
    # Assumes your training used 1=spoof, 0=bonafide
    if isinstance(y, str):
        s = y.strip().lower()
        if s in {"bona-fide", "bonafide"}:
            return "bona-fide"
        if s in {"spoof", "fake", "synthetic"}:
            return "spoof"
        return s
    return "spoof" if int(y) == 1 else "bona-fide"

def _read_meta():
    if not os.path.exists(META_CSV):
        # fallback: build a minimal df from files
        files = [f for f in os.listdir(AUDIO_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_EXT]
        return pd.DataFrame({"file": files, "speaker": ["unknown"]*len(files), "label": ["unknown"]*len(files)})
    df = pd.read_csv(META_CSV)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    file_col   = cols.get("file", "file")
    speaker_col= cols.get("speaker", "speaker")
    label_col  = cols.get("label", "label")
    df = df.rename(columns={file_col:"file", speaker_col:"speaker", label_col:"label"})
    return df[["file","speaker","label"]]

def _read_meta_ref():
    if not os.path.exists(REFERENCE_META):
        # fallback: build a minimal df from files
        files = [f for f in os.listdir(REFERENCE_DIR) if os.path.splitext(f)[1].lower() in ALLOWED_EXT]
        return pd.DataFrame({"file": files, "speaker": ["unknown"]*len(files), "label": ["unknown"]*len(files)})
    df = pd.read_csv(REFERENCE_META)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    file_col   = cols.get("file", "file")
    speaker_col= cols.get("speaker", "speaker")
    label_col  = cols.get("label", "label")
    df = df.rename(columns={file_col:"file", speaker_col:"speaker", label_col:"label"})
    return df[["file","speaker","label"]]

_DUR_CACHE = {}

def _get_duration_seconds(filepath: str) -> float:
    if filepath in _DUR_CACHE:
        return _DUR_CACHE[filepath]
    try:
        dur = librosa.get_duration(path=filepath)  # librosa 0.10+: get_duration(path=...) (older: filename=...)
    except TypeError:
        dur = librosa.get_duration(filename=filepath)
    except Exception:
        # last resort: 0.0 if unreadable
        dur = 0.0
    _DUR_CACHE[filepath] = float(dur)
    return float(dur)

def _fmt_dur(seconds: float) -> str:
    if not np.isfinite(seconds): return "00:00"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"

def _catalog():
    df = _read_meta()
    rows = []
    for _, r in df.iterrows():
        fname = str(r["file"])
        path  = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(path):
            continue
        dur_s = _get_duration_seconds(path)
        rows.append({
            "file": fname,
            "speaker": str(r["speaker"]),
            "label": _label_to_str(r["label"]),
            "duration_sec": dur_s,
            "duration": _fmt_dur(dur_s),
            "url": f"/audio/{fname}",
        })
    # optional: show most recent by name-like numeric sort
    def _key(x):
        base, _ = os.path.splitext(x["file"])
        return int(base) if base.isdigit() else base
    rows.sort(key=_key, reverse=True)
    return rows

# -----------------------------
# FILE HANDLING
# -----------------------------
def _allowed(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in ALLOWED_EXT

def _save_upload(file_storage) -> str:
    "Save uploaded file to UPLOAD_DIR and return absolute path."
    filename = secure_filename(file_storage.filename or f"upload_{uuid.uuid4().hex}.wav")
    if not _allowed(filename):
        # force .wav extension if missing/unknown
        root, _ = os.path.splitext(filename)
        filename = root + ".wav"
    out_path = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(out_path)
    return out_path

def _ensure_wav_if_needed(path_in: str) -> str:
    """
    If the file is .webm/.ogg/etc and librosa can't decode, try ffmpeg -> mono 16k wav.
    Returns the usable path (may be the same if already okay).
    """
    ext = os.path.splitext(path_in)[1].lower()
    if ext == ".wav":
        return path_in
    # First try: maybe librosa can handle it (ffmpeg available via audioread)
    try:
        _ = _get_duration_seconds(path_in)
        return path_in
    except Exception:
        pass

    # Fallback: ffmpeg transcode (requires ffmpeg in PATH)
    path_out = os.path.join(UPLOAD_DIR, f"conv_{uuid.uuid4().hex}.wav")
    cmd = ["ffmpeg", "-y", "-i", path_in, "-ac", "1", "-ar", str(DEFAULT_SR), path_out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return path_out
    except Exception as e:
        raise RuntimeError(f"ffmpeg transcode failed: {e}")

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/list", methods=["GET"])
def api_list():
    return jsonify({"items": _catalog()})

@app.route("/audio/<path:filename>", methods=["GET"])
def serve_audio(filename):
    # Try audio_files first
    path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(path):
        return send_from_directory(AUDIO_DIR, filename, as_attachment=False)
    # Then uploads
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)
    abort(404, description="Audio not found")

@app.route("/api/dbinfo", methods=["GET"])
def dbinfo():
    vdb = PIPELINE.vector_db
    return jsonify({
        "vector_db_path": PIPELINE.config.vector_db_path,
        "index_file_exists": os.path.exists(os.path.join(PIPELINE.config.vector_db_path, "faiss_index.bin")),
        "metadata_file_exists": os.path.exists(os.path.join(PIPELINE.config.vector_db_path, "metadata.pkl")),
        "has_index": vdb.index is not None,
        "ntotal": getattr(vdb.index, "ntotal", 0),
        "sample_vector_files": [os.path.basename(p) for p in vdb.vector_paths[:5]],
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if PIPELINE is None:
        return jsonify({"ok": False, "error": f"Model not loaded: {PIPELINE_ERROR or 'plug your pipeline in app.py _load_pipeline()'}"}), 500

    # Determine source: (1) uploaded/recorded file OR (2) existing filename from call log
    src_path = None
    used_existing = False

    if "filename" in request.form and request.form["filename"].strip():
        fname = os.path.basename(request.form["filename"].strip())
        candidate = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(candidate):
            return jsonify({"ok": False, "error": f"File not found: {fname}"}), 400
        src_path = candidate
        used_existing = True

    if ("file" in request.files) and request.files["file"].filename:
        # If both provided, uploaded/recorded wins
        saved = _save_upload(request.files["file"])
        src_path = saved
        used_existing = False

    if not src_path:
        return jsonify({"ok": False, "error": "Provide either an uploaded file or choose an existing filename."}), 400

    try:
        usable = _ensure_wav_if_needed(src_path)
        result = PIPELINE.predict(usable)  # must return dict described above

        # Build neighbor table for UI (speaker, duration, label, distance)
        df = _read_meta_ref()
        df["file_base"] = df["file"].astype(str).apply(os.path.basename)

        neighbors = []

        # Preferred: rich "retrieved"
        if "retrieved" in result and isinstance(result["retrieved"], list):
            for r in result["retrieved"]:
                fname = os.path.basename(r.get("file") or r.get("path") or "")
                print("fname:", fname)
                meta = df[df["file_base"] == fname].head(1)
                speaker = (meta["speaker"].iloc[0] if len(meta) else "unknown")
                label   = _label_to_str(meta["label"].iloc[0] if len(meta) else r.get("label", "unknown"))
                apath   = os.path.join(REFERENCE_DIR, fname)
                dur     = _get_duration_seconds(apath) if os.path.exists(apath) else 0.0
                neighbors.append({
                    "file": fname,
                    "speaker": str(speaker),
                    "label": str(label),
                    "duration": _fmt_dur(dur),
                    "duration_sec": float(dur),
                    "distance": r.get("distance", None),
                    "url": f"/audio/{fname}" if os.path.exists(apath) else ""
                })
        else:
            # Fallback: trio lists
            files = result.get("retrieved_files", []) or []
            labels= result.get("retrieved_labels", []) or []
            dists = result.get("retrieved_distances", []) or []
            for i, fname in enumerate(files):
                fname = os.path.basename(str(fname))
                print("fname:", fname)
                meta = df[df["file"] == fname].head(1)
                speaker = (meta["speaker"].iloc[0] if len(meta) else "unknown")
                label_val = labels[i] if i < len(labels) else "unknown"
                label = _label_to_str(label_val)
                apath = os.path.join(AUDIO_DIR, fname)
                dur   = _get_duration_seconds(apath) if os.path.exists(apath) else 0.0
                dist  = dists[i] if i < len(dists) else None
                neighbors.append({
                    "file": fname,
                    "speaker": str(speaker),
                    "label": str(label),
                    "duration": _fmt_dur(dur),
                    "duration_sec": float(dur),
                    "distance": None if (dist is None or (isinstance(dist, float) and math.isnan(dist))) else float(dist),
                    "url": f"/audio/{fname}" if os.path.exists(apath) else ""
                })
        # temporarily in /api/predict
        print("predict result keys:", list(result.keys()))
        print("retrieved:", result.get("retrieved"))
        print("retrieved_files:", result.get("retrieved_files"))
        resp = {
            "ok": True,
            "source": {
                "used_existing": used_existing,
                "path": src_path if used_existing else os.path.basename(src_path),
            },
            "prediction": result.get("prediction"),
            "probability": float(result.get("probability", 0.0)),
            "neighbors": neighbors,
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    # For local dev:
    #   $ export FLASK_ENV=development
    #   $ python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
