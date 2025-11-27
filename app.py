from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():
    result_text = None
    audio_filename = None

    if request.method == "POST":
        prompt_text = request.form.get("prompt_text") or ""
        output_type = request.form.get("output_type") or "asr"

        file = request.files.get("audio_file")
        live_filename = request.form.get("live_filename")

        if file and file.filename:
            audio_filename = _save_file(file)
        elif live_filename:
            audio_filename = live_filename

        prompt_audio = request.form.get("prompt_audio")
        _ = prompt_audio  

        if output_type == "asr":
            result_text = f"[ASR] Processed: {prompt_text}"
        elif output_type == "classification":
            result_text = f"[Classification] Processed: {prompt_text}"
        else:
            result_text = f"[Generation] Processed: {prompt_text}"

    return render_template("home.html",
                           result_text=result_text,
                           audio_filename=audio_filename)


@app.route("/api/upload", methods=["POST"])
def upload_blob():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400

    filename = _save_file(file)
    return jsonify({"filename": filename}), 201


@app.route("/media/<path:filename>")
def media(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


def _save_file(file_storage):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    _, ext = os.path.splitext(file_storage.filename)
    if not ext:
        ext = ".webm"
    filename = f"{ts}{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(path)
    return filename


if __name__ == "__main__":
    app.run(debug=True)

