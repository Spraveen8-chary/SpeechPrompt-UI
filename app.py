import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, session

import whisper
import ollama
from gtts import gTTS

# ==================================================
# SETUP LOGGING
# ==================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ==================================================
# LOAD WHISPER
# ==================================================
log.info("Loading Whisper model: small (fast & accurate)")
whisper_model = whisper.load_model("small")


# ==================================================
# FLASK SETUP
# ==================================================
app = Flask(__name__)
app.secret_key = "speechprompt-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_DOC_EXT = {".pdf", ".doc", ".docx", ".txt", ".md"}
MAX_DOCS = 5


# ==================================================
# SAVE FILES
# ==================================================
def _save_file(file):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    _, ext = os.path.splitext(file.filename)
    ext = ext or ".bin"
    filename = f"{ts}{ext}"

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    log.info(f"[FILE SAVED] {filename}")
    return filename


# ==================================================
# TRANSCRIPTION
# ==================================================
def transcribe_audio(path):
    log.info(f"[ASR] Starting transcription → {path}")

    start = datetime.utcnow()
    try:
        result = whisper_model.transcribe(path)

        lang = result.get("language", "unknown")
        text = result.get("text", "")
        duration = (datetime.utcnow() - start).total_seconds()

        log.info(f"[ASR DONE] lang={lang}, chars={len(text)}, time={duration:.2f}s")
        log.debug(f"[ASR TEXT] {text[:200]}...")

        return lang, text

    except Exception as e:
        log.error(f"[ASR ERROR] {e}")
        return "error", f"[ASR Error] {e}"


# ==================================================
# LLM CALL
# ==================================================
def mistral(messages):
    try:
        log.info(f"[LLM] Calling Mistral → prompt tokens={len(messages)}")
        log.debug(f"[LLM INPUT] {messages}")

        resp = ollama.chat(model="mistral:latest", messages=messages)
        output = resp["message"]["content"]

        log.info(f"[LLM DONE] chars={len(output)}")
        log.debug(f"[LLM OUTPUT] {output[:200]}...")

        return output
    except Exception as e:
        log.error(f"[LLM ERROR] {e}")
        return f"[LLM Error: {e}]"


# ==================================================
# TTS
# ==================================================
def tts_save(text):
    log.info("[TTS] Generating audio output")

    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_tts.mp3"
    out_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        gTTS(text=text, lang="en").save(out_path)
        log.info(f"[TTS DONE] {filename}")
        return filename
    except Exception as e:
        log.error(f"[TTS ERROR] {e}")
        return None


# ==================================================
# ROUTES
# ==================================================
@app.route("/", methods=["GET"])
def index():
    if "docs" not in session:
        session["docs"] = []

    log.info("[PAGE] GET /")
    return render_template("home.html",
                           result_text=None,
                           audio_filename=None,
                           doc_filenames=session["docs"])


@app.route("/", methods=["POST"])
def run_pipeline():
    log.info("======= RUN PIPELINE START =======")

    if "docs" not in session:
        session["docs"] = []

    output_type = request.form.get("output_type")
    user_prompt = (request.form.get("prompt_text") or "").strip()

    log.info(f"[MODE] Selected mode = {output_type}")
    log.info(f"[USER PROMPT] {user_prompt}")

    # -------------------------------
    # AUDIO HANDLING
    # -------------------------------
    audio_filename = None
    uploaded = request.files.get("audio_file")
    live_file = request.form.get("live_filename")

    if uploaded and uploaded.filename:
        audio_filename = _save_file(uploaded)
    elif live_file:
        audio_filename = live_file

    lang, transcription = None, ""

    if audio_filename:
        path = os.path.join(UPLOAD_FOLDER, audio_filename)
        lang, transcription = transcribe_audio(path)

    log.info(f"[ASR RESULT] lang={lang}, text_length={len(transcription)}")

    # -------------------------------
    # DOCUMENT UPLOADS
    # -------------------------------
    new_docs = request.files.getlist("session_docs")

    for d in new_docs:
        ext = os.path.splitext(d.filename)[1].lower()
        if ext in ALLOWED_DOC_EXT:
            saved = _save_file(d)
            session["docs"].append(saved)
            log.info(f"[DOC UPLOAD] Saved {saved}")

    session["docs"] = session["docs"][:MAX_DOCS]
    session.modified = True

    # -------------------------------
    # OUTPUT MODES
    # -------------------------------
    mistral_out = ""
    output_audio = None

    # 1️⃣ ASR MODE
    if output_type == "asr":
        log.info("[MODE] ASR only")

        result_text = (
            f"<b>Language:</b> {lang}<br><br>"
            f"<b>Transcription:</b><br>{transcription}"
        )
        output_audio = audio_filename

    # 2️⃣ CLASSIFICATION
    elif output_type == "classification":
        log.info("[MODE] Classification")

        messages = [
            {"role": "system", "content": "You are an audio classifier."},
            {"role": "user", "content": f"Transcription: {transcription}"}
        ]

        mistral_out = mistral(messages)
        output_audio = tts_save(mistral_out)

        result_text = (
            f"<b>Language:</b> {lang}<br><br>"
            f"<b>Transcription:</b><br>{transcription}<br><br>"
            f"<b>Classification:</b><br>{mistral_out}"
        )

    # 3️⃣ GENERATION
    elif output_type == "generation":
        log.info("[MODE] Generation")

        messages = [
            {"role": "system", "content": "You are a helpful generator assistant."},
            {"role": "user",
             "content": f"Transcription: {transcription}\nUser Prompt: {user_prompt}"}
        ]

        mistral_out = mistral(messages)
        output_audio = tts_save(mistral_out)

        result_text = (
            f"<b>Language:</b> {lang}<br><br>"
            f"<b>Transcription:</b><br>{transcription}<br><br>"
            f"<b>Generated Output:</b><br>{mistral_out}"
        )

    else:
        result_text = "[ERROR] Invalid output type"
        log.error("[MODE ERROR] Invalid output type")

    log.info("======= RUN PIPELINE END =======")

    return render_template("home.html",
                           result_text=result_text,
                           audio_filename=output_audio,
                           doc_filenames=session["docs"])


# ==================================================
# Remove document
# ==================================================
@app.route("/remove_doc", methods=["POST"])
def remove_doc():
    idx = int(request.form.get("index", -1))

    if 0 <= idx < len(session["docs"]):
        removed = session["docs"].pop(idx)
        log.info(f"[DOC REMOVE] Removed: {removed}")

        session.modified = True
        return jsonify({"ok": True})

    log.warning("[DOC REMOVE] Invalid index")
    return jsonify({"ok": False}), 400


# ==================================================
# Microphone upload endpoint
# ==================================================
@app.route("/api/upload", methods=["POST"])
def upload_blob():
    f = request.files.get("file")
    saved = _save_file(f)
    log.info(f"[MIC UPLOAD] Saved mic audio = {saved}")
    return jsonify({"filename": saved})


# ==================================================
# Serve audio files
# ==================================================
@app.route("/media/<path:filename>", endpoint="media")
def serve_media(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    log.info("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
