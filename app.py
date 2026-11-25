"""Flask Backend for AI-Based Adaptive Noise Removal for Clear Speech."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from models import (
    SpeechEnhancer,
    compute_metrics,
    generate_comparison_plots,
    load_audio,
    save_wav,
    validate_audio_file,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
IMAGES_FOLDER = os.path.join(BASE_DIR, "static", "images")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models", "cache")

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
JOB_TTL_SECONDS = 2 * 3600  # Clean up temporary job files older than 2 hours

# In-memory registry mapping job_id -> file paths & metadata
JOBS: Dict[str, Dict[str, Any]] = {}


def create_app() -> Flask:
    """Application factory creating and configuring the Flask instance."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
    app.config["IMAGES_FOLDER"] = IMAGES_FOLDER

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    enhancer = SpeechEnhancer()

    def cleanup_expired_jobs() -> None:
        """Remove old temporary files from uploads, outputs, and static/images."""
        now = time.time()
        expired_ids = [
            jid for jid, info in JOBS.items() if now - info.get("created_at", now) > JOB_TTL_SECONDS
        ]
        for jid in expired_ids:
            info = JOBS.pop(jid, {})
            for key in ("upload_path", "reference_path", "output_path"):
                path = info.get(key)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            for plot_name in info.get("plots", {}).values():
                plot_path = os.path.join(app.config["IMAGES_FOLDER"], plot_name)
                if os.path.exists(plot_path):
                    try:
                        os.remove(plot_path)
                    except OSError:
                        pass

    @app.errorhandler(413)
    def file_too_large(_err):
        return jsonify({"error": "File exceeds the 50 MB upload limit."}), 413

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "app": "AI-Based Adaptive Noise Removal for Clear Speech",
                "deep_learning_available": enhancer.is_deep_learning_available(),
                "primary_model": enhancer.primary_model_label(),
                "fallback_model": "Adaptive Spectral Gating (Librosa + NoiseReduce)",
            }
        )

    @app.route("/upload", methods=["POST"])
    def upload_audio():
        cleanup_expired_jobs()

        if "audio" not in request.files:
            return jsonify({"error": "No audio file field provided in request."}), 400

        file = request.files["audio"]
        if not file or not file.filename or file.filename.strip() == "":
            return jsonify({"error": "No audio file selected. Please choose a WAV or MP3 file."}), 400

        safe_name = secure_filename(file.filename)
        if not safe_name:
            return jsonify({"error": "Invalid filename provided."}), 400

        job_id = uuid.uuid4().hex[:12]
        upload_filename = f"{job_id}_{safe_name}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
        file.save(upload_path)

        is_valid, message = validate_audio_file(upload_path, safe_name)
        if not is_valid:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return jsonify({"error": message}), 400

        # Handle optional clean reference file for objective SNR comparison
        reference_path = None
        if "reference" in request.files:
            ref_file = request.files["reference"]
            if ref_file and ref_file.filename and ref_file.filename.strip():
                ref_safe_name = secure_filename(ref_file.filename)
                ref_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{job_id}_ref_{ref_safe_name}"
                )
                ref_file.save(ref_path)
                ref_valid, ref_msg = validate_audio_file(ref_path, ref_safe_name)
                if not ref_valid:
                    os.remove(upload_path)
                    if os.path.exists(ref_path):
                        os.remove(ref_path)
                    return jsonify({"error": f"Reference audio error: {ref_msg}"}), 400
                reference_path = ref_path

        audio, sr = load_audio(upload_path)
        duration_seconds = round(float(len(audio) / sr), 3)
        file_size_bytes = os.path.getsize(upload_path)

        JOBS[job_id] = {
            "job_id": job_id,
            "original_filename": safe_name,
            "upload_path": upload_path,
            "reference_path": reference_path,
            "output_path": None,
            "enhanced_filename": None,
            "sample_rate": sr,
            "duration_seconds": duration_seconds,
            "plots": {},
            "created_at": time.time(),
        }

        return jsonify(
            {
                "message": "Audio uploaded and validated successfully.",
                "job_id": job_id,
                "original_filename": safe_name,
                "duration_seconds": duration_seconds,
                "sample_rate": sr,
                "file_size_bytes": file_size_bytes,
                "original_audio_url": url_for("serve_original_audio", job_id=job_id),
            }
        )

    @app.route("/process", methods=["POST"])
    def process_audio():
        payload = request.get_json(silent=True) or request.form.to_dict()
        job_id = (payload.get("job_id") or "").strip()
        method = (payload.get("method") or "auto").strip()

        if not job_id or job_id not in JOBS:
            return jsonify({"error": "Invalid or expired job_id. Please upload an audio file first."}), 400

        job = JOBS[job_id]
        upload_path = job["upload_path"]
        if not os.path.exists(upload_path):
            return jsonify({"error": "Uploaded audio file is no longer available."}), 404

        try:
            start_time = time.perf_counter()
            original_audio, original_sr = load_audio(upload_path)

            enhanced_audio, enhanced_sr, method_used = enhancer.enhance(
                original_audio, original_sr, method=method
            )

            stem, _ = os.path.splitext(job["original_filename"])
            enhanced_filename = f"enhanced_{stem}_{job_id}.wav"
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], enhanced_filename)
            save_wav(output_path, enhanced_audio, enhanced_sr)

            plot_filenames = generate_comparison_plots(
                original_audio=original_audio,
                original_sr=original_sr,
                enhanced_audio=enhanced_audio,
                enhanced_sr=enhanced_sr,
                output_dir=app.config["IMAGES_FOLDER"],
                job_id=job_id,
            )

            processing_time = time.perf_counter() - start_time

            metrics = compute_metrics(
                original_audio=original_audio,
                original_sr=original_sr,
                enhanced_audio=enhanced_audio,
                enhanced_sr=enhanced_sr,
                processing_time_seconds=processing_time,
                method_used=method_used,
                reference_audio_path=job.get("reference_path"),
            )

            job["output_path"] = output_path
            job["enhanced_filename"] = enhanced_filename
            job["plots"] = plot_filenames

            return jsonify(
                {
                    "message": "Speech enhancement completed successfully.",
                    "job_id": job_id,
                    "enhanced_filename": enhanced_filename,
                    "enhanced_audio_url": url_for("serve_enhanced_audio", job_id=job_id),
                    "download_url": url_for("download_audio", job_id=job_id),
                    "metrics": metrics,
                    "plots": {
                        "original_waveform_url": url_for(
                            "static", filename=f"images/{plot_filenames['original_waveform']}"
                        ),
                        "enhanced_waveform_url": url_for(
                            "static", filename=f"images/{plot_filenames['enhanced_waveform']}"
                        ),
                        "original_spectrogram_url": url_for(
                            "static", filename=f"images/{plot_filenames['original_spectrogram']}"
                        ),
                        "enhanced_spectrogram_url": url_for(
                            "static", filename=f"images/{plot_filenames['enhanced_spectrogram']}"
                        ),
                    },
                }
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Audio processing failed: {exc}"}), 500

    @app.route("/audio/original/<job_id>", methods=["GET"])
    def serve_original_audio(job_id: str):
        job = JOBS.get(job_id)
        if not job or not os.path.exists(job["upload_path"]):
            return jsonify({"error": "Original audio file not found."}), 404
        return send_file(job["upload_path"])

    @app.route("/audio/enhanced/<job_id>", methods=["GET"])
    def serve_enhanced_audio(job_id: str):
        job = JOBS.get(job_id)
        if not job or not job.get("output_path") or not os.path.exists(job["output_path"]):
            return jsonify({"error": "Enhanced audio file not found."}), 404
        return send_file(job["output_path"], mimetype="audio/wav")

    @app.route("/download", methods=["GET"])
    def download_audio():
        job_id = (request.args.get("job_id") or "").strip()

        if job_id:
            job = JOBS.get(job_id)
        else:
            # Allow direct GET /download to return the most recently processed output
            processed_jobs = [
                j for j in JOBS.values() if j.get("output_path") and os.path.exists(j["output_path"])
            ]
            job = max(processed_jobs, key=lambda j: j["created_at"], default=None)

        if not job or not job.get("output_path") or not os.path.exists(job["output_path"]):
            return jsonify({"error": "No enhanced audio file is available for download."}), 404

        download_name = f"clean_{os.path.splitext(job['original_filename'])[0]}.wav"
        return send_file(
            job["output_path"],
            mimetype="audio/wav",
            as_attachment=True,
            download_name=download_name,
        )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
