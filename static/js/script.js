/* ==========================================================================
   AI-Based Adaptive Noise Removal for Clear Speech
   Frontend Dashboard Controller
   ========================================================================== */

(() => {
    const dropZone = document.getElementById("dropZone");
    const audioFileInput = document.getElementById("audioFileInput");
    const referenceFileInput = document.getElementById("referenceFileInput");
    const clearRefBtn = document.getElementById("clearRefBtn");

    const uploadBtn = document.getElementById("uploadBtn");
    const processBtn = document.getElementById("processBtn");
    const processSpinner = document.getElementById("processSpinner");
    const processBtnText = document.getElementById("processBtnText");
    const methodSelect = document.getElementById("methodSelect");

    const statusBanner = document.getElementById("statusBanner");
    const uploadMetaBox = document.getElementById("uploadMetaBox");
    const metaFilename = document.getElementById("metaFilename");
    const metaDuration = document.getElementById("metaDuration");
    const metaSampleRate = document.getElementById("metaSampleRate");
    const metaFileSize = document.getElementById("metaFileSize");

    const originalAudioPlayer = document.getElementById("originalAudioPlayer");
    const enhancedAudioPlayer = document.getElementById("enhancedAudioPlayer");
    const origPlayerDuration = document.getElementById("origPlayerDuration");
    const enhPlayerDuration = document.getElementById("enhPlayerDuration");

    const downloadBtn = document.getElementById("downloadBtn");
    const downloadSubtext = document.getElementById("downloadSubtext");

    const metricOrigDuration = document.getElementById("metricOrigDuration");
    const metricEnhDuration = document.getElementById("metricEnhDuration");
    const metricProcTime = document.getElementById("metricProcTime");
    const metricNoiseReduction = document.getElementById("metricNoiseReduction");
    const metricSnr = document.getElementById("metricSnr");
    const metricSnrNote = document.getElementById("metricSnrNote");
    const metricMethod = document.getElementById("metricMethod");

    const origWaveformImg = document.getElementById("origWaveformImg");
    const enhWaveformImg = document.getElementById("enhWaveformImg");
    const origSpecImg = document.getElementById("origSpecImg");
    const enhSpecImg = document.getElementById("enhSpecImg");

    const healthBadge = document.getElementById("healthBadge");
    const healthText = document.getElementById("healthText");

    let currentJobId = null;
    let currentUploadedFile = null;

    function formatSeconds(sec) {
        if (typeof sec !== "number" || Number.isNaN(sec)) {
            return "—";
        }
        return `${sec.toFixed(2)} s`;
    }

    function formatBytes(bytes) {
        if (typeof bytes !== "number" || bytes <= 0) {
            return "—";
        }
        if (bytes < 1024) {
            return `${bytes} B`;
        }
        if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(1)} KB`;
        }
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }

    function showStatus(message, type = "info") {
        statusBanner.textContent = message;
        statusBanner.className = `status-banner ${type}`;
        statusBanner.classList.remove("hidden");
    }

    function clearVisuals() {
        const frames = [
            { img: origWaveformImg, frame: document.getElementById("origWaveformFrame") },
            { img: enhWaveformImg, frame: document.getElementById("enhWaveformFrame") },
            { img: origSpecImg, frame: document.getElementById("origSpecFrame") },
            { img: enhSpecImg, frame: document.getElementById("enhSpecFrame") },
        ];
        frames.forEach(({ img, frame }) => {
            img.src = "";
            img.classList.add("hidden");
            const placeholder = frame.querySelector(".plot-placeholder");
            if (placeholder) {
                placeholder.classList.remove("hidden");
            }
        });
    }

    function setPlotImage(imgEl, frameId, url) {
        if (!url) return;
        const frame = document.getElementById(frameId);
        const placeholder = frame.querySelector(".plot-placeholder");
        if (placeholder) {
            placeholder.classList.add("hidden");
        }
        imgEl.src = `${url}?t=${Date.now()}`;
        imgEl.classList.remove("hidden");
    }

    function resetEnhancedOutputs() {
        enhancedAudioPlayer.removeAttribute("src");
        enhancedAudioPlayer.load();
        enhPlayerDuration.textContent = "Waiting for processing";

        downloadBtn.href = "#";
        downloadBtn.classList.add("disabled");
        downloadBtn.setAttribute("aria-disabled", "true");
        downloadSubtext.textContent = "Process an audio file to enable WAV download.";

        metricEnhDuration.textContent = "—";
        metricProcTime.textContent = "—";
        metricNoiseReduction.textContent = "—";
        metricSnr.textContent = "—";
        metricSnrNote.textContent = "Upload a clean reference file for SNR";
        metricMethod.textContent = "—";
        clearVisuals();
    }

    async function checkHealth() {
        try {
            const response = await fetch("/health");
            if (!response.ok) throw new Error("Health endpoint failed");
            const data = await response.json();
            healthBadge.className = "status-pill status-ok";
            const engineLabel = data.primary_model || "Speech Enhancer Ready";
            healthText.textContent = `Online · ${engineLabel}`;
        } catch (err) {
            healthBadge.className = "status-pill status-error";
            healthText.textContent = "Backend unreachable";
        }
    }

    function validateSelectedFile(file) {
        if (!file) {
            showStatus("Please select a WAV or MP3 audio file first.", "error");
            return false;
        }
        const nameLower = file.name.toLowerCase();
        if (!nameLower.endsWith(".wav") && !nameLower.endsWith(".mp3")) {
            showStatus("Unsupported file format. Please upload a .wav or .mp3 audio file.", "error");
            return false;
        }
        if (file.size === 0) {
            showStatus("The selected audio file is empty (0 bytes).", "error");
            return false;
        }
        return true;
    }

    async function uploadSelectedAudio() {
        const file = audioFileInput.files[0];
        if (!validateSelectedFile(file)) {
            return;
        }

        currentUploadedFile = file;
        currentJobId = null;
        processBtn.disabled = true;
        resetEnhancedOutputs();
        showStatus("Uploading and validating audio file...", "info");

        const formData = new FormData();
        formData.append("audio", file);

        const refFile = referenceFileInput.files[0];
        if (refFile) {
            formData.append("reference", refFile);
        }

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Upload failed.");
            }

            currentJobId = data.job_id;
            uploadMetaBox.classList.remove("hidden");
            metaFilename.textContent = data.original_filename;
            metaDuration.textContent = formatSeconds(data.duration_seconds);
            metaSampleRate.textContent = `${data.sample_rate} Hz`;
            metaFileSize.textContent = formatBytes(data.file_size_bytes);

            metricOrigDuration.textContent = formatSeconds(data.duration_seconds);
            origPlayerDuration.textContent = formatSeconds(data.duration_seconds);

            originalAudioPlayer.src = `${data.original_audio_url}?t=${Date.now()}`;
            originalAudioPlayer.load();

            processBtn.disabled = false;
            showStatus("Audio uploaded successfully! Click 'Remove Background Noise' to enhance speech.", "success");
        } catch (err) {
            showStatus(err.message || "Error uploading audio.", "error");
        }
    }

    async function processAudio() {
        if (!currentJobId) {
            showStatus("Please upload an audio file before processing.", "error");
            return;
        }

        processBtn.disabled = true;
        uploadBtn.disabled = true;
        processSpinner.classList.remove("hidden");
        processBtnText.textContent = "Enhancing Speech...";
        showStatus("Running AI noise removal and generating waveform/spectrogram comparisons...", "info");

        try {
            const response = await fetch("/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    job_id: currentJobId,
                    method: methodSelect.value,
                }),
            });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Audio enhancement failed.");
            }

            // Update Enhanced Player
            enhancedAudioPlayer.src = `${data.enhanced_audio_url}?t=${Date.now()}`;
            enhancedAudioPlayer.load();
            enhPlayerDuration.textContent = formatSeconds(data.metrics.enhanced_duration_seconds);

            // Enable Download Button
            downloadBtn.href = data.download_url;
            downloadBtn.classList.remove("disabled");
            downloadBtn.setAttribute("aria-disabled", "false");
            downloadSubtext.textContent = `Ready: ${data.enhanced_filename}`;

            // Populate Performance Metrics
            const m = data.metrics;
            metricOrigDuration.textContent = formatSeconds(m.original_duration_seconds);
            metricEnhDuration.textContent = formatSeconds(m.enhanced_duration_seconds);
            metricProcTime.textContent = formatSeconds(m.processing_time_seconds);
            metricNoiseReduction.textContent =
                typeof m.estimated_noise_reduction_db === "number"
                    ? `${m.estimated_noise_reduction_db.toFixed(2)} dB`
                    : "N/A";
            metricMethod.textContent = m.method_used || "AI Enhancer";

            if (m.snr_available) {
                metricSnr.textContent = `${m.snr_before_db.toFixed(2)} → ${m.snr_after_db.toFixed(2)} dB`;
                const gain = m.snr_improvement_db >= 0 ? `+${m.snr_improvement_db.toFixed(2)}` : m.snr_improvement_db.toFixed(2);
                metricSnrNote.textContent = `Improvement: ${gain} dB (vs clean reference)`;
            } else {
                metricSnr.textContent = "N/A";
                metricSnrNote.textContent = m.snr_note || "Requires a clean reference signal";
            }

            // Update Visualizations
            setPlotImage(origWaveformImg, "origWaveformFrame", data.plots.original_waveform_url);
            setPlotImage(enhWaveformImg, "enhWaveformFrame", data.plots.enhanced_waveform_url);
            setPlotImage(origSpecImg, "origSpecFrame", data.plots.original_spectrogram_url);
            setPlotImage(enhSpecImg, "enhSpecFrame", data.plots.enhanced_spectrogram_url);

            showStatus(
                `Enhancement complete in ${formatSeconds(m.processing_time_seconds)} using ${m.method_used}.`,
                "success"
            );
        } catch (err) {
            showStatus(err.message || "Error during audio processing.", "error");
        } finally {
            processBtn.disabled = false;
            uploadBtn.disabled = false;
            processSpinner.classList.add("hidden");
            processBtnText.textContent = "Remove Background Noise";
        }
    }

    // Drag and Drop + File Picker Events
    dropZone.addEventListener("click", () => audioFileInput.click());
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            audioFileInput.click();
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            audioFileInput.files = e.dataTransfer.files;
            uploadSelectedAudio();
        }
    });

    audioFileInput.addEventListener("change", () => {
        if (audioFileInput.files.length > 0) {
            uploadSelectedAudio();
        }
    });

    referenceFileInput.addEventListener("change", () => {
        if (currentUploadedFile && audioFileInput.files.length > 0) {
            uploadSelectedAudio();
        }
    });

    clearRefBtn.addEventListener("click", () => {
        referenceFileInput.value = "";
        if (currentUploadedFile && audioFileInput.files.length > 0) {
            uploadSelectedAudio();
        }
    });

    uploadBtn.addEventListener("click", uploadSelectedAudio);
    processBtn.addEventListener("click", processAudio);

    // Initialize
    checkHealth();
})();
