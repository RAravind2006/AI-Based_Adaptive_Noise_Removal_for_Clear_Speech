# AI-Based Adaptive Noise Removal for Clear Speech

An AI-powered web application that removes unwanted background noise from speech recordings while preserving human voice clarity using **PyTorch**, **Librosa**, **Matplotlib**, and **Flask**.

---

## Key Features

- **Audio Upload & Validation**: Supports `.wav` and `.mp3` recordings up to 50 MB, displaying filename, sample rate, file size, and duration.
- **Real Deep Learning Speech Enhancement**:
  - **Primary Model**: Pretrained **PyTorch Demucs DNS64** (`denoiser.pretrained.dns64`) operating at 16 kHz with overlap-add chunked inference for longer recordings.
  - **Fallback / Selectable Engine**: **Adaptive Spectral Gating** via `noisereduce` and `librosa` STFT noise-floor estimation.
- **Side-by-Side Audio Playback**: Compare original noisy audio and enhanced clean speech directly in the browser.
- **Waveform & Spectrogram Comparisons**: High-contrast dark-themed Matplotlib plots comparing time-domain amplitudes and frequency-domain energy (dB) with matched scales.
- **Honest Performance Analysis**: Reports original duration, enhanced duration, processing time, estimated noise-floor reduction (dB), active model used, and exact SNR (Before → After) when a clean reference audio file is provided.
- **WAV Export**: Download the enhanced speech recording as a 16-bit PCM `.wav` file.

---

## Project Structure

```text
AI-Based-Adaptive-Noise-Removal/
│
├── app.py                     # Flask backend server & REST routes
├── requirements.txt           # Python dependencies
├── README.md                  # Setup, usage, and testing guide
│
├── models/
│   ├── __init__.py
│   ├── speech_enhancer.py     # PyTorch Demucs DNS64 + Spectral Gating fallback
│   ├── audio_processor.py     # Audio validation, loading, saving & metrics
│   └── visualizations.py      # Matplotlib waveform & spectrogram plots
│
├── uploads/                   # Temporary uploaded noisy/reference audio
├── outputs/                   # Enhanced WAV audio outputs
│
├── static/
│   ├── css/
│   │   └── style.css          # Dark navy & purple/blue gradient UI
│   ├── js/
│   │   └── script.js          # Frontend upload, playback & dashboard logic
│   └── images/                # Generated waveform & spectrogram PNGs
│
├── templates/
│   └── index.html             # Responsive web dashboard
│
└── tests/
    └── test_app.py            # Automated backend, validation & AI pipeline tests
```

---

## Installation & Running Locally

### 1. Navigate to the project folder
```bash
cd "AI-Based-Adaptive-Noise-Removal"
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(On Windows PowerShell: `.venv\Scripts\Activate.ps1`)*

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Flask application
```bash
python app.py
```
Then open your browser and visit:
```text
http://127.0.0.1:5000
```

*(If port 5000 is already in use on macOS due to AirPlay Receiver, run with another port, e.g. `PORT=5050 python app.py` and open `http://127.0.0.1:5050`.)*

---

## Backend API Routes

| Route | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Renders the web dashboard (`templates/index.html`). |
| `/health` | `GET` | Returns application status and loaded AI model readiness. |
| `/upload` | `POST` | Validates and stores uploaded `.wav`/`.mp3` audio (and optional clean reference). |
| `/process` | `POST` | Runs AI noise removal, generates comparison plots, and computes metrics. |
| `/download` | `GET` | Downloads the enhanced WAV file (`?job_id=<id>` or latest output). |

---

## Running Automated Tests

Run the full `pytest` suite (covers noisy speech, clear speech, white/hum/pink noise, MP3 support, invalid/empty uploads, reference SNR, and WAV download):

```bash
.venv/bin/pytest -v
```

---

## Sample Audio Datasets for Testing & Demo

You can test the application with any noisy `.wav` or `.mp3` recording from your microphone, or download real noisy speech benchmarks from these public datasets:

1. **VoiceBank-DEMAND (Noisy Speech Benchmark)**:
   - https://datashare.ed.ac.uk/handle/10283/2791
   - Contains paired clean and noisy speech recordings (cafeteria, traffic, kitchen, bus, etc.) — ideal for testing both noise removal and exact SNR evaluation with the optional reference upload!
2. **Microsoft DNS Challenge (Deep Noise Suppression)**:
   - https://github.com/microsoft/DNS-Challenge
3. **LibriSpeech ASR Corpus (Clean Speech)**:
   - https://www.openslr.org/12
4. **UrbanSound8K / ESC-50 (Background Noise Samples)**:
   - https://github.com/karolpiczak/ESC-50
