# 🎬 Cinescore

Cinescore is an experimental Image-to-Music generation pipeline. It extracts visual features (time of day, weather, setting, energy, color palette) from any image and maps them deterministically into a music specification, which is then synthesized using Meta's MusicGen.

## Architecture
- **Frontend:** Streamlit web interface.
- **Backend:** FastAPI for async job queuing and execution.
- **Vision Models:** CLIP (zero-shot classification for mood/setting) and BLIP-2 (image captioning).
- **Audio Model:** MusicGen (small) for text-to-audio synthesis.

## Quick Start (Google Colab)
The easiest way to run this project with GPU acceleration is via Google Colab.

1. Create a new Google Colab notebook and set the runtime to **T4 GPU**.
2. Upload `cinescore.ipynb` to Colab and follow the instructions in the notebook.
3. The notebook handles all dependency installations, model downloading, and launching the Streamlit interface via `ngrok` or Colab proxy.

## Running Locally

### Prerequisites
- Python 3.10+
- NVIDIA GPU with at least 16GB VRAM (recommended)
- FFmpeg (required for audio processing)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/awsume18-collab/CineScore-AI.git
cd cinescore
```

2. Install the required system dependencies (Ubuntu/Debian):
```bash
sudo apt-get install libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev pkg-config
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. Start the FastAPI backend:
```bash
uvicorn imgtune.api.main:app --host 0.0.0.0 --port 8000
```

2. Start the Streamlit frontend (in a new terminal):
```bash
streamlit run web/app.py
```

3. Open your browser and navigate to `http://localhost:8501`.

## Customization
- **Rules (`imgtune/mapping/rules.py`):** The logic connecting visual features to musical traits (tempo, key, instruments).
- **Prompts (`imgtune/mapping/prompt_banks.py`):** The categories used by CLIP to score the image.

## Acknowledgments
- [MusicGen](https://github.com/facebookresearch/audiocraft) by Meta
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) by Salesforce
