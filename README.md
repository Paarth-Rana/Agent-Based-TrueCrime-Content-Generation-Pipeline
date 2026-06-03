# Agent-Based True Crime Content Generation Pipeline

An **agentic AI pipeline** that automatically generates documentary-style video assets (script, images, and narration) from real-world crime cases.

The system uses a multi-step workflow powered by **LangGraph**, where an AI agent selects a topic, gathers information, and generates consistent multimedia content.

---

## What It Does

1. Selects a real true-crime case (optionally guided by user input)
2. Retrieves factual information from Wikipedia  
3. Generates a structured narration script using an LLM  
4. Splits the story into scenes  
5. Generates realistic documentary-style images  
6. Generates voice narration using text-to-speech  
7. Exports all assets for video creation  

---

## Key Idea

This project focuses on **agent-based workflows**, where:

- Each step is handled by a node in a LangGraph pipeline
- Accepts optional user input to guide topic selection while ensuring a specific, valid case is chosen
- The system makes decisions (topic selection, filtering, validation, structuring)
- Outputs are kept aligned across text, images, and audio  

---

## Tech Stack

- **LangGraph** – agent workflow orchestration  
- **Transformers (Hugging Face)** – LLM + TTS  
- **Stable Diffusion (SDXL Turbo)** – image generation  
- **Bark** – text-to-speech  
- **PyTorch** – model execution  
- **Wikipedia API** – factual data source  

---

## Project Structure

```text
.
├── pipeline.py
├── utils.py
├── colab_runner.ipynb   # GPU testing (Colab + Cursor via git)
├── main.ipynb
├── run.py
├── requirements.txt
└── outputs/
```
---

## Cursor + Colab (GPU testing)

Edit in **Cursor**, run on **Colab GPU** (no copy-paste). Git is the bridge.

### One-time

1. Push this repo to GitHub (already: `Paarth-Rana/Agent-Based-TrueCrime-Content-Generation-Pipeline`).
2. Open [colab_runner.ipynb](colab_runner.ipynb) in Colab:
   - Upload the file, or **File → Open notebook → GitHub** → paste the repo URL.
3. **Runtime → Change runtime type → GPU**.
4. Run all **Setup** cells once per Colab session (clone, optional Drive cache, load models).

### After each code change in Cursor

```bash
git add pipeline.py utils.py   # files you changed
git commit -m "your message"
git push origin main
```

In Colab, run only the **Sync & test run** cell (pulls latest code, reloads module, runs pipeline).

| When | Colab action |
|------|----------------|
| Changed `.py` files | **Sync & test run** cell only |
| Changed `requirements.txt` | Re-run Setup + restart runtime |
| Colab disconnected | Re-run all Setup cells |

Models are **not** re-downloaded on every edit—only on new sessions (unless Hugging Face cache is on Google Drive; enable in the notebook).

### Open in Colab badge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Paarth-Rana/Agent-Based-TrueCrime-Content-Generation-Pipeline/blob/main/colab_runner.ipynb)

---

## How to Run

### Option 1: Notebook

```python
from pipeline import run_pipeline

state = run_pipeline("")
```

### Option 2 (optional CLI)

```python
python run.py
```
or provide a topic:

```python
python run.py "D. B. Cooper"
```
---


## Example Output

outputs/
  case_name_timestamp/
    story.txt
    sections.json
    img_01.png
    img_02.png
    audio_01.wav
    audio_02.wav
    manifest.json

---

## Current Limitations / Work in Progress

- Image generation is not always perfectly consistent with the story
- Audio narration can sometimes sound unnatural or misaligned
- Outputs may occasionally drift from the selected topic
- Final video stitching is not yet implemented

This project is still being improved to increase consistency, realism, and reliability across all generated assets.

---

## Future Improvements

- Automatic video generation (combine images + audio into final video)
- Improved consistency between script, images, and narration
- Better topic selection and filtering
- Optional UI or web interface
