# AI True Crime Story Generator

An AI pipeline that automatically generates **documentary-style crime stories** based on real notorious criminals and crime cases using **Wikipedia, LLMs, image generation, and text-to-speech**.

The system uses **LangGraph** to orchestrate a multi-step workflow that:
1. Selects a real crime case
2. Retrieves information from Wikipedia
3. Generates a narration script using an LLM
4. Splits the story into scenes
5. Generates documentary-style images using Stable Diffusion
6. Generates narration audio using Bark TTS
7. Exports assets for video creation

## Technologies

- LangGraph
- HuggingFace Transformers
- Stable Diffusion (SDXL Turbo)
- Bark Text-to-Speech
- Wikipedia API
- Python
- PyTorch

## Example Output

Each run generates a folder like:

- video_assets/
- case_name_timestamp/
- story.txt
- sections.json
- img_01.png
- img_02.png
- audio_01.wav
- audio_02.wav
- manifest.json



These assets can be combined into a **short-form documentary-style video**.

## Running the Pipeline

```python
state = app.invoke({"query": ""})

print("Chosen:", state.get("source_title"))
print("Wiki:", state.get("source_url"))
print("Output folder:", state.get("out_dir"))
