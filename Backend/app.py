from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import ollama
import base64
import io
import os
import tempfile
from PyPDF2 import PdfReader
from pydub import AudioSegment
from faster_whisper import WhisperModel
from typing import Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model for audio transcription
whisper_model = WhisperModel("tiny", device="cpu")  # Use "cuda" if GPU available

def process_file(file: UploadFile, prompt: str) -> dict:
    """
    Process uploaded file based on type and return Ollama parameters.
    """
    content_type = file.content_type
    file_bytes = file.file.read()

    # File size limit: 10MB
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    messages = [{"role": "user", "content": prompt}]

    if "image" in content_type:
        # Check if llava is available
        available_models = [m["name"] for m in ollama.list()["models"]]
        if "llava" not in available_models:
            raise HTTPException(status_code=400, detail="Image support requires llava model. Run: ollama pull llava")
        # Encode image to base64 for llava
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        messages[0]["images"] = [base64_image]
        return {"model": "llava", "messages": messages}

    elif "audio" in content_type or "video" in content_type:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if "video" in content_type:
            # Extract audio from video using pydub
            audio = AudioSegment.from_file(tmp_path)
            audio_path = tmp_path + ".wav"
            audio.export(audio_path, format="wav")
            tmp_path = audio_path

        # Transcribe audio
        segments, _ = whisper_model.transcribe(tmp_path)
        transcribed_text = " ".join([seg.text for seg in segments])
        os.unlink(tmp_path)  # Clean up

        messages[0]["content"] += f"\n\nTranscribed content: {transcribed_text}"
        return {"model": None, "messages": messages}

    elif "pdf" in content_type:
        # Extract text from PDF
        pdf = PdfReader(io.BytesIO(file_bytes))
        pdf_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        messages[0]["content"] += f"\n\nPDF content: {pdf_text}"
        return {"model": None, "messages": messages}

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

@app.post("/chat")
async def chat(prompt: str = Form(...), model: str = Form("llama_uncensored"), file: Optional[UploadFile] = File(None)):
    # Validate model
    if model not in ["llama_uncensored", "codegemma"]:
        raise HTTPException(status_code=400, detail="Invalid model. Choose 'llama_uncensored' or 'codegemma'")

    if file:
        ollama_params = process_file(file, prompt)
        selected_model = ollama_params["model"] or model
        messages = ollama_params["messages"]
    else:
        selected_model = model
        if "code" in prompt.lower():  # Auto-switch for code-related prompts
            selected_model = "codegemma"
        messages = [{"role": "user", "content": prompt}]

    # Stream response from Ollama
    def generate():
        try:
            stream = ollama.chat(model=selected_model, messages=messages, stream=True)
            for chunk in stream:
                yield chunk["message"]["content"]
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/event-stream")