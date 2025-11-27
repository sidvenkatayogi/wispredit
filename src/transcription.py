import whisper
import numpy as np
from scipy.io.wavfile import write
import os
import uuid
from .config import MODEL_SIZE, SAMPLE_RATE

model = None

def load_model():
    """Loads the Whisper model into memory."""
    global model
    print(f"Loading Whisper model: {MODEL_SIZE}...")
    try:
        model = whisper.load_model(MODEL_SIZE)
        print("Model loaded successfully.")
        print()
        return True
    except Exception as e:
        print(f"Error: Failed to load the Whisper model: {e}")
        return False

def transcribe_audio_frames(frames):
    """Transcribes audio frames using the loaded Whisper model."""
    global model
    if not frames:
        print("No audio frames to transcribe.")
        return None
    
    if model is None:
        print("Model not loaded.")
        return None

    print("Transcribing...")
    print()
    audio_np = np.concatenate(frames, axis=0)
    
    # Use a unique filename to avoid conflicts
    filename = f"temp_audio_{uuid.uuid4()}.wav"
    
    write(filename, SAMPLE_RATE, audio_np)
    
    try:
        result = model.transcribe(filename, fp16=False)
        transcribed_text = result["text"].strip()
        return transcribed_text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
    finally:
        if os.path.exists(filename):
            os.remove(filename)
