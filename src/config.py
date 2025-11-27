import os
from dotenv import load_dotenv
from pynput import keyboard

load_dotenv()

HOTKEY = {keyboard.Key.ctrl, keyboard.Key.space}
SAMPLE_RATE = 16000
AUDIO_FILENAME = "temp_audio.wav"
MODEL_SIZE = "tiny.en"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")