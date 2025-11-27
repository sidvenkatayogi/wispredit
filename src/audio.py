import sounddevice as sd
import numpy as np

# Global state for recording
is_recording = False
audio_frames = []
stream = None

def callback(indata, frames, time, status):
    global is_recording, audio_frames
    if is_recording:
        audio_frames.append(indata.copy())

def init_audio_stream(sample_rate):
    """Initializes the audio stream in the background."""
    global stream
    if stream is not None:
        return
    
    print("Initializing audio stream...")
    try:
        stream = sd.InputStream(callback=callback, samplerate=sample_rate, channels=1)
        stream.start()
        print("Audio stream initialized.")
    except Exception as e:
        print(f"Failed to initialize audio stream: {e}")
        stream = None

def close_audio_stream():
    """Closes the audio stream."""
    global stream
    if stream:
        stream.stop()
        stream.close()
        stream = None

def start_recording_stream(sample_rate):
    """Starts recording audio from the microphone. Non-blocking."""
    global is_recording, audio_frames, stream
    
    if stream is None:
        init_audio_stream(sample_rate)
        
    if is_recording:
        return
    
    # Clear frames and set flag
    audio_frames = []
    is_recording = True
    print("Recording started...")

def stop_recording_stream():
    """Stops the audio recording and returns the frames."""
    global is_recording, audio_frames
    if not is_recording:
        return []
    
    is_recording = False
    print("Recording stopped.")
    return list(audio_frames)
