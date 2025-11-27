import sounddevice as sd
import numpy as np

# Global state for recording
is_recording = False
audio_frames = []

def start_recording_stream(sample_rate):
    """Starts recording audio from the microphone. Blocking call."""
    global is_recording, audio_frames
    if is_recording:
        return
    
    is_recording = True
    audio_frames = []
    print("Recording started...")
    
    def callback(indata, frames, time, status):
        if is_recording:
            audio_frames.append(indata.copy())

    stream = sd.InputStream(callback=callback, samplerate=sample_rate, channels=1)
    with stream:
        while is_recording:
            sd.sleep(100)

def stop_recording_stream():
    """Stops the audio recording and returns the frames."""
    global is_recording, audio_frames
    if not is_recording:
        return []
    
    is_recording = False
    print("Recording stopped.")
    return list(audio_frames)
