import rumps
import pyperclip
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import numpy as np
from pynput import keyboard
from pynput.keyboard import Controller
import threading
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

HOTKEY = {keyboard.Key.shift, keyboard.Key.cmd, keyboard.Key.space}
SAMPLE_RATE = 16000
AUDIO_FILENAME = "temp_audio.wav"
MODEL_SIZE = "tiny.en"

current_keys = set()
is_recording = False
audio_frames = []
model = None
text_to_paste = None # Shared variable for thread communication
transcription_history = []

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables. Voice editing will be disabled.")

def load_model():
    """Loads the Whisper model into memory."""
    global model
    print(f"Loading Whisper model: {MODEL_SIZE}...")
    try:
        model = whisper.load_model(MODEL_SIZE)
        print("Model loaded successfully.")
        return True
    except Exception as e:
        rumps.alert("Error", f"Failed to load the Whisper model: {e}")
        return False

def get_selected_text():
    """
    Simulates Cmd+C to copy selected text and returns it.
    Restores the original clipboard content afterwards.
    """
    original_clipboard = pyperclip.paste()
    
    controller = Controller()
    
    with controller.pressed(keyboard.Key.cmd):
        controller.press('c')
        controller.release('c')
    
    # Wait a bit for clipboard to update
    time.sleep(0.1)
    
    selected_text = pyperclip.paste()
    
    return selected_text

def check_if_editing_command(current_text, new_command, history):
    # print("--- Checking for Edit Command ---")
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set. Skipping edit check.")
        return False, None

    try:
        # Use gemini-2.0-flash as it is faster and more current
        model_name = 'gemini-2.0-flash'
        model = genai.GenerativeModel(model_name)
        
        history_str = "\n".join(history)
        
        prompt = f"""
        You are a voice assistant helper.
        
        Context (previous commands):
        {history_str}
        
        Current Text (Context):
        {current_text}
        
        New Voice Command:
        {new_command}
        
        Task:
        Determine if the "New Voice Command" is an instruction to EDIT the "Current Text".
        Examples of edit commands: "replace hello with hi", "delete the last word", "make it all caps", "change the first sentence".
        Examples of non-edit commands (just dictation): "hello world", "this is a test", "I want to go to the store".
        
        Note: if the EDIT command seems to refer to an unspecifc sentence or part of the text, perform the edit on the Context/previous command, and then return the full current text with the new edited text. Still leave the rest of the text unchanged though.
        If it is an EDIT command, perform the edit on "Current Text" and return the RESULTING TEXT ONLY. Do not output any reasoning or explanation.
        
        If it is NOT an edit command (it's just new text to be typed), return exactly the string "NO_EDIT".
        """
        
        # print(f"Sending prompt to Gemini:\n{prompt}")
        response = model.generate_content(prompt)
        result = response.text.strip()
        # print(f"Gemini Response: {result}")
        
        if result == "NO_EDIT":
            return False, None
        else:
            return True, result + " "
    except Exception as e:
        print(f"Gemini Error: {e}")
        print("Listing available models:")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"- {m.name}")
        except:
            pass
        return False, None

def transcribe_audio(frames):
    """Transcribes audio and places the text in the shared variable."""
    global model, text_to_paste
    if not frames:
        print("No audio frames to transcribe.")
        return
    
    print("Transcribing...")
    audio_np = np.concatenate(frames, axis=0)
    
    # Use a unique filename to avoid conflicts
    import uuid
    filename = f"temp_audio_{uuid.uuid4()}.wav"
    
    write(filename, SAMPLE_RATE, audio_np)
    
    try:
        result = model.transcribe(filename, fp16=False)
        transcribed_text = result["text"].strip()
        if transcribed_text:
            text_to_paste = transcribed_text
            
            # Update history
            transcription_history.append(transcribed_text)
            if len(transcription_history) > 3:
                transcription_history.pop(0)

    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def start_recording():
    """Starts recording audio from the microphone."""
    global is_recording, audio_frames
    if is_recording:
        return
    
    is_recording = True
    audio_frames = []
    print("Recording started...")
    
    def callback(indata, frames, time, status):
        if is_recording:
            audio_frames.append(indata.copy())

    stream = sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1)
    with stream:
        while is_recording:
            sd.sleep(100)

def stop_recording():
    """Stops the audio recording and triggers transcription."""
    global is_recording, audio_frames
    if not is_recording:
        return
    
    is_recording = False
    print("Recording stopped.")
    # Pass a copy of the frames to the thread to avoid race conditions
    frames_to_process = list(audio_frames)
    threading.Thread(target=transcribe_audio, args=(frames_to_process,)).start()

class WsprEditApp(rumps.App):
    def __init__(self):
        if not os.path.exists("icon.png"):
            from PIL import Image
            img = Image.new('RGB', (16, 16), color='black')
            img.save('icon.png')
            
        super(WsprEditApp, self).__init__("Wspr Edit", icon="icon.png", quit_button="Quit")
        self.menu = ["Recording: OFF"]
        self.state_recording = False
        # Timer to check for text to paste, runs on the main thread
        self.paste_timer = rumps.Timer(self.check_and_paste, 0.1)
        self.paste_timer.start()

    def check_and_paste(self, _):
        """Checks for text and pastes it from the main thread."""
        global text_to_paste
        if text_to_paste:
            text = text_to_paste
            text_to_paste = None # Clear the variable
            
            controller = Controller()
            
            original_clipboard = pyperclip.paste()
            
            with controller.pressed(keyboard.Key.cmd):
                controller.press('a')
                controller.release('a')
            
            time.sleep(0.1)
            
            with controller.pressed(keyboard.Key.cmd):
                controller.press('c')
                controller.release('c')
                
            time.sleep(0.1)
            
            current_value = pyperclip.paste()
            
            is_edit = False
            edited_text = None
            
            if current_value and isinstance(current_value, str) and GEMINI_API_KEY:
                    # print("Calling Gemini to check for edit command...")
                    history_context = transcription_history[0:1]
                    is_edit, edited_text = check_if_editing_command(current_value, text, history_context)
            else:
                print(f"Skipping Gemini check. Value: {current_value}, Key Present: {bool(GEMINI_API_KEY)}")
            
            if is_edit and edited_text is not None:
                print(f"Editing detected. Old: {current_value}, New: {edited_text}")
                # We already Selected All, but just in case user had a long audio message and maybe clicked out, just select all again
                with controller.pressed(keyboard.Key.cmd):
                    controller.press('a')
                    controller.release('a')
                pyperclip.copy(edited_text)
                
                with controller.pressed(keyboard.Key.cmd):
                    controller.press('v')
                    controller.release('v')
            else:
                print(f"Pasting: {text}")
                print()
                
                # Move cursor to end (Right Arrow) bc we have currently selected all
                controller.press(keyboard.Key.right)
                controller.release(keyboard.Key.right)

                # add a space for user to continue (only needed for no edit)
                pyperclip.copy(text + " ")
                
                with controller.pressed(keyboard.Key.cmd):
                    controller.press('v')
                    controller.release('v')
            
            # Restore clipboard to original state
            time.sleep(0.2)
            pyperclip.copy(original_clipboard)

    def update_menu_state(self):
        self.menu["Recording: OFF"].title = f"Recording: {'ON' if self.state_recording else 'OFF'}"

    def on_press(self, key):
        """Handle key press events."""
        if key in HOTKEY:
            current_keys.add(key)
            if all(k in current_keys for k in HOTKEY) and not self.state_recording:
                self.state_recording = True
                self.update_menu_state()
                threading.Thread(target=start_recording).start()

    def on_release(self, key):
        """Handle key release events."""
        global current_keys
        if all(k in current_keys for k in HOTKEY):
            self.state_recording = False
            self.update_menu_state()
            stop_recording()
        
        if key in current_keys:
            current_keys.remove(key)

    def run_hotkey_listener(self):
        """Runs the keyboard listener."""
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

if __name__ == "__main__":
    app = WsprEditApp()
    
    model_thread = threading.Thread(target=load_model)
    model_thread.start()
    
    listener_thread = threading.Thread(target=app.run_hotkey_listener)
    listener_thread.start()

    app.run()
