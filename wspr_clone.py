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
from AppKit import NSWorkspace
import AppKit



# --- Configuration ---
HOTKEY = {keyboard.Key.shift, keyboard.Key.cmd, keyboard.Key.space}
SAMPLE_RATE = 16000
AUDIO_FILENAME = "temp_audio.wav"
MODEL_SIZE = "tiny.en"

# --- Global State ---
current_keys = set()
is_recording = False
audio_frames = []
model = None
text_to_paste = None # Shared variable for thread communication

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

def is_textbox_focused():
    """Check if the currently focused UI element is a text box."""
    workspace = NSWorkspace.sharedWorkspace()
    active_app = workspace.frontmostApplication()
    if not active_app:
        return False
    try:
        system_events = AppKit.SBApplication.applicationWithBundleIdentifier_("com.apple.systemevents")
        active_process = system_events.processes().filteredArrayUsingPredicate_(
            AppKit.NSPredicate.predicateWithFormat_("bundleIdentifier == %@", active_app.bundleIdentifier())
        ).firstObject()
        if not active_process:
            return False
        focused_element = active_process.attributes().objectForKey_(AppKit.NSAccessibilityFocusedUIElementAttribute)
        if not focused_element:
            return False
        role = focused_element.attributes().objectForKey_(AppKit.NSAccessibilityRoleAttribute)
        text_roles = [
            AppKit.NSAccessibilityTextFieldRole,
            AppKit.NSAccessibilityTextAreaRole,
            AppKit.NSAccessibilityTextViewRole,
            "AXTextArea",
            "AXTextField"
        ]
        return role in text_roles
    except Exception:
        return True

def transcribe_audio():
    """Transcribes audio and places the text in the shared variable."""
    global audio_frames, model, text_to_paste
    if not audio_frames:
        return
    
    print("Transcribing...")
    audio_np = np.concatenate(audio_frames, axis=0)
    audio_frames = []
    
    write(AUDIO_FILENAME, SAMPLE_RATE, audio_np)
    
    try:
        result = model.transcribe(AUDIO_FILENAME)
        transcribed_text = result["text"].strip()
        if transcribed_text:
            text_to_paste = transcribed_text # Put text in shared variable
            
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        if os.path.exists(AUDIO_FILENAME):
            os.remove(AUDIO_FILENAME)

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
    global is_recording
    if not is_recording:
        return
    
    is_recording = False
    print("Recording stopped.")
    threading.Thread(target=transcribe_audio).start()

class WsprCloneApp(rumps.App):
    def __init__(self):
        if not os.path.exists("icon.png"):
            from PIL import Image
            img = Image.new('RGB', (16, 16), color='black')
            img.save('icon.png')
            
        super(WsprCloneApp, self).__init__("WsprClone", icon="icon.png", quit_button="Quit")
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
            
            if is_textbox_focused():
                print(f"Pasting: {text}")
                pyperclip.copy(text)
                
                controller = Controller()
                controller.press(keyboard.Key.cmd)
                controller.press('v')
                controller.release('v')
                controller.release(keyboard.Key.cmd)

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
    app = WsprCloneApp()
    
    model_thread = threading.Thread(target=load_model)
    model_thread.start()
    
    listener_thread = threading.Thread(target=app.run_hotkey_listener)
    listener_thread.start()

    app.run()
