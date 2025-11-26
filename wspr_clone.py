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
import google.generativeai as genai
import ApplicationServices
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
transcription_history = []

# Configure Gemini
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

def get_focused_textbox_info():
    """Check if the currently focused UI element is a text box and return its value."""
    try:
        workspace = NSWorkspace.sharedWorkspace()
        front_app = workspace.frontmostApplication()
        if not front_app:
            print("No frontmost application found.")
            return False, None
        
        pid = front_app.processIdentifier()
        ax_app = ApplicationServices.AXUIElementCreateApplication(pid)
        
        # Get focused element
        error, focused_element = ApplicationServices.AXUIElementCopyAttributeValue(
            ax_app, ApplicationServices.kAXFocusedUIElementAttribute, None
        )
        
        if error != 0 or not focused_element:
            print(f"Could not get focused element. Error: {error}")
            return False, None

        # Get Role
        error, role = ApplicationServices.AXUIElementCopyAttributeValue(
            focused_element, ApplicationServices.kAXRoleAttribute, None
        )
        print(f"Focused Element Role: {role}")
        
        # Get Value
        error_val, value = ApplicationServices.AXUIElementCopyAttributeValue(
            focused_element, ApplicationServices.kAXValueAttribute, None
        )
        # Handle AXValue being an AXValue object or other types, though usually it's a string for text fields
        print(f"Focused Element Value: {value}, Error: {error_val}")
        
        text_roles = ['AXTextArea', 'AXTextField']
        
        if role in text_roles:
            return True, value
            
        # Check if editable if role is not standard
        error_settable, is_settable = ApplicationServices.AXUIElementIsAttributeSettable(
            focused_element, ApplicationServices.kAXValueAttribute, None
        )
        is_editable = (error_settable == 0) and is_settable
        
        if is_editable:
             print("Element is editable.")
             return True, value

        return False, None
    except Exception as e:
        print(f"Error in get_focused_textbox_info: {e}")
        return False, None

def check_if_editing_command(current_text, new_command, history):
    print("--- Checking for Edit Command ---")
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
        
        Current Text in Textbox:
        {current_text}
        
        New Voice Command:
        {new_command}
        
        Task:
        Determine if the "New Voice Command" is an instruction to EDIT the "Current Text in Textbox".
        Examples of edit commands: "replace hello with hi", "delete the last word", "make it all caps", "change the first sentence".
        Examples of non-edit commands (just dictation): "hello world", "this is a test", "I want to go to the store".
        
        If it is an EDIT command, perform the edit on "Current Text in Textbox" and return the RESULTING TEXT ONLY. Do not output any reasoning or explanation.
        If it is NOT an edit command (it's just new text to be typed), return exactly the string "NO_EDIT".
        """
        
        print(f"Sending prompt to Gemini:\n{prompt}")
        response = model.generate_content(prompt)
        result = response.text.strip()
        print(f"Gemini Response: {result}")
        
        if result == "NO_EDIT":
            return False, None
        else:
            return True, result
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
        result = model.transcribe(filename)
        transcribed_text = result["text"].strip()
        if transcribed_text:
            text_to_paste = transcribed_text # Put text in shared variable
            
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
            
            is_textbox, current_value = get_focused_textbox_info()
            
            if is_textbox:
                print(f"Focused element is a textbox. Current value: '{current_value}'")
                is_edit = False
                edited_text = None
                
                if current_value and isinstance(current_value, str) and GEMINI_API_KEY:
                     print("Calling Gemini to check for edit command...")
                     # Use history excluding the current command (which is 'text')
                     # Note: 'text' was already added to transcription_history in transcribe_audio
                     history_context = transcription_history[:-1]
                     is_edit, edited_text = check_if_editing_command(current_value, text, history_context)
                else:
                    print(f"Skipping Gemini check. Value: {current_value}, Key Present: {bool(GEMINI_API_KEY)}")
                
                if is_edit and edited_text is not None:
                    print(f"Editing detected. Old: {current_value}, New: {edited_text}")
                    # Replace text
                    controller = Controller()
                    
                    # Select All
                    controller.press(keyboard.Key.cmd)
                    controller.press('a')
                    controller.release('a')
                    controller.release(keyboard.Key.cmd)
                    
                    # Copy new text
                    pyperclip.copy(edited_text)
                    
                    # Paste
                    controller.press(keyboard.Key.cmd)
                    controller.press('v')
                    controller.release('v')
                    controller.release(keyboard.Key.cmd)
                else:
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
