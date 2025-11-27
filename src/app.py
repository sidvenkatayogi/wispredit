import threading
import time
import pyperclip
import os
from pynput import keyboard

from .config import HOTKEY, GEMINI_API_KEY, SAMPLE_RATE
from .audio import start_recording_stream, stop_recording_stream
from .transcription import load_model, transcribe_audio_frames
from .llm import check_if_editing_command
from .clipboard import select_all, copy_selection, paste_text, move_cursor_right

class WsprEditApp:
    def __init__(self):
        self.state_recording = False
        
        # Shared state
        self.text_to_paste = None
        self.transcription_history = []
        self.current_keys = set()

    def transcribe_worker(self, frames):
        """Background worker for transcription."""
        transcribed_text = transcribe_audio_frames(frames)
        if transcribed_text:
            self.text_to_paste = transcribed_text
            
            # Update history
            self.transcription_history.append(transcribed_text)
            if len(self.transcription_history) > 3:
                self.transcription_history.pop(0)

    def start_recording_thread(self):
        """Starts the recording in a separate thread (since it blocks)."""
        start_recording_stream(SAMPLE_RATE)

    def check_and_paste(self):
        """Checks for text and pastes it from the main thread."""
        if self.text_to_paste:
            text = self.text_to_paste
            self.text_to_paste = None # Clear the variable
            
            original_clipboard = pyperclip.paste()
            
            # Select all and copy to get context
            select_all()
            current_value = copy_selection()
            
            is_edit = False
            edited_text = None
            
            if current_value and isinstance(current_value, str) and GEMINI_API_KEY:
                    history_context = self.transcription_history[0:1]
                    is_edit, edited_text = check_if_editing_command(current_value, text, history_context)
            else:
                print(f"Skipping Gemini check. Value: {current_value}, Key Present: {bool(GEMINI_API_KEY)}")
            
            if is_edit and edited_text is not None:
                # print(f"Editing detected. Old: {current_value}, New: {edited_text}")
                print(f"Editing command detected: Command: {text}")
                print()
                # print(f"Editing detected. Old: {current_value}, New: {edited_text}")
                # We already Selected All, but just in case user had a long audio message and maybe clicked out, just select all again
                select_all()
                paste_text(edited_text)
            else:
                print(f"Pasting: {text}")
                print()
                
                # Move cursor to end (Right Arrow) bc we have currently selected all
                move_cursor_right()

                # add a space for user to continue (only needed for no edit)
                paste_text(text + " ")
            
            # Restore clipboard to original state
            time.sleep(0.2)
            pyperclip.copy(original_clipboard)

    def update_menu_state(self):
        print(f"Recording: {'ON' if self.state_recording else 'OFF'}")

    def on_press(self, key):
        """Handle key press events."""
        if key in HOTKEY:
            self.current_keys.add(key)
            if all(k in self.current_keys for k in HOTKEY) and not self.state_recording:
                self.state_recording = True
                # self.update_menu_state()
                threading.Thread(target=self.start_recording_thread).start()

    def on_release(self, key):
        """Handle key release events."""
        if all(k in self.current_keys for k in HOTKEY):
            self.state_recording = False
            # self.update_menu_state()
            frames = stop_recording_stream()
            threading.Thread(target=self.transcribe_worker, args=(frames,)).start()
        
        if key in self.current_keys:
            self.current_keys.remove(key)

    def run_hotkey_listener(self):
        """Runs the keyboard listener."""
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def run(self):
        """Main loop to keep the application running."""
        print("Wspr Edit running... Hold Ctrl + Space to record. Press Ctrl + C to end")
        try:
            while True:
                self.check_and_paste()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Exiting...")
