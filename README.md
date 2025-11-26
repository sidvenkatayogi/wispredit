# Wispr Edit

**Wispr Edit** is a voice-to-text application inspired by [Wispr Flow](https://www.wispr.ai/). It brings fast, accurate voice typing to any application on your computer and adds a powerful "Smart Edit" feature that lets you modify text using natural language voice commands.

## Features

*   **Global Voice Typing:** Hold `Ctrl + Space` to record audio anywhere. Release to transcribe and paste the text instantly.
*   **Local Transcription:** Uses OpenAI's [Whisper](https://github.com/openai/whisper) model running locally on your machine for privacy and speed.
*   **Smart Editing:** Powered by Google Gemini. Instead of just typing, you can say commands like "Make that bullet points" or "Rewrite that to be more professional," and it will edit the text in your active window.
*   **Context Aware:** Intelligently distinguishes between dictation and edit commands based on context.

## Prerequisites

*   Python 3.8+
*   **FFmpeg**: Required for Whisper.
    *   macOS: `brew install ffmpeg`
    *   Windows: `choco install ffmpeg`
    *   Linux: `sudo apt install ffmpeg`
*   **Google Gemini API Key**: Required for the Smart Edit functionality. Get one [here](https://aistudio.google.com/app/apikey).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sidvenkatayogi/wispredit.git
    cd wispredit
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up configuration:**
    Create a `.env` file in the root directory and add your Gemini API key:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

## Usage

1.  **Start the application:**
    ```bash
    python wispr_edit.py
    ```
    You will see a message indicating the model is loading and the app is running.

2.  **Dictation:**
    *   Click into any text field (Notepad, Slack, Browser, VS Code, etc.).
    *   **Hold** `Ctrl + Space`.
    *   Speak your text.
    *   **Release** keys.
    *   The transcribed text will be typed out automatically.

3.  **Smart Editing:**
    *   Select some text (or just place your cursor in a text field with existing text).
    *   **Hold** `Ctrl + Space`.
    *   Say a command, e.g., *"Fix the grammar"* or *"Change hello to hi"*.
    *   **Release** keys.
    *   The application will detect the command and replace the text with the edited version.

## Configuration

You can adjust settings in `src/config.py`:

*   `HOTKEY`: Change the key combination (default: `{keyboard.Key.ctrl, keyboard.Key.space}`).
*   `MODEL_SIZE`: Change the Whisper model size (default: `tiny.en`). Larger models are more accurate but slower.
*   `SAMPLE_RATE`: Audio sample rate (default: `16000`).

## License

[MIT](LICENSE)
