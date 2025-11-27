import google.generativeai as genai
from .config import GEMINI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def check_if_editing_command(current_text, new_command, history):
    """
    Checks if the new command is an edit instruction for the current text.
    Returns (is_edit, result_text).
    """
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
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
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