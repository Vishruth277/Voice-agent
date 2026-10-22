import os
import time
import json
import requests
import pyaudio
import wave
from vosk import Model, KaldiRecognizer

# Suppress driver warning tags to keep console readable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ==========================================
# 1. SYSTEM CONFIGURATION & LINKS
# ==========================================
OLLAMA_URL = "http://localhost:11434/api/chat"  # Changed from /generate to /chat for memory
LLM_MODEL = "llama3.2:1b"                      # Switch to "llama3.2:1b" when you want to try it
PIPER_MODEL = "en_US-lessac-medium.onnx"
VOSK_MODEL_PATH = "vosk-model-small-en-in-0.4"  # Using your local Indian English model

# Verify that the necessary offline models exist before booting up
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(f"Missing Vosk model folder! Download and extract '{VOSK_MODEL_PATH}' inside this folder.")
if not os.path.exists(PIPER_MODEL):
    raise FileNotFoundError(f"Missing Piper voice file! Ensure '{PIPER_MODEL}' is in this folder.")

print("-> Loading Local Speech Engine...")
vosk_model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(vosk_model, 16000)

# ==========================================
# 2. CONVERSATIONAL MEMORY STATE
# ==========================================
# This holds the system instructions and appends all user/assistant dialogue turns dynamically
CHAT_HISTORY = [
    {
        "role": "system",
        "content": (
            "You are a helpful, professional corporate voice assistant appliance. "
            "The user input text is derived from an imperfect local speech-to-text decoder. "
            "Ignore minor phonetic spelling mistakes or missing punctuation. "
            "Keep your responses friendly, strictly conversational, brief, and under two sentences."
        )
    }
]

# ==========================================
# 3. CORE AUDIO PIPELINE FUNCTIONS
# ==========================================

def speak_local(text):
    """Converts AI text into speech using local Piper binary and plays it natively via PyAudio."""
    print(f"\n[AI Assistant]: {text}")
    clean_text = text.replace('"', '\\"') # Escape quotes for bash command execution safety
    
    # Triggering your local standalone binary package folder directly
    os.system(f'echo "{clean_text}" | ./piper/piper --model {PIPER_MODEL} --output_file response.wav')
    
    # Native Python Audio Playback Loop (Bypasses desktop sound card driver blocks)
    try:
        wf = wave.open("response.wav", 'rb')
        p_out = pyaudio.PyAudio()
        
        stream_out = p_out.open(
            format=p_out.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        while len(data) > 0:
            stream_out.write(data)
            data = wf.readframes(chunk_size)
            
        stream_out.stop_stream()
        stream_out.close()
        p_out.terminate()
        wf.close()
        
        # Housekeeping: Clear file out to ensure no disk pollution
        if os.path.exists("response.wav"):
            os.remove("response.wav")
            
    except Exception as e:
        print(f"[Audio Playback System Warning]: Natively playing voice failed: {e}")
        
def listen_to_user_offline():
    """Opens the system microphone and captures user speech completely offline."""
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    except Exception as e:
        print(f"\n[ERROR] Could not access your microphone: {e}")
        print("Ensure a USB or built-in microphone is plugged into your device.")
        p.terminate()
        return None
        
    stream.start_stream()
    print("\n[Status]: Listening... Speak into your microphone now.")
    
    user_text = ""
    while True:
        data = stream.read(2000, exception_on_overflow=False)
        if len(data) == 0:
            break
            
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            user_text = result.get("text", "").strip()
            if user_text: 
                break

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print(f"[You Said]: {user_text}")
    return user_text


def query_local_llm(user_message):
    """Sends the conversation history to Ollama with strict parameter overrides."""
    global CHAT_HISTORY
    
    # 1. Clear text injection fallback
    if not user_message.strip():
        return ""
        
    # 2. Append the fresh user utterance
    CHAT_HISTORY.append({"role": "user", "content": user_message})
    
    # 3. Context window control: Keep only the system prompt + last 4 turns
    # This prevents old hallucinations from polluting the model's active memory
    if len(CHAT_HISTORY) > 5:
        # Keep the original system prompt at index 0, and append the latest turns
        CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-4:]

    # 4. Strict structural parameters payload
    payload = {
        "model": LLM_MODEL,
        "messages": CHAT_HISTORY,
        "stream": False,
        # --- THE ANTI-HALLUCINATION CONFIGURATION ---
        "options": {
            "temperature": 0.3,     # Drops creativity down (0.0 = completely deterministic, 1.0 = highly creative)
            "top_p": 0.7,           # Restricts the model to only select the most logical, high-probability words
            "num_predict": 50,      # Force stops the model if it tries to babble past 2 short sentences
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=15)
        ai_reply = response.json().get("message", {}).get("content", "").strip()
        
        # 5. Save clean reply back into the conversation memory frame
        CHAT_HISTORY.append({"role": "assistant", "content": ai_reply})
        return ai_reply
        
    except Exception as e:
        return "System error: I cannot reach my local server window."

# ==========================================
# 4. MAIN RUNTIME EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    speak_local("System diagnostic successful. Voice assistant engine is online.")
    
    while True:
        # Step A: Wait and listen for user voice
        user_input = listen_to_user_offline()
        
        if user_input:
            # Check for system termination trigger words
            if "exit application" in user_input or "shutdown assistant" in user_input or "goodbye" in user_input:
                speak_local("Exiting assistant pipeline. Goodbye.")
                break
                
            # Step B: Process multi-turn text prompt arrays through Ollama Chat API
            ai_response = query_local_llm(user_input)
            
            # Step C: Turn the context-rich reply text into spoken audio waves
            speak_local(ai_response)
            
        time.sleep(0.2)

