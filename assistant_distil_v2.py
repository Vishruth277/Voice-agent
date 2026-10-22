import os
import time
import json
import requests
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel

# Suppress driver warning tags to keep console readable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ==========================================
# 1. SYSTEM CONFIGURATION & LINKS
# ==========================================
OLLAMA_URL = "http://localhost:11434/api/chat"  
LLM_MODEL = "llama3.2:1b"                      
PIPER_MODEL = "en_US-lessac-medium.onnx"

if not os.path.exists(PIPER_MODEL):
    raise FileNotFoundError(f"Missing Piper voice file! Ensure '{PIPER_MODEL}' is in this folder.")

print("-> Loading Local Speech Engine (Faster-Whisper Tiny with Native VAD)...")
asr_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# ==========================================
# 2. CONVERSATIONAL MEMORY STATE
# ==========================================
CHAT_HISTORY = [
    {
        "role": "system",
        "content": (
            "You are a helpful, professional corporate voice assistant appliance. "
            "The user input comes from a live microphone speech decoder. Ignore minor typos. "
            "Crucial: Remember details, contexts, and names mentioned earlier in the chat to hold a continuous conversation. "
            "Keep your responses friendly, brief, and under two sentences."
        )
    }
]

# ==========================================
# 3. CORE AUDIO PIPELINE FUNCTIONS
# ==========================================

def speak_local(text):
    """Converts AI text into speech using local Piper binary and plays it natively via PyAudio."""
    print(f"\n[AI Assistant]: {text}")
    clean_text = text.replace('"', '\\"') 
    
    os.system(f'echo "{clean_text}" | ./piper/piper --model {PIPER_MODEL} --output_file response.wav')
    
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
        
        if os.path.exists("response.wav"):
            os.remove("response.wav")
            
    except Exception as e:
        print(f"[Audio Playback System Warning]: Natively playing voice failed: {e}")

def listen_to_user_offline():
    """Captures infinite speech length dynamically, surviving short pauses and cutting off only after sustained silence."""
    p = pyaudio.PyAudio()
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    # Adaptive configuration to allow natural speech pauses
    SILENCE_THRESHOLD = 450   # Amplitude threshold (increase if it triggers on background fans)
    PAUSE_LIMIT_SECONDS = 1.5 # Allows you to pause for up to 1.5 seconds mid-sentence without cutting off
    MAX_RECORD_SECONDS = 20.0 # Absolute safety cap for a single sentence block
    
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"\n[ERROR] Could not access your microphone: {e}")
        p.terminate()
        return None
        
    stream.start_stream()
    print("\n[Status]: Listening... Speak now (You can take pauses).")
    
    frames = []
    has_spoken = False
    silent_chunks = 0
    
    max_silent_chunks = int(PAUSE_LIMIT_SECONDS * RATE / CHUNK)
    max_total_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK)

    while len(frames) < max_total_chunks:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Audio energy math
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.std(audio_data) if len(audio_data) > 0 else 0
        
        if amplitude > SILENCE_THRESHOLD:
            if not has_spoken:
                print("[Status]: Recording speech...")
            has_spoken = True
            silent_chunks = 0  # User is talking, reset the pause counter
        else:
            if has_spoken:
                silent_chunks += 1
                
        # Only break if the user has spoken, and then remained completely silent past the pause limit
        if has_spoken and silent_chunks > max_silent_chunks:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    if not has_spoken:
        print("[Status]: No speech detected.")
        return ""

    print("[Status]: Filtering noise and processing speech...")
    raw_audio = b"".join(frames)
    audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

    # vad_filter=True drops background room static hum
    segments, info = asr_model.transcribe(
        audio_np, 
        beam_size=2, # Cut down from 3 to 2 to minimize CPU processing time
        vad_filter=True,
        vad_parameters=dict(min_speech_duration_ms=200, max_speech_duration_s=20)
    )
    
    user_text = "".join([segment.text for segment in segments]).strip()
    print(f"[You Said]: {user_text}")
    return user_text

def query_local_llm(user_message):
    """Sends conversational history to Ollama with tightly optimized CPU execution overrides."""
    global CHAT_HISTORY
    if not user_message.strip():
        return ""
        
    # Append fresh turn
    CHAT_HISTORY.append({"role": "user", "content": user_message})
    
    # Context sliding window: Keep system instruction rules at, plus the last 6 active turns.
    # This prevents the context from bloating and slowing down your EliteBook CPU.
    if len(CHAT_HISTORY) > 7:
        CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-6:]

    # Highly optimized Ollama options payload
    payload = {
        "model": LLM_MODEL,
        "messages": CHAT_HISTORY,
        "stream": False,
        "options": {
            "num_ctx": 1024,        # Shrunk from default 2048+ to save system RAM usage
            "temperature": 0.4,     # Balances conversational awareness with factual focus
            "top_p": 0.8,           
            "num_predict": 45,      # Forces model cutoff quickly so it doesn't ramble
            "num_thread": 4         # Pins processing across 4 CPU compute execution structures
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=15)
        ai_reply = response.json().get("message", {}).get("content", "").strip()
        CHAT_HISTORY.append({"role": "assistant", "content": ai_reply})
        return ai_reply
    except Exception as e:
        return "System error: I cannot reach my local server window."

# ==========================================
# 4. MAIN RUNTIME EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    speak_local("System diagnostic successful. Voice engine online.")
    
    while True:
        user_input = listen_to_user_offline()
        
        if user_input:
            if any(trigger in user_input.lower() for trigger in ["exit application", "shutdown assistant", "goodbye"]):
                speak_local("Exiting assistant pipeline. Goodbye.")
                break
                
            ai_response = query_local_llm(user_input)
            speak_local(ai_response)
            
        time.sleep(0.2)

