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
# Internal VAD filter initialization
asr_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# ==========================================
# 2. CONVERSATIONAL MEMORY STATE
# ==========================================
CHAT_HISTORY = [
    {
        "role": "system",
        "content": (
            "You are a helpful, professional corporate voice assistant appliance. "
            "Ignore minor spelling mistakes or missing punctuation from speech inputs. "
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
    """Captures a steady block of audio directly, allowing faster-whisper's internal engine to process speech regions."""
    p = pyaudio.PyAudio()
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 4.5  # Captures a conversational phrase block cleanly
    
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"\n[ERROR] Could not access your microphone: {e}")
        p.terminate()
        return None
        
    stream.start_stream()
    print("\n[Status]: Listening...")
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("[Status]: Filtering noise and processing speech...")
    raw_audio = b"".join(frames)
    audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

    # vad_filter=True invokes the native backend handler to drop static/room hum
    segments, info = asr_model.transcribe(
        audio_np, 
        beam_size=3,
        vad_filter=True,
        vad_parameters=dict(min_speech_duration_ms=250, max_speech_duration_s=5)
    )
    
    user_text = "".join([segment.text for segment in segments]).strip()
    
    print(f"[You Said]: {user_text}")
    return user_text

def query_local_llm(user_message):
    """Sends the conversation history to Ollama."""
    global CHAT_HISTORY
    if not user_message.strip():
        return ""
        
    CHAT_HISTORY.append({"role": "user", "content": user_message})
    
    if len(CHAT_HISTORY) > 5:
        CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-4:]

    payload = {
        "model": LLM_MODEL,
        "messages": CHAT_HISTORY,
        "stream": False,
        "options": {
            "temperature": 0.3,     
            "top_p": 0.7,           
            "num_predict": 50,      
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

