import os
import time
import json
import requests
import pyaudio
import numpy as np
from gtts import gTTS
from faster_whisper import WhisperModel

# Suppress driver warning tags to keep console readable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ==========================================
# 1. SYSTEM CONFIGURATION & LINKS
# ==========================================
OLLAMA_URL = "http://localhost:11434/api/chat"  
LLM_MODEL = "llama3.2:1b"                      

# Using a standard system temporary directory path for the WAV audio block
WAV_OUTPUT_PATH = "/tmp/assistant_voice.wav"

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
    """Converts AI text into an high-quality voice file and forces execution via native system pathways."""
    print(f"\n[AI Assistant]: {text}")
    
    if os.path.exists(WAV_OUTPUT_PATH):
        try: os.remove(WAV_OUTPUT_PATH)
        except Exception: pass

    try:
        # 1. Generate the speech token with a clean Indian English accent profile
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
        
        # Save temporarily as an mp3 file
        temp_mp3 = "/tmp/temp_voice.mp3"
        tts.save(temp_mp3)
        
        # 2. Convert it locally to a standard WAV format using native system tools
        # This allows us to use Ubuntu's native audio drivers directly
        os.system(f"ffmpeg -y -i {temp_mp3} -acodec pcm_s16le -ac 1 -ar 16000 {WAV_OUTPUT_PATH} > /dev/null 2>&1")
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

        # 3. Play it using Ubuntu's standard native sound command (Forces Default Hardware Speakers)
        # paplay interacts directly with PulseAudio/PipeWire natively
        exit_status = os.system(f"paplay {WAV_OUTPUT_PATH} > /dev/null 2>&1")
        
        # Fallback to standard aplay if PulseAudio channel tags aren't responding
        if exit_status != 0:
            os.system(f"aplay {WAV_OUTPUT_PATH} > /dev/null 2>&1")
        
    except Exception as e:
        print(f"[Audio System Notification]: Core audio routing failed, trying text fallback: {e}")
        clean_text = text.replace('"', '\\"').replace("'", "\\'")
        os.system(f'spd-say "{clean_text}"')

def listen_to_user_offline():
    """Captures infinite speech length dynamically, surviving short pauses and cutting off only after sustained silence."""
    p = pyaudio.PyAudio()
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    SILENCE_THRESHOLD = 450   
    PAUSE_LIMIT_SECONDS = 1.0 
    MAX_RECORD_SECONDS = 20.0 
    
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
        
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.std(audio_data) if len(audio_data) > 0 else 0
        
        if amplitude > SILENCE_THRESHOLD:
            if not has_spoken:
                print("[Status]: Recording speech...")
            has_spoken = True
            silent_chunks = 0  
        else:
            if has_spoken:
                silent_chunks += 1
                
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

    segments, info = asr_model.transcribe(
        audio_np, 
        beam_size=2, 
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
        
    CHAT_HISTORY.append({"role": "user", "content": user_message})
    
    if len(CHAT_HISTORY) > 7:
        CHAT_HISTORY = [CHAT_HISTORY] + CHAT_HISTORY[-6:]

    payload = {
        "model": LLM_MODEL,
        "messages": CHAT_HISTORY,
        "stream": False,
        "options": {
            "num_ctx": 1024,        
            "temperature": 0.3,   #0.4  
            "top_p": 0.8,           
            "num_predict": 45,      
            "num_thread": 4         
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
    speak_local("Voice assistant engine online.")
    
    while True:
        user_input = listen_to_user_offline()
        
        if user_input:
            if any(trigger in user_input.lower() for trigger in ["exit application", "shutdown assistant", "goodbye"]):
                speak_local("Exiting assistant pipeline. Goodbye.")
                break
                
            ai_response = query_local_llm(user_input)
            speak_local(ai_response)
            
        time.sleep(0.2)

