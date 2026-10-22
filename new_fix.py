def listen_to_user_offline():
    """Opens the system microphone and captures user speech completely offline with static filtering."""
    import numpy as np # Ensure numpy is imported
    
    p = pyaudio.PyAudio()
    
    # 1. Lower buffer size (from 4000 to 1024) for faster, lower-latency chunks
    CHUNK_SIZE = 1024
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNK_SIZE)
    except Exception as e:
        print(f"\n[ERROR] Could not access your microphone: {e}")
        print("Ensure a USB or built-in microphone is plugged into your device.")
        p.terminate()
        return None
        
    stream.start_stream()
    print("\n[Status]: Listening... Speak into your microphone now.")
    
    # 2. Define your static threshold. 
    # 300-600 works well for most mics. Increase it if it still hears static.
    STATIC_THRESHOLD = 500 
    
    user_text = ""
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        if len(data) == 0:
            break
            
        # 3. Convert raw bytes to numbers to read the volume level
        audio_array = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array**2))
        
        # 4. If it's just room static, ignore it entirely and don't feed it to Vosk
        if rms < STATIC_THRESHOLD:
            continue
            
        # 5. Only feed the audio engine when real noise/speech happens
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            user_text = result.get("text", "").strip()
            if user_text: 
                break

    # Final cleanup check for any leftover speech trailing off
    partial_result = json.loads(rec.FinalResult())
    if not user_text:
        user_text = partial_result.get("text", "").strip()

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    if user_text:
        print(f"[You Said]: {user_text}")
    return user_text


"""
old code 
"""

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
