import os
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# Suppress unnecessary driver logging warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("-> Loading Local Speech Engine (Faster-Whisper Multilingual Small)...")
# Automatically downloads the model weights on the first run (~460MB)
# Using "small" provides significantly higher accuracy for conversational speech and accents.
asr_model = WhisperModel("small", device="cpu", compute_type="int8")


def listen_and_transcribe():
  """Captures speech dynamically using a pre-rolling buffer,

  surviving short pauses and cutting off only after sustained silence.
  """
  p = pyaudio.PyAudio()

  CHUNK = 1024
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 16000

  SILENCE_THRESHOLD = 500  # Adjust based on your microphone sensitivity
  PAUSE_LIMIT_SECONDS = 1.0  # Sustained silence duration before clipping speech block
  MAX_RECORD_SECONDS = 20.0  # Upper security safety timeout cap
  PRE_ROLL_CHUNKS = int(
      0.3 * RATE / CHUNK
  )  # Keep 300ms history so starting syllables aren't cut off

  try:
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
  except Exception as e:
    print(f"\n[ERROR] Could not access your microphone: {e}")
    p.terminate()
    return None

  stream.start_stream()
  print("\n[Status]: Listening... Speak now (Press Ctrl+C to exit).")

  frames = []
  pre_roll_buffer = []
  has_spoken = False
  silent_chunks = 0

  max_silent_chunks = int(PAUSE_LIMIT_SECONDS * RATE / CHUNK)
  max_total_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK)

  total_chunks_read = 0

  while total_chunks_read < max_total_chunks:
    data = stream.read(CHUNK, exception_on_overflow=False)
    total_chunks_read += 1

    # Calculate live room volume standard deviation
    audio_data = np.frombuffer(data, dtype=np.int16)
    amplitude = np.std(audio_data) if len(audio_data) > 0 else 0

    if amplitude > SILENCE_THRESHOLD:
      if not has_spoken:
        print("[Status]: Recording human speech...")
        # Prepend history so the start of words isn't clipped
        frames.extend(list(pre_roll_buffer))
      has_spoken = True
      silent_chunks = 0  # Reset trailing pause timer
      frames.append(data)
    else:
      if has_spoken:
        silent_chunks += 1
        frames.append(data)
      else:
        # Keep a rolling buffer before speech is detected
        pre_roll_buffer.append(data)
        if len(pre_roll_buffer) > PRE_ROLL_CHUNKS:
          pre_roll_buffer.pop(0)

    # Cut loop instantly when speaker finishes talking past the pause gap limit
    if has_spoken and silent_chunks > max_silent_chunks:
      break

  stream.stop_stream()
  stream.close()
  p.terminate()

  if not has_spoken:
    print("[Status]: No human speech detected.")
    return

  print("[Status]: Processing audio data pipeline...")
  raw_audio = b"".join(frames)
  audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

  # Transcribe audio segment using optimized beam search
  segments, info = asr_model.transcribe(
      audio_np,
      beam_size=5,  # Increased beam search size for optimal phrase context & accuracy
      vad_filter=True,
      task="translate",
      vad_parameters=dict(min_speech_duration_ms=250, max_speech_duration_s=20),
  )

  user_text = "".join([segment.text for segment in segments]).strip()

  # Print transcription output
  print("\n==================================================")
  print(
      f"[Detected Language]  : {info.language.upper()}"
      f" (Confidence: {info.language_probability:.2%})"
  )
  print(f"[Core Transcription] : {user_text}")
  print("==================================================\n")


if __name__ == "__main__":
  while True:
    try:
      listen_and_transcribe()
    except KeyboardInterrupt:
      print("\nExiting transcription script pipeline. Goodbye.")
      break
