import os
import sys
import tempfile

import numpy as np

PORTAUDIO_PATH = "/home/Student/.local/portaudio/usr/lib/x86_64-linux-gnu/libportaudio.so.2.0.0"

if PORTAUDIO_PATH and os.path.exists(PORTAUDIO_PATH):
    import ctypes.util
    _real_find_library = ctypes.util.find_library

    def _find_library_patched(name):
        if name == "portaudio":
            return PORTAUDIO_PATH
        return _real_find_library(name)

    ctypes.util.find_library = _find_library_patched

import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import pygame

from webapp.companion import (
    MODEL,
    load_memories,
    save_memories,
    build_system_prompt,
    send_with_retry,
    wrap_up,
)
from google import genai
from google.genai import types


SAMPLE_RATE = 16000
INPUT_DEVICE = None  
CHUNK_SECONDS = 0.1
SILENCE_AFTER_SPEECH = 1.2   
MAX_RECORD_SECONDS = 20   
GIVE_UP_AFTER = 8        
THRESHOLD_MULTIPLIER = 3     
MIN_THRESHOLD = 150          


def _rms(chunk):
    return np.sqrt(np.mean(chunk.astype(np.float64) ** 2))


def _calibrate_ambient_noise(duration=0.5):
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=INPUT_DEVICE,
    )
    sd.wait()
    return _rms(frames)


def listen():
    ambient = _calibrate_ambient_noise()
    threshold = max(ambient * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)

    print("Listening... (go ahead, I'm here)")

    chunk_frames = int(CHUNK_SECONDS * SAMPLE_RATE)
    silence_limit = int(SILENCE_AFTER_SPEECH / CHUNK_SECONDS)
    max_chunks = int(MAX_RECORD_SECONDS / CHUNK_SECONDS)
    timeout_chunks = int(GIVE_UP_AFTER / CHUNK_SECONDS)

    chunks = []
    speech_started = False
    silence_run = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=INPUT_DEVICE
    )
    stream.start()
    try:
        for i in range(max_chunks):
            data, _ = stream.read(chunk_frames)
            level = _rms(data)

            if level > threshold:
                speech_started = True
                silence_run = 0
                chunks.append(data.copy())
            elif speech_started:
                silence_run += 1
                chunks.append(data.copy())
                if silence_run > silence_limit:
                    break
            elif i > timeout_chunks:
                break  
    finally:
        stream.stop()
        stream.close()

    if not speech_started or not chunks:
        return None

    audio_np = np.concatenate(chunks)
    peak = np.abs(audio_np).max()
    print(f"[debug: peak volume = {peak} out of 32767]")

    audio_bytes = audio_np.tobytes()
    audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)  # 2 bytes = int16

    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data)
        print(f"You (heard): {text}")
        return text
    except sr.UnknownValueError:
        print("[didn't catch that — try again]")
        return None
    except sr.RequestError as e:
        print(f"[speech recognition service error: {e}]")
        return None


def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    try:
        gTTS(text=text, lang="en").save(path)
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"[couldn't play audio: {e}]")
    finally:
        pygame.mixer.quit()
        os.remove(path)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set your GEMINI_API_KEY environment variable first.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    memories = load_memories()

    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(memories),
        ),
    )

    print("Nasha is here — say something. (say 'quit' to exit)\n")
    if memories:
        print(f"[remembers {len(memories)} thing(s) about you]\n")

    while True:
        user_input = listen()
        if user_input is None:
            continue

        if user_input.strip().lower() in ("quit", "exit"):
            print("Nasha: Talk soon.")
            speak("Talk soon.")
            wrap_up(client, chat, memories)
            break

        response = send_with_retry(chat, user_input)
        if response is None:
            continue

        print(f"Nasha: {response.text}\n")
        speak(response.text)


if __name__ == "__main__":
    main()