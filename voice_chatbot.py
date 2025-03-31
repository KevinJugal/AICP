import ollama
import whisper
import sounddevice as sd
import numpy as np
import os
import datetime
import scipy.io.wavfile as wav
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

# Load models
model = whisper.load_model("base")
tts = TTS(model_name="tts_models/en/ljspeech/vits", gpu=True)

# Define folder to save conversations
SAVE_FOLDER = r"C:\Users\jugal\Documents\Projects\idk\files"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def record_audio(duration=5, samplerate=16000):
    print("Recording... Speak Now!")
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
        sd.wait()
        print("Recording Finished")
        return np.squeeze(audio)
    except Exception as e:
        print(f"Error while recording: {e}")
        return None

def save_audio(filename, audio_data, samplerate=16000):
    try:
        audio_data_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        wav.write(filename, samplerate, audio_data_int16)
        print(f"Audio saved as {filename}")
    except Exception as e:
        print(f"Error saving audio: {e}")

def transcribe_audio(filename):
    if not os.path.exists(filename):
        print("Error: Audio file not found!")
        return None
    try:
        result = model.transcribe(filename)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def generate_response(user_input):
    response = ollama.chat(model="mistral", messages=[{"role":"user","content":user_input}])
    return response['message']['content']

def speak_text(text):
    """Converts text to speech, saves the audio, and plays it."""
    try:
        tts_file = os.path.join(SAVE_FOLDER, "ai_voice.wav")

        # If the file exists, remove it before saving a new one
        if os.path.exists(tts_file):
            os.remove(tts_file)

        # Generate TTS output
        tts.tts_to_file(text=text, file_path=tts_file)

        # Wait a bit to ensure file is written before playing
        while not os.path.exists(tts_file):
            pass  

        # Load and play the audio
        audio = AudioSegment.from_wav(tts_file)
        play(audio)

    except Exception as e:
        print(f"Error in text-to-speech: {e}")

while True:
    # Record and save audio
    audio_data = record_audio()
    if audio_data is None:
        continue

    audio_filename = os.path.join(SAVE_FOLDER, f"speech_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    save_audio(audio_filename, audio_data)

    # Transcribe audio
    user_text = transcribe_audio(audio_filename)
    if user_text is None or user_text.strip() == "":
        print("No speech detected. Try again.")
        continue

    print(f"You said: {user_text}")

    # Exit condition
    if "exit" in user_text.lower():
        print("Exiting...")
        break

    # Get AI response
    ai_response = generate_response(user_text)
    print(f"AI: {ai_response}")

    # Save conversation to a new file
    convo_filename = os.path.join(SAVE_FOLDER, f"convo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(convo_filename, "w", encoding="utf-8") as f:
        f.write(f"User: {user_text}\n")
        f.write(f"AI: {ai_response}\n")
    
    print(f"Conversation saved to {convo_filename}")

    # Speak AI response
    speak_text(ai_response)
