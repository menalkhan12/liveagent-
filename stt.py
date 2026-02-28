import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio:
        transcription = client.audio.transcriptions.create(
            file=audio,
            model="whisper-large-v3"
        )
    text = transcription.text
    text = text.replace("ees", "fees")
    return text