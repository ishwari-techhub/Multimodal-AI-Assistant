import os
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API")
)


def record_voice():

    audio = mic_recorder(
        start_prompt="🎙️ Start Speaking",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        format="wav",
        key="voice_recorder"
    )

    return audio


def transcribe_voice(audio):

    try:

        transcription = client.audio.transcriptions.create(
            file=(
                "voice.wav",
                audio["bytes"]
            ),
            model="whisper-large-v3",
            response_format="text"
        )

        return transcription

    except Exception as e:

        print("Whisper Error:", e)

        return None

def voice_chat(text):

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Give clear and concise answers."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:

        print("Voice Chat Error:", e)

        return None