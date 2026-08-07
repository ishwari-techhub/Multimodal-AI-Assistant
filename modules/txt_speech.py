import asyncio
import edge_tts
import tempfile
import os


def text_to_speech(text):

    try:
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        output_file.close()

        async def generate_audio():

            communicate = edge_tts.Communicate(
                text,
                voice="en-US-AriaNeural"
            )

            await communicate.save(output_file.name)

        asyncio.run(generate_audio())

        return output_file.name

    except Exception as e:

        print("TTS Error:", e)
        return None