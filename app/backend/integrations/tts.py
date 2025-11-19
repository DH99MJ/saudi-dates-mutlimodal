from elevenlabs.client import ElevenLabs
from elevenlabs import save
from dotenv import load_dotenv
import os
import uuid
load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVEN_LABS_TTS_API")
)

# Generates TTS audio from text and returns the path to the saved audio file
def generate_tts(text: str) -> str:
    audio = client.text_to_speech.convert(
    text=text,
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
    )

    filename = f"{uuid.uuid4()}.mp3"
    file_path = f"static/images/audio/{filename}"
    save(audio, file_path)
    return file_path
    
generate_tts("سلام عليكم يا شباب قاعد اشتغل على مشروع Deep learning and NLP ازهلوها يالربععععع")

