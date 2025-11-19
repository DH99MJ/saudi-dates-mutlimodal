from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
  api_key=os.getenv("OPEN_AI_API_KEY")
)


def generate_text(input: str, model="gpt-5-nano",) -> str :
    response = client.responses.create(
        model=model,
        input=input
    )
    

    output_message = response.output[1]                 
    text = output_message.content[0].text              

    print(text)
    return text



generate_text("""
    تكلم باللهجة السعودية، ووصف لي تمر.
    ركّز على اللون والطعم وسمعته عند أهل القصيم والأحساء.
    الأسلوب يكون بسيط، شعبي، لطيف.
    """)


