from google import genai
from google.genai import types

client = genai.Client()

QApair = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="Você é um especialista e Doutor em Ciência da Computação. Você terá o papel "
    ),
    contents="all"
)

print(QApair.text)