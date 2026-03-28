import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

def callApiDeepseek(CurrID: int, question: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are an expert in Computer Science and Python documentation.\n"
                    "Answer the given question completely, technically, and directly, without introductions.\n"
                    "Return ONLY a valid JSON with keys: 'id': an integer; 'question': a string; 'answer': a string."
                )
            },
            {"role": "user", "content": f"ID: {CurrID}\n Question provided: {question}"},
        ],
        stream=False,
        max_tokens=1024
    )       
    return response.choices[0].message.content