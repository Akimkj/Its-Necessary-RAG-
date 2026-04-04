from google import genai
from google.genai import types


client = genai.Client() 


def callApiGemini(currID: int, question) -> str:
    result = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    max_output_tokens=4096,
                    system_instruction=[
                        "You are an expert in Computer Science and Python documentation.", 
                        "Your task is to answer the provided question in a complete, technical, and detailed manner.",
                        "Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the question provided; 'answer': a string that will be the returned answer.",
                        "Rules: 1. Do NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input. If not provided, use 0; 4. The 'question' must exactly match the input question; 5. The 'answer' must be detailed, technical, and well-structured; 6. Escape all special characters properly inside JSON strings; 7. Do NOT include markdown; 8. Do NOT omit any required field."
                    ]
                ),
                contents=f"ID: {currID}\n Question provided: {question}",
            )
    return result.text or ""