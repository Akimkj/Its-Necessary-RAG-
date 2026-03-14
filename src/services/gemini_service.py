from google import genai
from google.genai import types


client = genai.Client() 


def callApiGemini(currID: int, question):
    result = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    system_instruction=[
                        "You are an expert in Computer Science and Python documentation.", 
                        "Answer the given question completely, technically, and directly, but without introductions like 'Sure', 'okay', 'certainly', etc.",
                        "Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the question provided; 'answer': a string that will be the returned answer."
                    ]
                ),
                contents=f"ID: {currID}\n Question provided: {question}"
            )
    return result.text