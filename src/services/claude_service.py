import anthropic

client = anthropic.Anthropic()

def callApiClaude(currID: int, question: str) -> str:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are an expert in Computer Science and Python documentation.\nAnswer the given question completely, technically, and directly, but without introductions like 'Sure', 'okay', 'certainly', etc.\nReturn ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the question provided; 'answer': a string that will be the returned answer.",
        messages=[
            {
                "role": "user",
                "content": f"ID: {currID}\n Question provided: {question}"
            }
        ]
    )
    return response.content[0].text
