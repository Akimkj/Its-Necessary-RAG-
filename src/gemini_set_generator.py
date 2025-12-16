import os, re, json, data_format, time
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

client = genai.Client()

def generate_with_retry(max_retries=5, base_delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=[
                        "Você é um especialista em documentação Python. Baseada na pergunta que vai ser passada para você, você deve fornecer uma resposta completa e de forma clara, ao longo da sua explicação, se necessário, dê exemplos em códigos para exemplificar e deixar mais claro o entendimento do usuário.",
                        "Na sua resposta final, Você deve retornar um JSON com três partes principais: id: um inteiro que representa a identidade do par Pergunta-Resposta; expectedQuestion: uma string que será justamente a pergunta que vou te passar; expectedAnswer: uma string que será a sua resposta para a pegunta.",
                    ]
                ),
                contents="Por que usamos __init__ em classes Python?"
            )

        except ServerError as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries:
                wait_time = base_delay ** attempt
                print(f"[Retry {attempt}/{max_retries}] Modelo sobrecarregado. Tentando novamente em {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("Limite de cota atingido. Aguardando 50 segundos conforme instrução da API...")
                time.sleep(50)
            else:
                raise

QApair = generate_with_retry()


def clean_block_json(text):
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})", text)
    if match:
        return match.group(1) or match.group(2)
    
    return text.strip()

try:

    json_string = clean_block_json(QApair.text)

    qa_data_dict = json.loads(json_string)

    new_qa_entry = data_format.QAPairModel(**qa_data_dict)

    DATASET_PATH = "datasets/qa_dataset_gemini.json"

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        dataset = data_format.QADataSet(**existing_data)
    else:
        dataset = data_format.QADataSet()

    new_qa_entry.id = len(dataset.data) + 1

    dataset.data.append(new_qa_entry)

    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        f.write(dataset.model_dump_json(indent=2))


except json.JSONDecodeError as e:
    print(f"Erro ao decodificar JSON: Erro: {e}")
except Exception as e:
    print(f"Erro de validação Pydantic ou outro erro: {e}")



