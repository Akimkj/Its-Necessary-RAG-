import json
import os
import data_format
from google import genai
from google.genai import types

client = genai.Client()

QApair = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=[
            "Você é um especialista e Doutor em Ciência da Computação. Sua especialidade é em documentação Python, acima de tudo. Baseada na pergunta que vai ser passada para você, é para você responder de forma clara e objetiva, como um verdadeiro profissional na área.",
            "Na sua resposta final, Você deve retornar um JSON com três partes principais: id: um inteiro que representa a identidade do par Pergunta-Resposta; expectedQuestion: uma string que será justamente a pergunta que vou te passar; expectedAnswer: uma string que será a sua resposta para a pegunta.",
        ]
    ),
    contents="como calcular bhaskara em python?"
)

try:
    qa_data_dict = json.loads(QApair.text)

    new_qa_entry = data_format.QAPairModel(**qa_data_dict)

    DATASET_PATH = "qa_dataset.json"

    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        dataset = data_format.QADataSet(**existing_data)
    else:
        dataset = data_format.QADataSet()

    dataset.data.append(new_qa_entry)

    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        f.write(dataset.model_dump_json(indent=2))
except json.JSONDecodeError as e:
    print(f"Erro ao decodificar JSON: Erro: {e}")
except Exception as e:
    print(f"Erro de validação Pydantic ou outro erro: {e}")



