import json, os, dataFormat, time
from google import genai
from google.genai import types
from pydantic import ValidationError

DATASET_GEMINI_PATH = "datasets\geminiQA.json" #mudar nome
TIME_BETWEEN_CALLS = 10
client = genai.Client()


def process_questions(goldenSet: list):

    # -- 1. LÓGICA DE PERSISTÊNCIA -- #
    if (os.path.exists(DATASET_GEMINI_PATH)):
        with open(DATASET_GEMINI_PATH, 'r', encoding='utf-8') as f:
            try:

                rawData = json.load(f)
                datasetGemini = dataFormat.QADataSet(**rawData)
            except (json.decoder.JSONDecodeError, ValidationError):
                print("Arquivo corrompido ou vazio, criando um dataset novo vazio...")
                datasetGemini = dataFormat.QADataSet()
    else:
        datasetGemini = dataFormat.QADataSet()

    processedIDs = {item.id for item in datasetGemini.data}

    # -- 2. Loop principal do algoritmo -- #
    for i, rawItem in enumerate(goldenSet):

        currentID = rawItem.get('id') #verificar nome !!!!!!
        questionText = rawItem.get('goldenQuestion') #verificar nome !!!!!!

        if (currentID in processedIDs):
            continue

        print(f"{currentID} Processando com gemini...")

        try:
            # -- 3. CONFIGURAÇÃO API GEMINI --
            QApair = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    system_instruction=[
                        "Você é um especialista em Ciência da Computação e em documentação Python.", 
                        
                        "Responda a pergunta fornecida de forma completa, técnica e direta, mas sem introduções de 'Claro', 'tudo bem', 'com certeza' etc.",

                        "Retorne APENAS um JSON válido com chaves: 'id': um inteiro que representa a identidade do par Pergunta-Resposta; 'expectedQuestion': uma string que será justamente a pergunta fornecida; 'expectedAnswer': uma string que será resposta retornada.",
                    ]
                ),
                contents=f"ID: {currentID}\n Pergunta fornecida: {questionText}"
            )

            # -- 4. Validação e Parsing -- #
            QApair_dict = json.loads(QApair.text)

            QApair_dict['id'] = currentID

            validatedEntry = dataFormat.QAPairModel(QApair_dict)

            datasetGemini.data.append(validatedEntry)
            processedIDs.add(currentID)
            print(f"ID {currentID} Feito com sucesso")

        except json.decoder.JSONDecodeError:
            print(f"ID {currentID}: Gemini retornou um JSON inválido.")
        except ValidationError as e:
            print(f"ID {currentID}: Erro de validação Pydantic: {e}")
        except Exception as e:
            print(f"ID {currentID}: Erro desconhecido: {e}")


        # -- 5. Salvamento interminente
        # O salvamento no dataset Gemini ocorre a cada 5 pares (sucesso ou falha)
        if (i + 1) % 5 == 0:
            with open(DATASET_GEMINI_PATH, 'w', encoding='utf-8') as f:
                f.write(datasetGemini.model_dump_json(indent=2))
        
        time.sleep(TIME_BETWEEN_CALLS)

    with open(DATASET_GEMINI_PATH, 'w', encoding='utf-8') as f:
        f.write(datasetGemini.model_dump_json(indent=2))


