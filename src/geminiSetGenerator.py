import json, os, dataFormat, time
from google import genai
from google.genai import types, errors
from pydantic import ValidationError

DATASET_GEMINI_PATH = os.path.join("data", "gemini_dataset.json") #path de destino
TIME_BETWEEN_CALLS = 10 #Espera (em segundos de cada solicitação)
client = genai.Client() 


def process_questions(goldenSet: list):

    # -- 1. LÓGICA DE PERSISTÊNCIA -- #
    #verifica se o caminho de destino existe, se sim, carrega com os dados já existentes, se nao, carrega com o dataset vazio  
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

    #parte da lógica de persistencia
    processedIDs = {item.id for item in datasetGemini.data}

    # -- 2. Loop principal do algoritmo -- #
    for i, rawItem in enumerate(goldenSet):

        currentID = rawItem.get('id') #id do item atual
        questionText = rawItem.get('question') #questao do item atual 

        #Pula se o id atual do goldenset ja foi processado pro geminiset
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
                        "You are an expert in Computer Science and Python documentation.", 
                        
                        "Answer the given question completely, technically, and directly, but without introductions like 'Sure', 'okay', 'certainly', etc.",

                        "Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'expectedQuestion': a string that will be the question provided; 'expectedAnswer': a string that will be the returned answer.",
                    ]
                ),
                contents=f"ID: {currentID}\n Question provided: {questionText}"
            )

            # -- 4. Validação e Parsing -- #
            # carrega apenas o campo .text da resposta do gemini
            QApair_dict = json.loads(QApair.text)

            #garante que o id do item atual pro geminiset sera o idCurrent do goldenset
            QApair_dict['id'] = currentID

            #validação e parsing por meio do pydantic
            validatedEntry = dataFormat.QAPairModel(QApair_dict)

            datasetGemini.data.append(validatedEntry) #add no datasetgemini
            processedIDs.add(currentID) #o id atual entra nos processados
            print(f"ID {currentID} Feito com sucesso")

        #verificações de erros
        except json.decoder.JSONDecodeError:
            print(f"ID {currentID}: Gemini retornou um JSON inválido.")
        except ValidationError as e:
            print(f"ID {currentID}: Erro de validação Pydantic: {e}")
        except errors.RateLimitError:
            print("Limite rate atingido, esperado 60 segundos...")
            time.sleep(60)
        except Exception as e:
            print(f"ID {currentID}: Erro desconhecido: {e}")


        # -- 5. Salvamento interminente -- #
        # O salvamento no dataset Gemini ocorre a cada 5 pares (sucesso ou falha)
        if (i + 1) % 5 == 0:
            with open(DATASET_GEMINI_PATH, 'w', encoding='utf-8') as f:
                f.write(datasetGemini.model_dump_json(indent=2))
        
        time.sleep(TIME_BETWEEN_CALLS)

    with open(DATASET_GEMINI_PATH, 'w', encoding='utf-8') as f:
        f.write(datasetGemini.model_dump_json(indent=2))


