import json, os, time
from . import dataFormat, utils
from google.genai import errors
from pydantic import ValidationError

DATASET_GEMINI_PATH = os.path.join("data", "gemini_dataset_v2.json") #path de destino
TIME_BETWEEN_CALLS = 10 #Espera (em segundos de cada solicitação)


def process_questions(goldenSet: list):

    # -- 1. LÓGICA DE PERSISTÊNCIA -- #
    #Carrega os dados existentes no caminho informado  
    rawData = utils.loadData(DATASET_GEMINI_PATH)

    if not rawData: #se não existir dados, carrega um dataset vazio
        print("Criando novo dataset vazio...")
        datasetGemini = dataFormat.QADataSet()
    else:
        try: #Se existir dados, carrega um dataset com os dados ja existentes
            datasetGemini = dataFormat.QADataSet(**rawData)
        except ValidationError:
            print("Arquivo com formato inválido, resetando...")
            datasetGemini = dataFormat.QADataSet()
    

    #parte da lógica de persistencia
    processedIDs = {item.id for item in datasetGemini.data}

    # -- 2. Loop principal do algoritmo -- #
    for i, rawItem in enumerate(goldenSet):

        currentID = rawItem.get('id') #id do item atual
        questionText = rawItem.get('question') #questao do item atual 

        #Pula se o id atual do goldenset ja foi processado pro geminiset
        if (currentID in processedIDs):
            print(f"ID {currentID} já foi processado")
            continue

        print(f"{currentID} Processando com gemini...")

        try:
            # -- 3. CHAMADA DA API GEMINI --
            QApair = utils.callApiGemini(currentID, questionText)

            # -- 4. Validação e Parsing -- #
            # carrega apenas o campo .text da resposta do gemini
            QApair_dict = json.loads(QApair.text)

            #garante que o id do item atual pro geminiset sera o idCurrent do goldenset
            QApair_dict['id'] = currentID

            #validação e parsing por meio do pydantic
            validatedEntry = dataFormat.QAPairModel(**QApair_dict)

            datasetGemini.data.append(validatedEntry) #add no datasetgemini
            processedIDs.add(currentID) #o id atual entra nos processados
            print(f"ID {currentID} Feito com sucesso")

        #verificações de erros
        except json.decoder.JSONDecodeError as e:
            print(f"ID {currentID}: Gemini retornou um JSON inválido: {e}")
        except ValidationError as e:
            print(f"ID {currentID}: Erro de validação Pydantic: {e}")
        except errors.APIError as e:
            print(f"ID {currentID}: Erro da API Gemini: {e}")
            time.sleep(60)
        except Exception as e:
            print(f"ID {currentID}: Erro desconhecido: {e}")


        # -- 5. Salvamento intermitente -- #
        # O salvamento no dataset Gemini ocorre a cada 5 pares (sucesso ou falha)
        if (i + 1) % 5 == 0:
            print(f"Salvando dados no {DATASET_GEMINI_PATH}")
            with open(DATASET_GEMINI_PATH, 'w', encoding='utf-8') as f:
                f.write(datasetGemini.model_dump_json(indent=2))
        
        time.sleep(TIME_BETWEEN_CALLS)

    #Ordenando todos os dados por meio do ID
    datasetGemini.data.sort(key=lambda x: x.id)

    #Ultimo salvamento de todos os dados prontos e ordenados
    with open(DATASET_GEMINI_PATH, 'w', encoding='utf-8') as f:
        f.write(datasetGemini.model_dump_json(indent=2))

    

