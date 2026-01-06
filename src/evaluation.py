from bert_score import BERTScorer
import pandas as pd
import os
from . import dataFormat

def bertEvaluation(rawGemini: dict, rawGolden: dict):

    #Padroniza a estrutura de ambos os datasets com pydantic
    goldenset = dataFormat.QADataSet(data=rawGolden)
    geminiset = dataFormat.QADataSet(**rawGemini)

    #Carrega apenas uma vez as informações do modelo pré-treinado
    scorer = BERTScorer(model_type='roberta-large',
                                lang='en',
                                rescale_with_baseline=False,
                                device='cpu')


    results = []

    #Loop principal que percorre a lista (campo data) do goldenset
    for gold in goldenset.data:

        #Procura o item do geminiset que tenha o mesmo id do item do goldenset
        geminiItem = next((item for item in geminiset.data if item.id == gold.id), None)

        #Se encontrar, é realizada a operação com o BertScore
        if geminiItem:
            reference = gold.answer 
            candidate = geminiItem.answer
            
            #realiza o calculo com a referencia e o candidato
            P, R, F1 = scorer.score([reference], [candidate])

            #adiciona as informações do id e pergunta junto com seus resultados
            results.append({
                "id": geminiItem.id,
                "question": geminiItem.question,
                "bert_precision": round(P.item(), 4),
                "bert_recall": round(R.item(), 4),
                "bert_F1": round(F1.item(), 4)
            })
        else:
            print("Não foi achado o ID correspondente...")

    #Carrega a lista de resultados em DataFrame para transformar em .csv
    df = pd.DataFrame(results)
    outputPath = os.path.join("results", "avBert_gemini_v2.csv")
    df.to_csv(outputPath, index=False, encoding='utf-8')

    return df




        











