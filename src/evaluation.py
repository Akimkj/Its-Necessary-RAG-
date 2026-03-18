from __future__ import annotations
from bert_score import BERTScorer
import pandas as pd
import os
from . import dataFormat

def bertEvaluation(rawCandidate: list, rawGolden: list) -> Dataframe:

    #Padroniza a estrutura de ambos os datasets com pydantic
    goldenset = dataFormat.QADataSet(data=rawGolden)
    candidateset = dataFormat.QADataSet(data=rawCandidate)

    #Carrega apenas uma vez as informações do modelo pré-treinado
    scorer = BERTScorer(model_type='roberta-large',
                                lang='en',
                                rescale_with_baseline=False,
                                device='cpu')


    results = []

    #Loop principal que percorre a lista goldenset
    for gold in goldenset.data:

        #Procura o item do geminiset que tenha o mesmo id do item do goldenset
        candidateItem = next((item for item in candidateset.data if item.id == gold.id), None)

        if candidateItem:
            reference = gold.answer 
            candidate = candidateItem.answer
            
            #realiza o calculo com a referencia e o candidato
            P, R, F1 = scorer.score([reference], [candidate])

            #adiciona as informações do id e pergunta junto com seus resultados
            results.append({
                "id": candidateItem.id,
                "question": candidateItem.question,
                "bert_precision": round(P.item(), 4),
                "bert_recall": round(R.item(), 4),
                "bert_F1": round(F1.item(), 4)
            })
        else:
            print("Não foi achado o ID correspondente...")

    df = pd.DataFrame(results)
    #Mudar o nome do arquivo para outras comparações
    outputPath = os.path.join("results", "csv", "avBert_gemini.csv")
    df.to_csv(outputPath, index=False, encoding='utf-8')


    return df




        











