from src.utils import loadData, bertCompareGraph
from src.generator import process_questions
import os
from src.evaluation import bertEvaluation
import pandas as pd

#carrega os caminhos dos datasets // alterar posteriormente
PATH_GOLDENSET = os.path.join("data", "raw", "stackoverflow_dataset.json") 
PATH_DATASET = os.path.join("data", "processed", "gemini_dataset.json")
PATH_DATASETBERT1 = os.path.join("results", "csv", "avBert_gemini_v1.csv")
PATH_DATASETBERT2 = os.path.join("results", "csv", "avBert_gemini_v2.csv")



print("MENU - ARTIGO RAG")
while(True):
    print("1 - Gerar novo dataset")
    print("2 - Compara dataset com Goldenset")
    print("3 - Criar gráficos de comparação")
    op = int(input("Opção: "))

    if (op == 1):
        rawGolden = loadData(PATH_GOLDENSET)
        process_questions(rawGolden, PATH_DATASET)
    elif (op == 2):
        rawGolden = loadData(PATH_GOLDENSET)
        rawDataset = loadData(PATH_DATASET)
        bertEvaluation(rawDataset, rawGolden)
    elif (op == 3):
        df1 = pd.read_csv(PATH_DATASETBERT1)
        df2 = pd.read_csv(PATH_DATASETBERT2)

        bertCompareGraph(df1, df2)



