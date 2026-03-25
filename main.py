import os
from dotenv import load_dotenv
load_dotenv()

from src.utils import loadData, bertCompareGraph
from src.generator import process_questions
import os
from src.evaluation import bertEvaluation
import pandas as pd

#carrega os caminhos dos datasets // alterar posteriormente
PATH_GOLDENSET = os.path.join("data", "raw", "stackoverflow_dataset.json") 
PATH_GEMINISET = os.path.join("data", "processed", "gemini_dataset_v2.json")
PATH_GEMINIBERT1 = os.path.join("results", "csv", "avBert_gemini_v1.csv")
PATH_GEMINIBERT2 = os.path.join("results", "csv", "avBert_gemini_v2.csv")
PATH_CLAUDESET = os.path.join("data", "processed", "claude_dataset.json")
PATH_CLAUDEBERT = os.path.join("results", "csv", "avBert_claude.csv")



print("MENU - ARTIGO RAG")
while(True):
    print("1 - Gerar novo dataset")
    print("2 - Compara dataset com Goldenset")
    print("3 - Criar gráficos de comparação")
    op = int(input("Opção: "))

    if (op == 1):
        print("Qual modelo deseja usar?")
        print("1 - Gemini")
        print("2 - Claude")
        mod_op = int(input("Opção: "))
        rawGolden = loadData(PATH_GOLDENSET)
        
        if mod_op == 2:
            process_questions(rawGolden, PATH_CLAUDESET, "claude")
        else:
            process_questions(rawGolden, PATH_GEMINISET, "gemini")
            
    elif (op == 2):
        print("Qual dataset deseja avaliar?")
        print("1 - Gemini")
        print("2 - Claude")
        mod_op = int(input("Opção: "))
        
        rawGolden = loadData(PATH_GOLDENSET)
        if mod_op == 2:
            rawDataset = loadData(PATH_CLAUDESET)
            bertEvaluation(rawDataset, rawGolden, PATH_CLAUDEBERT)
        else:
            rawDataset = loadData(PATH_GEMINISET)
            bertEvaluation(rawDataset, rawGolden, PATH_GEMINIBERT2)
    elif (op == 3):
        import os
        df1 = None
        df2 = None
        
        if os.path.exists(PATH_GEMINIBERT1):
            df1 = pd.read_csv(PATH_GEMINIBERT1)
        else:
            print(f"Aviso: Dataset Versão 1 não encontrado ({PATH_GEMINIBERT1})")
            
        if os.path.exists(PATH_GEMINIBERT2):
            df2 = pd.read_csv(PATH_GEMINIBERT2)
        else:
            print(f"Aviso: Dataset Versão 2 não encontrado ({PATH_GEMINIBERT2})")

        bertCompareGraph(df1, df2)



