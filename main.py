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

PATH_GEMINISET = os.path.join("data", "processed", "gemini_dataset.json")
PATH_GEMINIBERT = os.path.join("results", "csv", "avBert_gemini.csv")

PATH_CLAUDESET = os.path.join("data", "processed", "claude_dataset.json")
PATH_CLAUDEBERT = os.path.join("results", "csv", "avBert_claude.csv")

PATH_OPENAISET = os.path.join("data", "processed", "openai_dataset.json")
PATH_OPENAIBERT = os.path.join("results", "csv", "avBert_openai.csv")





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
        print("3 - OpenAi")
        mod_op = int(input("Opção: "))
        rawGolden = loadData(PATH_GOLDENSET)
        if mod_op == 1:
            process_questions(rawGolden, PATH_GEMINISET, "gemini")
        elif mod_op == 2:
            process_questions(rawGolden, PATH_CLAUDESET, "claude")
        elif mod_op == 3:
            process_questions(rawGolden, PATH_OPENAI, "openai")
            
    elif (op == 2):
        print("Qual dataset deseja avaliar?")
        print("1 - Gemini")
        print("2 - Claude")
        print("3 - OpenAi")
        mod_op = int(input("Opção: "))
        rawGolden = loadData(PATH_GOLDENSET)
        rawDataset = None

        if mod_op == 1:
            rawDataSet = loadData(PATH_GEMINISET)
        elif mod_op == 2:
            rawDataSet = loadData(PATH_CLAUDESET)
        elif mod_op == 3:
            rawDataSet = loadData(PATH_OPENAISET)
            
        bertEvaluation(rawDataset, rawGolden)
    elif (op == 3):
        df1 = None
        df2 = None
        
        if os.path.exists(PATH_GEMINIBERT1):
            df1 = pd.read_csv(PATH_GEMINIBERT1)
        else:
            print(f"Aviso: Dataset Versão 1 não encontrado ({PATH_GEMINIBERT1})")
            continue
            
        if os.path.exists(PATH_GEMINIBERT2):
            df2 = pd.read_csv(PATH_GEMINIBERT2)
        else:
            print(f"Aviso: Dataset Versão 2 não encontrado ({PATH_GEMINIBERT2})")
            continue

        bertCompareGraph(df1, df2)



