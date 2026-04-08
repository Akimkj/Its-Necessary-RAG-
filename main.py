import os
from dotenv import load_dotenv
load_dotenv()

from src.utils import loadData, bertCompareGraph
from src.generator import process_questions
from src.evaluation import bertEvaluation
from src.bertStatistics import generate_statistics
import pandas as pd

# Dataset paths
PATH_GOLDENSET = os.path.join("data", "raw", "stackoverflow_dataset.json")

PATH_GEMINISET  = os.path.join("data", "processed", "gemini_dataset.json")
PATH_GEMINIBERT = os.path.join("results", "csv", "avBert_gemini.csv")

PATH_CLAUDESET  = os.path.join("data", "processed", "claude_dataset.json")
PATH_CLAUDEBERT = os.path.join("results", "csv", "avBert_claude.csv")

PATH_DEEPSEEKSET  = os.path.join("data", "processed", "deepseek_dataset.json")
PATH_DEEPSEEKBERT = os.path.join("results", "csv", "avBert_deepseek.csv")

PATH_OPENAISET  = os.path.join("data", "processed", "openai_dataset.json")
PATH_OPENAIBERT = os.path.join("results", "csv", "avBert_openai.csv")

PATH_STATS_CSV = os.path.join("results", "stats", "bert_statistics.csv")

# Mapping used by the statistics option
MODELS_BERT = {
    "Gemini":   PATH_GEMINIBERT,
    "Claude":   PATH_CLAUDEBERT,
    "OpenAI":   PATH_OPENAIBERT,
    "DeepSeek": PATH_DEEPSEEKBERT,
}


print("MENU - RAG PAPER")
while True:
    print("\n1 - Gerar novo dataset")
    print("2 - Compare dataset com Goldenset")
    print("3 - Criar grafico comparativo")
    print("4 - Calcular estatisticas BERT")
    print("5 - Sair")
    op = int(input("Option: "))

    if op == 1:
        print("Qual modelo deseja usar ?")
        print("1 - Gemini")
        print("2 - Claude")
        print("3 - OpenAi")
        print("4 - DeepSeek")
        mod_op = int(input("Option: "))
        rawGolden = loadData(PATH_GOLDENSET)
        if mod_op == 1:
            process_questions(rawGolden, PATH_GEMINISET, "gemini")
        elif mod_op == 2:
            process_questions(rawGolden, PATH_CLAUDESET, "claude")
        elif mod_op == 3:
            process_questions(rawGolden, PATH_OPENAISET, "openai")
        elif mod_op == 4:
            process_questions(rawGolden, PATH_DEEPSEEKSET, "deepseek")

    elif op == 2:
        print("Qual dataset deseja avaliar")
        print("1 - Gemini")
        print("2 - Claude")
        print("3 - OpenAi")
        print("4 - DeepSeek")
        mod_op = int(input("Option: "))
        rawGolden = loadData(PATH_GOLDENSET)
        if mod_op == 1:
            rawDataset = loadData(PATH_GEMINISET)
            bertEvaluation(rawDataset, rawGolden, PATH_GEMINIBERT)
        elif mod_op == 2:
            rawDataset = loadData(PATH_CLAUDESET)
            bertEvaluation(rawDataset, rawGolden, PATH_CLAUDEBERT)
        elif mod_op == 3:
            rawDataset = loadData(PATH_OPENAISET)
            bertEvaluation(rawDataset, rawGolden, PATH_OPENAIBERT)
        elif mod_op == 4:
            rawDataset = loadData(PATH_DEEPSEEKSET)
            bertEvaluation(rawDataset, rawGolden, PATH_DEEPSEEKBERT)

    elif op == 3:
        '''df1 = None
        df2 = None
        
        if os.path.exists(PATH_GEMINIBERT1):
            df1 = pd.read_csv(PATH_GEMINIBERT1)
        else:
            print(f"Warning: Version 1 dataset not found ({PATH_GEMINIBERT1})")
            continue
            
        if os.path.exists(PATH_GEMINIBERT2):
            df2 = pd.read_csv(PATH_GEMINIBERT2)
        else:
            print(f"Warning: Version 2 dataset not found ({PATH_GEMINIBERT2})")
            continue

        bertCompareGraph(df1, df2)'''

    elif op == 4:
        generate_statistics(MODELS_BERT, PATH_STATS_CSV)

    elif op == 5:
        break