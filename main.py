#from src.geminiSetGenerator import process_questions
from src.utils import loadData, bertCompareGraph
import os
from src.evaluation import bertEvaluation
import pandas as pd

#carrega os caminhos dos datasets
PATH_GOLDENSET = os.path.join("data", "stackoverflow_dataset.json") 
PATH_GEMINISET1 = os.path.join("data", "gemini_dataset_v1.json")
PATH_GEMINISET2 = os.path.join("data", "gemini_dataset_v2.json")
PATH_GEMINIBERT1 = os.path.join("results", "avBert_gemini_v1.csv")
PATH_GEMINIBERT2 = os.path.join("results", "avBert_gemini_v2.csv")

'''
#Carrega os dados dos datasets da forma que estão disponíveis
rawGolden = loadData(PATH_GOLDENSET, default_type=list)
rawGemini1 = loadData(PATH_GEMINISET1)
rawGemini2 = loadData(PATH_GEMINISET2)

bertEvaluation(rawGemini1, rawGolden)
bertEvaluation(rawGemini2, rawGolden)
'''

df1 = pd.read_csv(PATH_GEMINIBERT1)
df2 = pd.read_csv(PATH_GEMINIBERT2)

bertCompareGraph(df1, df2)



