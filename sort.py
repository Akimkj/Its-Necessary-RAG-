import os
from src.evaluators.modernBert_eva import modernBertEvaluation
from src.utils import loadData

# Caminhos para depuração do BertScore
PATH_GEMINIBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_gemini_debug.csv")
PATH_CLAUDEBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_claude_debug.csv")
PATH_DEEPSEEKBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_deepseek_debug.csv")
PATH_OPENAIBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_openai_debug.csv")
PATH_STATS_CSV = os.path.join("results", "stats", "bert_statistics.csv")
# ── Caminhos dos datasets ──
PATH_GOLDENSET = os.path.join("data", "raw", "stackoverflow_dataset.json")
PATH_GEMINISET  = os.path.join("data", "processed", "gemini_dataset.json")
PATH_CLAUDESET  = os.path.join("data", "processed", "claude_dataset.json")
PATH_DEEPSEEKSET  = os.path.join("data", "processed", "deepseek_dataset.json")
PATH_OPENAISET  = os.path.join("data", "processed", "openai_dataset.json")

modernBertEvaluation(loadData(PATH_GEMINISET), loadData(PATH_GOLDENSET), PATH_GEMINIBERT_DEBUG)
modernBertEvaluation(loadData(PATH_CLAUDESET), loadData(PATH_GOLDENSET), PATH_CLAUDEBERT_DEBUG)
modernBertEvaluation(loadData(PATH_OPENAISET), loadData(PATH_GOLDENSET), PATH_OPENAIBERT_DEBUG)
modernBertEvaluation(loadData(PATH_DEEPSEEKSET), loadData(PATH_GOLDENSET), PATH_DEEPSEEKBERT_DEBUG)



'''with open(PATH_GOLDENSET, 'r', encoding='utf-8') as file:
    rawData = json.load(file)

rawData['data'].sort(key=lambda x: len(x["answer"]))

with open(PATH_GOLDENSET, 'w', encoding='utf-8') as file:
    json.dump(rawData,file, indent=4, ensure_ascii=False)'''


