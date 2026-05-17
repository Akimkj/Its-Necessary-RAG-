import os
from dotenv import load_dotenv
load_dotenv()

from src.utils import loadData
from src.generator import process_questions
from src.evaluators.modernBert_eva import modernBertEvaluation
from src.evaluators.robertaLarge_eva import robertaLargeEvaluation
from src.bertStatistics import generate_statistics
from src.charts import run_charts
from src.ingestionPipeline import run_ingestion_pipeline

# ── Caminhos dos datasets ──
PATH_GOLDENSET = os.path.join("data", "raw", "popular_StackOverflow.json")


PATH_GEMINISET  = os.path.join("data", "processed", "gemini_dataset.json")
PATH_GEMINISET_LIMITED = os.path.join("data", "processed", "limited", "gemini_dataset_limited.json")
PATH_GEMINIBERT_BASE = os.path.join("results", "csv", "base", "avBert_gemini.csv")
PATH_GEMINIBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_gemini_limited.csv")


PATH_CLAUDESET  = os.path.join("data", "processed", "claude_dataset.json")
PATH_CLAUDESET_LIMITED = os.path.join("data", "processed", "limited", "claude_dataset_limited.json")
PATH_CLAUDEBERT_BASE = os.path.join("results", "csv", "base", "avBert_claude.csv")
PATH_CLAUDEBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_claude_limited.csv")


PATH_DEEPSEEKSET  = os.path.join("data", "processed", "deepseek_dataset.json")
PATH_DEEPSEEKSET_LIMITED = os.path.join("data", "processed", "limited", "deepseek_dataset_limited.json")
PATH_DEEPSEEKBERT_BASE = os.path.join("results", "csv", "base", "avBert_deepseek.csv")
PATH_DEEPSEEKBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_deepseek_limited.csv")


PATH_OPENAISET  = os.path.join("data", "processed", "openai_dataset.json")
PATH_OPENAISET_LIMITED =  os.path.join("data", "processed", "limited", "openai_dataset_limited.json")
PATH_OPENAIBERT_BASE = os.path.join("results", "csv", "base", "avBert_openai.csv")
PATH_OPENAIBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_openai_limited.csv")


PATH_STATS_BASE_CSV = os.path.join("results", "stats", "base", "bert_base_statistics.csv")
PATH_STATS_LIMITED_CSV = os.path.join("results", "stats", "limited", "bert_limited_statistics.csv")

PATH_BERTSCORE_BASE_CSV = os.path.join("results", "csv", "base")
PATH_BERTSCORE_LIMITED_CSV = os.path.join("results", "csv", "limited")



# Mapeamento modelo → caminho do CSV BERT (usado pela opção 4)
MODELS_BERT_BASE = {
    "Gemini 2.5 Flash":   PATH_GEMINIBERT_BASE,
    "Claude Sonnet 4.6":   PATH_CLAUDEBERT_BASE,
    "GPT-4.1 mini":   PATH_OPENAIBERT_BASE,
    "DeepSeek-V3": PATH_DEEPSEEKBERT_BASE,
}

MODELS_BERT_LIMITED = {
    "Gemini 2.5 Flash":   PATH_GEMINIBERT_LIMITED,
    "Claude Sonnet 4.6":   PATH_CLAUDEBERT_LIMITED,
    "GPT-4.1 mini":   PATH_OPENAIBERT_LIMITED,
    "DeepSeek-V3": PATH_DEEPSEEKBERT_LIMITED,
}


print("MENU - RAG PAPER")
while True:
    print("\n1 - Gerar novo dataset")
    print("2 - Comparar dataset com Goldenset (BERT)")
    print("3 - Gerar graficos comparativos")
    print("4 - Calcular estatisticas BERT")
    print("5 - Carregar documento no banco de dados (chunking + embedding + MongoDB)")
    print("6 - Sair")
    op = int(input("Option: "))

    # ── Opção 1: gera respostas do modelo escolhido ──
    if op == 1:
        print("Qual modelo deseja usar?")
        print("1 - Gemini")
        print("2 - Claude")
        print("3 - OpenAI")
        print("4 - DeepSeek")
        mod_op = int(input("Option: "))
        rawGolden = loadData(PATH_GOLDENSET)
        if mod_op == 1:
            process_questions(rawGolden, PATH_GEMINISET_LIMITED, "gemini")
        elif mod_op == 2:
            process_questions(rawGolden, PATH_CLAUDESET_LIMITED, "claude")
        elif mod_op == 3:
            process_questions(rawGolden, PATH_OPENAISET_LIMITED, "openai")
        elif mod_op == 4:
            process_questions(rawGolden, PATH_DEEPSEEKSET_LIMITED, "deepseek")

    # ── Opção 2: avalia um dataset com BERT Score ──
    elif op == 2:
        print("Qual modelo deseja usar?")
        print("1 - ModernBERT")
        print("2 - RobertaLarge")
        mod_op = int(input("Option: "))
        if mod_op == 1:
            print("Qual dataset deseja avaliar?")
            print("1 - Gemini")
            print("2 - Claude")
            print("3 - OpenAI")
            print("4 - DeepSeek")
            mod_op = int(input("Option: "))
            rawGolden = loadData(PATH_GOLDENSET)
            if mod_op == 1:
                rawDataset = loadData(PATH_GEMINISET)
                modernBertEvaluation(rawDataset, rawGolden, PATH_GEMINIBERT_BASE)
            elif mod_op == 2:
                rawDataset = loadData(PATH_CLAUDESET)
                modernBertEvaluation(rawDataset, rawGolden, PATH_CLAUDEBERT_BASE)
            elif mod_op == 3:
                rawDataset = loadData(PATH_OPENAISET)
                modernBertEvaluation(rawDataset, rawGolden, PATH_OPENAIBERT_BASE)
            elif mod_op == 4:
                rawDataset = loadData(PATH_DEEPSEEKSET)
                modernBertEvaluation(rawDataset, rawGolden, PATH_DEEPSEEKBERT_BASE)
        elif mod_op == 2:
            print("Qual dataset deseja avaliar?")
            print("1 - Gemini")
            print("2 - Claude")
            print("3 - OpenAI")
            print("4 - DeepSeek")
            mod_op = int(input("Option: "))
            rawGolden = loadData(PATH_GOLDENSET)
            if mod_op == 1:
                rawDataset = loadData(PATH_GEMINISET)
                robertaLargeEvaluation(rawDataset, rawGolden, PATH_GEMINIBERT_BASE)
            elif mod_op == 2:
                rawDataset = loadData(PATH_CLAUDESET)
                robertaLargeEvaluation(rawDataset, rawGolden, PATH_CLAUDEBERT_BASE)
            elif mod_op == 3:
                rawDataset = loadData(PATH_OPENAISET)
                robertaLargeEvaluation(rawDataset, rawGolden, PATH_OPENAIBERT_BASE)
            elif mod_op == 4:
                rawDataset = loadData(PATH_DEEPSEEKSET)
                robertaLargeEvaluation(rawDataset, rawGolden, PATH_DEEPSEEKBERT_BASE)

    # ── Opção 3: gera gráficos a partir das estatísticas BERT ──
    elif op == 3:
        run_charts(PATH_STATS_LIMITED_CSV, PATH_BERTSCORE_LIMITED_CSV)

    # ── Opção 4: calcula e salva as estatísticas BERT ──
    elif op == 4:
        generate_statistics(MODELS_BERT_LIMITED, PATH_STATS_LIMITED_CSV)

    # ── Opção 5: pipeline de ingestião de documentos no MongoDB ──
    elif op == 5:
        run_ingestion_pipeline()

    elif op == 6:
        break
