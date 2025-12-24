from src.geminiSetGenerator import process_questions
import os, json

GOLDENSET_PATH = os.path.join("data", "stackoverflow_dataset.json") 

def loadData(filePath: str):
    
    try:
        with open(filePath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            print("[MAIN] - dataset carregado")
            return dataset
    except FileNotFoundError:
        print(f"Arquivo {filePath} não encontrado")
        return []
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON do arquivo {filePath}")
        return []

goldenSet = loadData(GOLDENSET_PATH)

if (goldenSet):
    process_questions(goldenSet)