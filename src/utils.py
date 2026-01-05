import json, re

def clean_text(text: str):
    text_cleaned = re.sub(r'\s+', ' ', text).strip()
    return text_cleaned


def loadData(filePath: str, default_type=dict):
    try:
        with open(filePath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            print("[MAIN] - dataset carregado")
            return dataset
    except FileNotFoundError:
        print(f"Arquivo {filePath} não encontrado")
        return default_type()
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON do arquivo {filePath}")
        return default_type()
