import matplotlib.pyplot as plt
import json, re, os
import pandas as pd

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


def bertCompareGraph(df1, df2):
    pathResult = os.path.join("results", "avBert_prompt_compare.png")

    cleanV1 = df1[['id', 'bert_F1']].rename(columns={'bert_F1': 'F1_v1'})
    cleanV2 = df2[['id', 'bert_F1']].rename(columns={'bert_F1': 'F1_v2'})


    df = pd.merge(cleanV1, cleanV2, on='id')
    df.set_index('id').plot.bar(figsize=(14, 7))
    plt.ylim(0.000, 1.0)
    plt.tight_layout()

    plt.savefig(pathResult, dpi=300, bbox_inches='tight')
    plt.show()