import matplotlib.pyplot as plt
import json, re, os
import pandas as pd

def clean_text(text: str):
    text_cleaned = re.sub(r'\s+', ' ', text).strip()
    return text_cleaned


def loadData(filePath: str, default_type=list):
    try:
        with open(filePath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
            if (isinstance(content, dict)):
                return content.get("data", [])
            else:
                return content
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erro ao carregar dados de {filePath}: {e}")
        return default_type()


def bertCompareGraph(df1=None, df2=None):
    pathResult = os.path.join("results", "plots", "avBert_prompt_compare.png")
    os.makedirs(os.path.dirname(pathResult), exist_ok=True)

    if df1 is not None and df2 is not None:
        cleanV1 = df1[['id', 'bert_F1']].rename(columns={'bert_F1': 'F1_v1'})
        cleanV2 = df2[['id', 'bert_F1']].rename(columns={'bert_F1': 'F1_v2'})
        df = pd.merge(cleanV1, cleanV2, on='id')
        df.set_index('id').plot.bar(figsize=(14, 7))
    elif df2 is not None:
        cleanV2 = df2[['id', 'bert_F1']].rename(columns={'bert_F1': 'F1_v2'})
        cleanV2.set_index('id').plot.bar(figsize=(14, 7), color='orange')
    elif df1 is not None:
        cleanV1 = df1[['id', 'bert_F1']].rename(columns={'bert_F1': 'F1_v1'})
        cleanV1.set_index('id').plot.bar(figsize=(14, 7), color='blue')
    else:
        print("Nenhum dado para plotar.")
        return

    plt.ylim(0.000, 1.0)
    plt.tight_layout()

    plt.savefig(pathResult, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {pathResult}")
    plt.show()



