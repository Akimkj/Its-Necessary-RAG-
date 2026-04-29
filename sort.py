import os
from src.evaluators.modernBert_eva import modernBertEvaluation
from src.charts import run_charts
from src.utils import loadData
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
import os
from bert_score import BERTScorer
from typing import cast

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


def _get_embeddings(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
    inputs.pop("token_type_ids", None)  # ModernBERT não usa token_type_ids

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state.squeeze(0)

    # Remove tokens especiais (BOS/EOS/PAD) dos embeddings
    special_ids = set(tokenizer.all_special_ids)
    keep = [i for i, tid in enumerate(inputs["input_ids"].squeeze(0).tolist()) if tid not in special_ids]

    return hidden[keep] if keep else hidden

def _bertscore_prf(candidate, reference, tokenizer, model, device):
    # Extrai e normaliza os embeddings de cada texto
    emb_cand = F.normalize(_get_embeddings(candidate, tokenizer, model, device), p=2, dim=-1)
    emb_ref  = F.normalize(_get_embeddings(reference,  tokenizer, model, device), p=2, dim=-1)

    # Matriz de similaridade de cosseno e greedy matching
    sim_matrix = torch.mm(emb_cand, emb_ref.T)
    precision  = sim_matrix.max(dim=1).values.mean().item()
    recall     = sim_matrix.max(dim=0).values.mean().item()
    f1         = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


'''
modernBertEvaluation(loadData(PATH_GEMINISET), loadData(PATH_GOLDENSET), PATH_GEMINIBERT_DEBUG)
modernBertEvaluation(loadData(PATH_CLAUDESET), loadData(PATH_GOLDENSET), PATH_CLAUDEBERT_DEBUG)
modernBertEvaluation(loadData(PATH_OPENAISET), loadData(PATH_GOLDENSET), PATH_OPENAIBERT_DEBUG)
modernBertEvaluation(loadData(PATH_DEEPSEEKSET), loadData(PATH_GOLDENSET), PATH_DEEPSEEKBERT_DEBUG)
'''

'''PATH_DIR_DEBUG = os.path.join("results", "csv", "debug")
run_charts(PATH_STATS_CSV, PATH_DIR_DEBUG)'''

'''with open(PATH_GOLDENSET, 'r', encoding='utf-8') as file:
    rawData = json.load(file)

rawData['data'].sort(key=lambda x: len(x["answer"]))

with open(PATH_GOLDENSET, 'w', encoding='utf-8') as file:
    json.dump(rawData,file, indent=4, ensure_ascii=False)'''

MODEL_ID = "answerdotai/ModernBERT-large"
device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
model.to(device)
model.eval()

scorer = BERTScorer(
        model_type='roberta-large',
        lang='en',
        rescale_with_baseline=False,
        device=None,
    )

#P, R, F1 = scorer.score(["In Python, use if not my_list:. In JavaScript, check my_list.length === 0. Generally, verifying if the size or length is zero is the safest way."],["if not a: print(\"List is empty\") Using the implicit booleanness of the empty list is quite Pythonic."])

P, R, F1 = _bertscore_prf("In Python, use if not my_list:. In JavaScript, check my_list.length === 0. Generally, verifying if the size or length is zero is the safest way.", "if not a: print(\"List is empty\") Using the implicit booleanness of the empty list is quite Pythonic.", tokenizer, model, device)

print(f"F1 teste: {F1}")

'''round(cast(torch.Tensor, F1).item(), 4)'''
