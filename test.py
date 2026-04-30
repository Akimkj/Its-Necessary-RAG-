import os, torch, json
from src.evaluators.modernBert_eva import modernBertEvaluation
from src.charts import run_charts
from src.utils import loadData
from src.services.openai_service import callApiOpenai
from src.services.deepseek_service import callApiDeepseek
from src.services.claude_service import callApiClaude
from src.services.gemini_service import callApiGemini
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from src.count_tokens import count_tokens
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


"""
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
"""
tokens_answer_stackOverflow = count_tokens("If the reason you're checking is so you can do something like if file_exists: open_it(), it's safer to use a try around the attempt to open it. Checking and then opening risks the file being deleted or moved or something between when you check and when you try to open it. If you're not planning to open the file immediately, you can use os.path.isfile if you need to be sure it's a file. Return True if path is an existing regular file. This follows symbolic links, so both islink() and isfile() can be true for the same path. import os.path os.path.isfile(fname) pathlib Starting with Python 3.4, the pathlib module offers an object-oriented approach (backported to pathlib2 in Python 2.7): from pathlib import Path my_file = Path(\"/path/to/file\") if my_file.is_file(): # file exists To check a directory, do: if my_file.is_dir(): # directory exists To check whether a Path object exists independently of whether is it a file or directory, use exists(): if my_file.exists(): # path exists You can also use resolve(strict=True) in a try block: try: my_abs_path = my_file.resolve(strict=True) except FileNotFoundError: # doesn't exist else: # exists", "openai")
QApair = callApiGemini(1, "How do I check whether a file exists without exceptions?", tokens_answer_stackOverflow)

clean_json = QApair.strip().replace("```json", "").replace("```", "")

QApair_dict = json.loads(clean_json)

resposta = QApair_dict['answer']

print(f"Resposta: {resposta}")
print(f"tokens referencia: {tokens_answer_stackOverflow}")
print(f"tokens candidato: {count_tokens(resposta, "gemini")}")

