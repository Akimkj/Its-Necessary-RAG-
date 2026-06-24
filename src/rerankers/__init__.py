import os
import importlib.util

# Para evitar problemas com hífens no nome do arquivo ao importar em Python,
# importamos dinamicamente a função rerank_chunks.
_current_dir = os.path.dirname(__file__)
_file_path = os.path.join(_current_dir, "Qwen3-Reranker-0.6B.py")

if os.path.exists(_file_path):
    _spec = importlib.util.spec_from_file_location("qwen3_reranker_module", _file_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    rerank_chunks = _module.rerank_chunks
else:
    raise FileNotFoundError(f"Não foi possível encontrar o arquivo {_file_path}")
