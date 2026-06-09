import json
import uuid
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

_MODEL = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("\nCarregando o modelo Qwen3-Embedding-0.6B...")
        # Carrega o modelo Qwen3-Embedding-0.6B
        _MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
        print("Modelo Qwen3-Embedding-0.6B carregado com sucesso!")
    return _MODEL

def chunks_to_embeddings(chunks: list[Document]) -> str:
    if not chunks:
        return json.dumps([])

    total = len(chunks)
    print(f"Iniciando processamento de {total} chunks...")

    # Extrai todos os textos e formata com o prompt de inserção para documentação Python
    texts = [
        f"Instruct: Represent the Python code documentation passage for retrieval\nQuery: {chunk.page_content}"
        for chunk in chunks
    ]
    
    # Obtém o modelo carregado sob demanda
    model = _get_model()
    
    print("Gerando embeddings em lote...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Montagem do resultado final 
    result = []
    for chunk, embedding in zip(chunks, embeddings):
        result.append({
            "id": str(uuid.uuid4()),
            "chunk": chunk.page_content,
            "embeddings": embedding.tolist(), 
            "metadata": chunk.metadata
        })
    
    print(f"[{total}/{total}] Todos os embeddings foram processados com sucesso!")
    return json.dumps(result, ensure_ascii=False, indent=2)
