import json
import uuid
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Carrega o modelo Qwen3-VL-Embedding-2B
print("Carregando o modelo Qwen3-VL-Embedding-2B...")
MODEL = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", trust_remote_code=True)
print("Modelo carregado com sucesso!")

def chunks_to_embeddings(chunks: list[Document]) -> str:
    if not chunks:
        return json.dumps([])

    total = len(chunks)
    print(f"Iniciando processamento de {total} chunks...")

    # Extrai todos os textos para processar em lotes
    texts = [chunk.page_content for chunk in chunks]
    
    # model.encode processa a lista inteira
    print("Gerando embeddings em lote...")
    embeddings = MODEL.encode(texts, show_progress_bar=True)

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