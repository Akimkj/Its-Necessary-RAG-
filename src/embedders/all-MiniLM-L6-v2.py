from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import json
import uuid

def chunks_to_embeddings(chunks: list[Document]) -> str:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    result = []
    total = len(chunks)
    print(f"Iniciando processamento de {total} chunks...")
    for i, chunk in enumerate(chunks, 1):
        text = chunk.page_content
        embedding = model.encode(text).tolist()

        print(f"[{i}/{total}] Embeddings processados...")

        result.append({
            "id": str(uuid.uuid4()),
            "chunk": text,
            "embeddings": embedding,
            "metadata": chunk.metadata
        })
    
    return json.dumps(result, ensure_ascii=False, indent=2)