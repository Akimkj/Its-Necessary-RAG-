from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import json
import uuid

def chunks_to_embeddings(chunks: list[Document]) -> str:
    model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", trust_remote_code=True)
    
    result = []
    for chunk in chunks:
        text = chunk.page_content
        embedding = model.encode(text).tolist()
        
        result.append({
            "id": str(uuid.uuid4()),
            "chunk": text,
            "embeddings": embedding,
            "metadata": chunk.metadata
        })
    
    return json.dumps(result, ensure_ascii=False, indent=2)
