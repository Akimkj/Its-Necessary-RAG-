from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import json

def chunks_to_embeddings(chunks: list[Document]) -> str:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    result = []
    for chunk in chunks:
        text = chunk.page_content
        embedding = model.encode(text).tolist()
        
        result.append({
            "chunk": text,
            "embeddings": embedding
        })
    
    return json.dumps(result, ensure_ascii=False, indent=2)