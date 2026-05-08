from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def insert_embeddings_to_mongodb(json_data: str | list):
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    inserted = 0
    errors = 0

    for item in data:
        try:
            collection.insert_one(item)
            inserted += 1
            doc_id = item.get('_id', item.get('id', 'unknown'))
            print(f"Inserido: {doc_id}")
        except Exception as e:
            errors += 1
            doc_id = item.get('_id', item.get('id', 'unknown'))
            print(f"Erro ao inserir {doc_id}: {e}")

    client.close()
    print(f"\nResultado: {inserted} inseridos, {errors} erros")

def semantic_search(query: str, limit: int = 5):
    from sentence_transformers import SentenceTransformer
    
    # Inicializando o modelo para gerar o embedding da busca
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(query).tolist()
    
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embbeding_index",
                "path": "embeddings",
                "queryVector": query_vector,
                "numCandidates": limit * 10,
                "limit": limit
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"Erro ao realizar a busca semântica: {e}")
        results = []
    finally:
        client.close()
        
    return results    