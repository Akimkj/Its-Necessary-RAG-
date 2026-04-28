from pymongo import MongoClient
import json
from langchain_core.documents import Document

MONGO_URI = "mongodb+srv://didizst:Tabita1989.@cluster-rag.hworbxl.mongodb.net/?appName=Cluster-RAG"
DB_NAME = "RAG"
COLLECTION_NAME = "RAG"

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
            print(f"✅ Inserido: {doc_id}")
        except Exception as e:
            errors += 1
            doc_id = item.get('_id', item.get('id', 'unknown'))
            print(f"❌ Erro ao inserir {doc_id}: {e}")

    client.close()
    print(f"\n📊 Resultado: {inserted} inseridos, {errors} erros")