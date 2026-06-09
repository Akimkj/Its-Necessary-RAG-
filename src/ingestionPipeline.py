"""
ingestion_pipeline.py
Pipeline: documento -> chunks -> embeddings -> MongoDB
"""

import os
import importlib.util

from src.chunking.textSplitter import split_documents
from src.dataBaseLoader import insert_embeddings_to_mongodb

EMBEDDER_MAP = {
    1: {
        "name": "all-MiniLM-L6-v2",
        "file": os.path.join("src", "embedders", "all-MiniLM-L6-v2.py"),
        "function": "chunks_to_embeddings",
    },
    2: {
        "name": "Qwen3-Embedding-0.6B",
        "file": os.path.join("src", "embedders", "Qwen3-Embedding-0.6B.py"),
        "function": "chunks_to_embeddings",
    },
}

def _import_embedder(option_key: int):
    info = EMBEDDER_MAP[option_key]
    spec = importlib.util.spec_from_file_location("embedder_module", info["file"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, info["function"])
    return fn, info["name"]

def _load_document(path: str):
    if os.path.isdir(path):
        from langchain_community.document_loaders import ReadTheDocsLoader
        loader = ReadTheDocsLoader(path, encoding="utf-8")
        print("   -> Usando ReadTheDocsLoader (pasta HTML).")
    elif path.lower().endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        print("   -> Usando PyPDFLoader (.pdf).")
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(path, encoding="utf-8")
        print("   -> Usando TextLoader.")
    return loader.load()

def run_ingestion_pipeline():
    print("\n[INGESTION] Informe o caminho do documento ou pasta:")
    doc_path = input("Caminho: ").strip().strip('"').strip("'")

    if not os.path.exists(doc_path):
        print(f"Caminho nao encontrado: {doc_path}")
        return

    print("\n[INGESTION] Escolha o embedder:")
    for key, info in EMBEDDER_MAP.items():
        print(f"  {key} - {info['name']}")
    emb_op = int(input("Option: "))

    if emb_op not in EMBEDDER_MAP:
        print("Opcao de embedder invalida.")
        return

    embed_fn, emb_name = _import_embedder(emb_op)
    print(f"\nEmbedder: {emb_name}")

    print("\nCarregando documento(s)...")
    raw_docs = _load_document(doc_path)
    print(f"   -> {len(raw_docs)} documento(s) carregado(s).")

    print("\nDividindo em chunks...")
    chunks = split_documents(raw_docs)
    print(f"   -> {len(chunks)} chunk(s) gerado(s).")

    print(f"\nGerando embeddings com '{emb_name}'...")
    embeddings_json = embed_fn(chunks)
    print("   -> Embeddings gerados.")

    print("\nEnviando para o MongoDB...")
    insert_embeddings_to_mongodb(embeddings_json)
    print("\nPipeline de ingestao concluido!")
