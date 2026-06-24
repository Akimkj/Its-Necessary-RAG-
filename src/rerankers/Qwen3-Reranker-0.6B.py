import torch
from sentence_transformers import CrossEncoder

_MODEL = None

def _get_model() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        print("\nCarregando o modelo Qwen3-Reranker-0.6B para reranking...")
        # Seleciona o dispositivo (GPU se disponível, senão CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Carrega o CrossEncoder do Qwen3-Reranker-0.6B
        _MODEL = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", device=device, trust_remote_code=True)
        print("Modelo Qwen3-Reranker-0.6B carregado com sucesso!")
    return _MODEL

def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Realiza o reranking de uma lista de chunks com base na pergunta/query fornecida.
    
    Args:
        query (str): A pergunta ou termo de busca.
        chunks (list[dict]): Lista de dicionários representando os chunks retornados pela busca semântica.
                             Cada dicionário deve conter a chave 'chunk'.
                             
    Returns:
        list[dict]: A lista de chunks reordenada de forma decrescente pelo score do reranker,
                    com os valores de 'score' atualizados.
    """
    if not chunks:
        return []
        
    model = _get_model()
    
    # Prepara os pares (query, chunk_text)
    pairs = []
    for item in chunks:
        chunk_text = item.get('chunk', '')
        pairs.append((query, chunk_text))
        
    print(f"Iniciando reranking de {len(chunks)} chunks com Qwen3-Reranker-0.6B...")
    
    # Realiza a predição dos scores
    scores = model.predict(pairs)
    
    # Atualiza o score de cada chunk na lista de retorno
    reranked_chunks = []
    for item, score in zip(chunks, scores):
        item_copy = item.copy()
        item_copy['score'] = float(score)
        reranked_chunks.append(item_copy)
        
    # Ordena pelo novo score em ordem decrescente (do maior para o menor)
    reranked_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    print("Processo de reranking concluído com sucesso!")
    return reranked_chunks
