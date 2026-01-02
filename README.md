# Its-Necessary-RAG-?
O objetivo desta pesquisa é definir se é realmente necessário o uso de RAG na geração de respostas com base em documentações em comparação com o uso das LLM's.


# Prompts
Neste tópico, será mostrado os prompts usados para a geração das respostas geradas (tanto a versão com ou sem RAG):

## Versão 1
"You are an expert in Computer Science and Python documentation.", 
"Answer the given question completely, technically, and directly, but without introductions like 'Sure', 'okay', 'certainly', etc.",
"Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'expectedQuestion': a string that will be the question provided; 'expectedAnswer': a string that will be the returned answer."



