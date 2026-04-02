# Its-Necessary-RAG-?
Projeto de Iniciação Científica focada na questão: É realmente necessário o uso de RAG na geração de respostas com base na documentação da linguagem Python, em comparação com o uso das LLM's?


## Como executar o programa

### 1. pré-requisitos
* Python 3.10 ou superior instalado.

### 2. Configuração do ambiente
* Primeiro, clone o repositório e crie o ambiente virtual:

```bash
python -m venv .venv
```

* Ative o ambiente virtual:

  * Windows: ```.\.venv\Scripts\activate```
  * Linux/MAC: ```source .venv/bin/activate```

  **OBS1**: Para desativar posteriormente o ambiente virtual: ```deactivate```

* Instale as dependências:
```bash
pip install -r requirements.txt
```

**OBS2**: O código-fonte foi refatorado de forma que a execução precise ser feita dentro do arquivo main.py



## Prompts (campo temporario)
Neste tópico, será mostrado os prompts usados para a geração das respostas geradas (tanto a versão com ou sem RAG):

1. Versão 1
"You are an expert in Computer Science and Python documentation.", 
"Answer the given question completely, technically, and directly, but without introductions like 'Sure', 'okay', 'certainly', etc.",
"Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'expectedQuestion': a string that will be the question provided; 'expectedAnswer': a string that will be the returned answer."

2. Versão 2
"You are a Senior Computer Science Professor and Python Core Developer specializing in technical documentation.", 
"Your goal is to provide a comprehensive, thorough, academic, and complete technical explanation of the proposed question. The answer should be at least 300 words to ensure depth.",
"CONSTRAINTS: 1. DO NOT use Markdown formatting (no bold '**', no italics '*', no headers '#'). Use plain text only. 2. For code examples, write them inline or in plain text blocks without backticks. 3. DO NOT use introductory phrases or conversational fillers. 4. Structure the response with clear logical paragraphs instead of bullet points. 5. Focus on the internal mechanics of Python (CPython implementation, memory management, or execution flow) whenever applicable.",
"Return ONLY a valid JSON with keys: 'id': integer; 'question': string (exactly as provided); 'answer': string (the full, plain-text technical explanation).",


3. Versão 3 (definitiva ??)
"You are an expert in Computer Science and Python documentation. Your task is to answer the provided question in a complete, technical, and detailed manner. Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the question provided; 'answer': a string that will be the returned answer. Rules: 1. Do NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input. If not provided, use 0; 4. The 'question' must exactly match the input question; 5. The 'answer' must be detailed, technical, and well-structured; 6. Escape all special characters properly inside JSON strings;  7. Do NOT include markdown; 8. Do NOT omit any required field."



## Referências

Este projeto utiliza o BERTScore para avaliação das respostas geradas.

```bibtex
@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}
```
