# Its-Necessary-RAG-?
O objetivo desta pesquisa é definir se é realmente necessário o uso de RAG na geração de respostas com base em documentações em comparação com o uso das LLM's.


## Prompts
Neste tópico, será mostrado os prompts usados para a geração das respostas geradas (tanto a versão com ou sem RAG):

1. Versão 1
"You are an expert in Computer Science and Python documentation.", 
"Answer the given question completely, technically, and directly, but without introductions like 'Sure', 'okay', 'certainly', etc.",
"Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'expectedQuestion': a string that will be the question provided; 'expectedAnswer': a string that will be the returned answer."



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
