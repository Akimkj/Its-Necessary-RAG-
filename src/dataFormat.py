from pydantic import BaseModel, Field, field_validator
import re
from typing import List

def clean_text(text: str):
    text_cleaned = re.sub(r'\s+', ' ', text).strip()
    return text_cleaned

class QAPairModel(BaseModel):
    id: int = Field(gt=0)
    expectedQuestion: str = Field(strip_whitespace=True)
    expectedAnswer: str = Field(strip_whitespace=True)
    


    @field_validator('expectedQuestion', 'expectedAnswer', mode='before')
    @classmethod
    def validade_content(cls, text: str):

        if not isinstance(text, str):
            text = str(text)

        new_text = clean_text(text)

        if not new_text:
            raise ValueError("O campo não pode ser vazio")
        if new_text.lower() in ["i don't know", "no response"]:
            raise ValueError("Placehorder invalido")
        
        return new_text
    
    @field_validator('expectedAnswer', mode='after')
    @classmethod
    def check_final_answer(cls, answer: str):
        if len(answer.split()) < 10:
            raise ValueError(f"Resposta esperada \n'{answer}'\n tem menos de 10 palavras, muito pequena")
        return answer


class QADataSet(BaseModel):
    data: List[QAPairModel] = Field(default_factory=list)


dataset = QADataSet()

