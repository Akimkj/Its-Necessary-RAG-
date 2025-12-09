from pydantic import BaseModel, Field, field_validator
import re

def clean_text(text: str):
    text_cleaned = re.sub(r'\s+', ' ', text).strip()
    return text_cleaned

class QAPairModel(BaseModel):
    expectedQuestion: str = Field(strip_whitespace=True, max_length=1000)
    expectedAnswer: str = Field(strip_whitespace=True, max_length=1000)
    id: int = Field(gt=0)


    @field_validator('expectedQuestion', 'expectedAnswer', mode='before')
    @classmethod
    def validade_content(cls, text: str):

        if not isinstance(text, str):
            text = str(text)

        new_text = clean_text(text)

        if not new_text:
            raise ValueError("O campo não pode ser vazio")
        if new_text.lower() in ["não sei", "sem resposta"]:
            raise ValueError("Placehorder invalido")
        
        return new_text
    
    @field_validator('expectedAnswer', mode='after')
    @classmethod
    def check_final_answer(cls, answer: str):
        if len(answer.split()) < 10:
            raise ValueError(f"Resposta esperada \n'{answer}'\n tem menos de 10 palavras, muito pequena")
        return answer


