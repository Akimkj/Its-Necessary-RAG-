from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document]) -> list[Document]:
    """
    ### split_documents
    Implementa o uso da função RecursiveCharacterTextSplitter, onde pega o texto da documentação e divide com base num tamanho específicado e seguindo semânticamente o texto. Mais informações: https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    return chunks