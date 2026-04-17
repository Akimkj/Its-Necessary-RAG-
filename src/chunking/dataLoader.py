from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_core.documents import Document

def load_documentation(path: str) -> list[Document]:
    """
    ### load_documentation
    Recebe o caminho da pasta contendo a documentação a ser lida.
    """

    loader = ReadTheDocsLoader(path, encoding="utf-8")
    rawDocs = loader.load()

    return rawDocs
    