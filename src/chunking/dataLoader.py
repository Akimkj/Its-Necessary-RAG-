from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def Docs_load():

    PATH_DOCS_PYTHON = os.path.join("data", "docs", "python-docs-html")

    loader = ReadTheDocsLoader(PATH_DOCS_PYTHON, features="lxml")
    rawDocs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )