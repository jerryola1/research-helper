import os
import time
import arxiv
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputparser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharachterSplitter
from langchain_community.embeddings import GPT4AllEmbeddings 

# 1. Search arxiv for papers and download them
dirpath = "arxiv_papers"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# search arxiv for papers related to "LLM"
client = arxiv.Client()
search = client.search(
    query="LLM", 
    max_results=5,
    sort_order=arxiv.SortOrder.Descending
)

# Download and save the papers 
for result in client.results(search):
    while True:
        try:
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title {result.title} is downloaded")
            break
        except (FileNotFoundError, ConnectionResetError) as e:
            print(f"Error downloading: {e}")
            time.sleep(5)

# 2. Load the papers, concatenate them, and split into chunks
papers = []
loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader) 
try:
    papers = loader.load()
except Exception as e:
    print(f"Error loading papers: {e}")
print("Total number of papers loaded: ", len(papers))

# Combine all the pages in paper into a single string
full_text = ""
for paper in papers:
    full_text += paper.page_content

# Remove empty lines and join lines into a single string
full_text = "\n".join(line for line in full_text.splitlines("\n") if line)
print("Total number of characters in the papers: ", len(full_text))

# Split the text into chunck
