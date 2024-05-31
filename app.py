import os
from dotenv import load_dotenv
import time
#import arxiv
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
import gradio as gr
import urllib
import re
import cohere
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Batch, PointStruct
import torch

load_dotenv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
qdrant_client = QdrantClient(host='localhost', port=6333)
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(api_key=cohere_api_key)

def download_pdf_with_retry(url, path, max_retries=5):
    retry_delay = 1  # start with 1 second delay
    for attempt in range(max_retries):
        try:
            # Attempt to download the PDF
            urllib.request.urlretrieve(url, path)
            print(f"Download successful: {path}")
            return  # Exit the function if download is successful
        except urllib.error.ContentTooShortError as e:
            print(f"Download failed on attempt {attempt + 1} due to incomplete retrieval: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Double the delay for the next retry
        except (FileNotFoundError, ConnectionResetError) as e:
            print(f"Download failed on attempt {attempt + 1}: {e}")
            time.sleep(retry_delay)
            # You might choose not to increase delay for these errors, depending on their nature
    else:
        # If the loop completes without returning, all retries have failed
        print(f"Failed to download after {max_retries} attempts.")

def sanitize_filename(title):
    """Sanitize the paper title to be used as a filename."""
    sanitized = re.sub(r'[^a-zA-Z0-9 \n.]', '', title)  
    sanitized = sanitized.replace(' ', '_')  # Replace spaces with underscores
    return sanitized

def generate_chunk_uid(title, paper_id, chunk_index):
    """Generate a unique identifier for a text chunk."""
    sanitized_title = re.sub(r'[^a-zA-Z0-9]', '', title)
    return f"{sanitized_title}_{paper_id}_chunk{chunk_index}"

# def embeddings_exist(collection_name, chunk_uid, qdrant_client):
#     """
#     Check if embeddings for a given chunk UID already exist in Qdrant by using the filter functionality.
    
#     Args:
#         collection_name (str): The name of the collection in Qdrant.
#         chunk_uid (str): The unique identifier of the text chunk.
#         qdrant_client: Instance of the Qdrant client for API interactions.
    
#     Returns:
#         bool: True if the embedding exists, False otherwise.
#     """
#     try:
#         # Define a filter that matches points with the specified chunk_uid in their payload
#         filter_condition = {
#             "must": [
#                 { "key": "chunk_uid", "match": { "value": chunk_uid } }
#             ]
#         }
#         # Perform a search with the filter, looking for at least one matching point
#         search_response = qdrant_client.search(
#             collection_name=collection_name,
#             filter=filter_condition,
#             top=1  # We only need to check if at least one result comes back
#         )
#         # Check if any points were returned in the search response
#         return len(search_response["result"]) > 0
#     except Exception as e:
#         print(f"Error checking for embedding existence: {e}")
#         return False


def store_embedding_in_qdrant(collection_name, chunk_uid, embedding, qdrant_client):
    """
    Store the embedding of a text chunk in the Qdrant database with its UID.
    
    Args:
        collection_name (str): The name of the collection in Qdrant.
        chunk_uid (str): The unique identifier of the text chunk.
        embedding (list): The embedding vector of the text chunk.
        qdrant_client: Instance of the Qdrant client for API interactions.
    """
    try:
        # Ensure embedding is a list of floats
        if isinstance(embedding, list) and all(isinstance(x, float) for x in embedding):
            points_batch = Batch(ids=[chunk_uid], vectors=[embedding])
            qdrant_client.upsert(collection_name=collection_name, points=points_batch)
            print(f"Successfully stored embedding for {chunk_uid}.")
        else:
            print("Embedding format error: Embedding must be a list of floats.")
    except Exception as e:
        print(f"Error storing embedding in Qdrant: {e}")
# 1. Search arxiv for papers and download them
def process_papers(query, question_text):
    dirpath = "arxiv_papers"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # search arxiv for papers 
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, 
        max_results=5,
        sort_order=arxiv.SortOrder.Descending
    )

    # Download and save the papers 
    for result in client.results(search):
        pdf_url = result.pdf_url
        paper_id = result.get_short_id()  # Extract paper ID
        sanitized_title = sanitize_filename(result.title)
        # Combine sanitized title and paper ID for a unique filename
        pdf_filename = f"{sanitized_title}_{paper_id}.pdf"
        pdf_path = os.path.join(dirpath, pdf_filename)
        
        # Check if the file already exists
        if not os.path.exists(pdf_path):
            download_pdf_with_retry(pdf_url, pdf_path)
            print(f"Downloaded and saved: {pdf_path}")
        else:
            print(f"File already exists and will not be downloaded again: {pdf_path}")

    # 2. Load the papers, concatenate them, and split into chunks
    papers = []
    loader = PyPDFDirectoryLoader("arxiv_papers/") 
    try:
        papers = loader.load()
    except Exception as e:
        print(f"Error loading papers: {e}")

    # full_text = " ".join(paper.page_content for paper in papers)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # paper_chunks = text_splitter.create_documents([full_text])

    # chunk_index = 0
    # for chunk in paper_chunks:
    #     chunk_uid = generate_chunk_uid(sanitized_title, paper_id, chunk_index)
    #     if not embeddings_exist('arxiv_papers', chunk_uid, qdrant_client):
    #         # Generate and store embeddings if they don't exist
    #         embedding = GPT4AllEmbeddings().embed_query(chunk.page_content)
    #         store_embedding_in_qdrant('arxiv_papers', chunk_uid, embedding, qdrant_client)
    #     else:
    #         print(f"Embedding for chunk UID {chunk_uid} already exists. Skipping.")
    #     chunk_index += 1

    print("Total number of papers loaded: ", len(papers))

    # Combine all the pages in paper into a single string
    full_text = ""
    for paper in papers:
        full_text += paper.page_content

    # Remove empty lines and join lines into a single string
    full_text = " ".join(line for line in full_text.splitlines() if line)
    print("Total number of characters in the papers: ", len(full_text))

    # Split the text into chunck
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])
    # Check for empty paper chunks
    if not paper_chunks:
        print("No text chunks available for embedding.")
        return 
    
    # Verify the embeddings process is working correctly
    text_chunks = [paper.page_content for paper in paper_chunks]  
    test_embedding = GPT4AllEmbeddings().embed_query(text_chunks[0] if text_chunks else "")

    if not test_embedding:
        print("Failed to generate embeddings for the test chunk.")
        return

    #3. create Qdrant vector store and store embeddings
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding= GPT4AllEmbeddings(),
        path="./tmp/local_qdrant",
        # collection_name="arxiv_papers",
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    )
    retreival = qdrant.as_retriever()

    # 4. Define prompt template and initialize Ollama
    template = """Answer the following questions based on only the given context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # initialize Ollama
    ollama_llm = "llama2:7b-chat"
    model = ChatOllama(model=ollama_llm)

    # Define the processing chanin
    chain = (
        RunnableParallel({"context": retreival, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

# docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")

    # Add typing for input
    class Question(BaseModel):
        __root__: str

    # Apply input type to the chain
    chain = chain.with_types(input_type=Question)
    result = chain.invoke(question_text)
    return result

# # # Ask a question
# iface = gr.Interface(
#     fn=process_papers,
#     inputs=["text", "text"],
#     outputs="text",
#     description=
#     """
#         This interface allows you to search for academic papers from arXiv based on a search query 
#     and then ask a question related to the content of these papers. It downloads the papers, 
#     processes them to extract text, and uses a language model to generate responses to your question. 
#     First, enter a search query to find papers related to your topic of interest. Then, ask a specific 
#     question about these papers. The system will attempt to provide an answer based on the content 
#     of the downloaded papers. Ideal for researchers, students, or anyone interested in gaining insights 
#     from scientific literature quickly.
#     """,
# )


# iface.launch(share=True)

import streamlit as st



# Set up the Streamlit app
st.title("Ask a question")
st.markdown(
    """
    This interface allows you to search for academic papers from arXiv based on a search query 
    and then ask a question related to the content of these papers. It downloads the papers, 
    processes them to extract text, and uses a language model to generate responses to your question. 
    First, enter a search query to find papers related to your topic of interest. Then, ask a specific 
    question about these papers. The system will attempt to provide an answer based on the content 
    of the downloaded papers. Ideal for researchers, students, or anyone interested in gaining insights 
    from scientific literature quickly.
    """
)

# Input fields
search_query = st.text_input("Enter your search query")
question = st.text_input("Ask a specific question about the papers")

# Button to trigger the processing
if st.button("Submit"):
    if search_query and question:
        answer = process_papers(search_query, question)
        st.write(answer)
    else:
        st.write("Please provide both a search query and a question.")

# button to trigger the processing
if st.button("Submit"):
    # check if both search_query and question are provided
    if search_query and question:
        # process the papers with the provided search_query and question
        answer = process_papers(search_query, question)
        # display the answer
        st.write(answer)
    else:
        # display a message to provide both fields
        st.write("Please provide both a search query and a question.")