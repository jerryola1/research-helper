import os
import time
import arxiv
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings 
import gradio as gr
import urllib
import re

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
    sanitized = re.sub(r'[^a-zA-Z0-9 \n.]', '', title)  # Remove invalid characters
    sanitized = sanitized.replace(' ', '_')  # Replace spaces with underscores
    return sanitized



# 1. Search arxiv for papers and download them
def process_papers(query, question_text):
    dirpath = "arxiv_papers"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # search arxiv for papers related to "LLM"
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, 
        max_results=10,
        sort_order=arxiv.SortOrder.Descending
    )

    # for result in client.results(search):
    #     pdf_url = result.pdf_url
    #     pdf_path = os.path.join(dirpath, f"{result.get_short_id()}.pdf")
    #     download_pdf_with_retry(pdf_url, pdf_path)

    for result in client.results(search):
        pdf_url = result.pdf_url
        sanitized_title = sanitize_filename(result.title)
        pdf_path = os.path.join(dirpath, f"{sanitized_title}.pdf")
        
        # Check if the file already exists
        if not os.path.exists(pdf_path):
            download_pdf_with_retry(pdf_url, pdf_path)
        else:
            print(f"File already exists: {pdf_path}")

    # Download and save the papers 
    # for result in client.results(search):
    #     while True:
    #         try:
    #             result.download_pdf(dirpath=dirpath)
    #             print(f"-> Paper id {result.get_short_id()} with title {result.title} is downloaded")
    #             break
    #         except (FileNotFoundError, ConnectionResetError) as e:
    #             print(f"Error downloading: {e}")
    #             time.sleep(5)

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
    test_embedding = GPT4AllEmbeddings().embed_text(paper_chunks[0] if paper_chunks else "")
    if not test_embedding:
        print("Failed to generate embeddings for the test chunk.")
        return

    #3. create Qdrant vector store and store embeddings
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding= GPT4AllEmbeddings(),
        path="./tmp/local_qdrant",
        collection_name="arxiv_papers",
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

    # Add typing for input
    class Question(BaseModel):
        __root__: str

    # Apply input type to the chain
    chain = chain.with_types(input_type=Question)
    result = chain.invoke(question_text)
    return result

# # Ask a question
# result = chain.invoke("Explain about vision enhancing LLMs")
# print(result)

iface = gr.Interface(
    fn=process_papers,
    inputs=["text", "text"],
    outputs="text",
    description="Enter a search query and a question to ask about the papers.",
)

iface.launch(share=True)