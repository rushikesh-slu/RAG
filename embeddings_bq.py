import mimetypes
import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch

PROJECT_ID = 'durable-return-430917-b5'
embeddings = VertexAIEmbeddings(
        model_name="textembedding-gecko",
        batch_size=1,
        requests_per_minute=60
    )

store = BigQueryVectorSearch(
        project_id=PROJECT_ID,
        dataset_name='vector_db',
        table_name='test',
        embedding=embeddings,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

def load_and_split_document(file_path):
    # Determine the MIME type of the file
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Select the appropriate loader based on the file type
    if mime_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
    elif mime_type == 'text/plain':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")
    
    # Load the document
    document = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    return chunks

def upload_to_bq(chunks):
    texts = []
    metadatas = []
   
    for chunk in chunks:
      if chunk.page_content.strip():
                  texts.append(chunk.page_content)
                  metadatas.append(chunk.metadata)
                 
    
    store.add_texts(texts, metadatas=metadatas)
    

def load_file_from_GCS():
    from langchain_google_community import GCSFileLoader
    loader = GCSFileLoader(project_name="durable-return-430917-b5", bucket="durable-return-430917-docs", blob="Test/Invitation_letter.pdf")
    document = loader.load()
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    print(chunks)
    # return chunks





# all_texts = ["Apples and oranges", "Cars and airplanes", "Pineapple", "Train", "Banana"]
# metadatas = [{"len": len(t)} for t in all_texts]


# chunks = load_and_split_document('Experience.txt')
# upload_to_bq(chunks)

# load_file_from_GCS()

query = "Research"
docs = store.similarity_search(query,k=1)
print(docs[0].page_content)

