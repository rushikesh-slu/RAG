
import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
import mimetypes
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from google.cloud import storage
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch

# def upload_to_gcs(destination_blob_name, file_path):
    
#     bucket_name = 'durable-return-430917-rag'
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
    
#     blob.upload_from_filename(file_path)

# def load_and_split_document(file_path):
#     # Determine the MIME type of the file
#     mime_type, _ = mimetypes.guess_type(file_path)
    
#     # Select the appropriate loader based on the file type
#     if mime_type == 'application/pdf':
#         loader = PyPDFLoader(file_path)
#     elif mime_type == 'text/plain':
#         loader = TextLoader(file_path)
#     else:
#         raise ValueError(f"Unsupported file type: {mime_type}")
    
#     # Load the document
#     document = loader.load()
    
#     # Split the document into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
#     chunks = text_splitter.split_documents(document)

#     original_file_name = os.path.basename(file_path)
    

#     upload_to_gcs(original_file_name, file_path)
    
#     save_to_chroma(chunks)



# def save_to_chroma(chunks):
#     embeddings = VertexAIEmbeddings(
#         model_name="textembedding-gecko",
#         batch_size=1,
#         requests_per_minute=60
#     )
#     db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
#     batch_size = 10  # Adjust as needed
    
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i+batch_size]
#         texts = []
#         metadatas = []
#         ids = []
#         curr_sum = 0
#         lastidx = ''
#         for chunk in batch:
#             if chunk.page_content.strip():
#                 texts.append(chunk.page_content)
#                 metadatas.append(chunk.metadata)
#                 chunksrc = chunk.metadata.get("source").split('\\')[-1]
#                 if chunksrc == lastidx:
#                     curr_sum+=1
#                     id = chunksrc + str(curr_sum)
#                 else:
#                     curr_sum = 1
#                     id = chunksrc + str(curr_sum)
#                 lastidx = chunksrc
#                 ids.append(id)


        
#         if texts:
#             try:
#                 db.add_texts(texts=texts, metadatas=metadatas,ids=ids)
               
#             except Exception as e:
#                 print(f"Error adding batch {i//batch_size + 1} to Chroma: {e}")
    
#     db.persist()

#     return db


# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = VertexAIEmbeddings(
#         model_name="textembedding-gecko",
#         batch_size=1,
#         requests_per_minute=60
#     )
#     db = Chroma(persist_directory='chroma_db', embedding_function=embedding_function)

#     # print(len(db.get()['ids']))
    

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=1)
#     documents = []
#     for doc,_score in  results:
#         documents.append(doc.metadata["source"].split('\\')[-1])
#     print(documents)

#     docs = []
#     for document in documents:
#         docs.append(document+"1")
#         docs.append(document+"2")

#     doc_result = db.get(ids=docs)
#     query_docs = doc_result["documents"]

#     context_text = "\n\n---\n\n".join([doc for doc in query_docs])
#     return context_text
# load_and_split_document('requirements.txt')

def query_bq(query_text: str):
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
    
    docs = store.similarity_search(query_text,k=1)
    if docs:
        return docs[0].page_content
    return None


