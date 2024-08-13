from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,CSVLoader,DirectoryLoader,TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH ="data\london_travel_data.csv"
DB_FAISS_PATH ="vectorstores/db_faiss"

def create_vector_db():
    csv_loader =CSVLoader(file_path= DATA_PATH,encoding="utf-8",csv_args={'delimiter':',',
                                                                      "fieldnames": ["Section", "Subsection", "Content"]})
    csv_documents = csv_loader.load()

    pdf_loader = DirectoryLoader('data/', glob = "**/*.pdf",show_progress=True,loader_cls=PyPDFLoader)
                                 
    pdf_documents  = pdf_loader.load()

    text_loader = DirectoryLoader("data/",glob="**/*.txt",loader_cls=TextLoader)

    text_documents = text_loader.load()

    loaded_documents = csv_documents + text_documents + pdf_documents
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
    texts = text_splitter.split_documents(loaded_documents)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device':'cpu'})
    
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()

