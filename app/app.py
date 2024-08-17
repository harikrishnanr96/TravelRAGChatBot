from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers, LlamaCpp
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, DirectoryLoader, TextLoader
import os
from dotenv import load_dotenv
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from accelerate import Accelerator
from langchain.retrievers.document_compressors import CohereRerank
from langchain_openai import ChatOpenAI
import openai
import asyncio
import torch
from flask import Flask, request, jsonify, redirect, url_for, render_template, session, render_template_string

load_dotenv()

class ChatbotConfig:
    def __init__(self) -> None:
        self.app = Flask(__name__)
        self.app.secret_key = os.urandom(24)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accelerator = Accelerator()
        self.DB_FAISS_PATH = "vectorstores/db_faiss"
        openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptManager:
    def __init__(self) -> None:
        self.custom_prompt_template =  """
            Use the following pieces of information to answer the user's question.
            If you don't know the answer, don't try to make up an answer.
            Context: {context}
            Question: {question}
            Only return the helpful answer below and nothing else.
            Helpful answer:
        """
    def get_prompt_template(self):
        return PromptTemplate(template=self.custom_prompt_template, input_variables=['context', 'question'])

class LLMSelect:
    def __init__(self, model_name, accelerator):
        self.model_name = model_name
        self.accelerator = accelerator
        self.llm = self.load_llm()
    
    def load_llm(self):
        config = {'max_new_tokens': 1024, 'repetition_penalty': 1.1, 'context_length': 8000, 'temperature': 0.5}

        if self.model_name == "Llama-2-7B":
            llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", config=config)
        elif self.model_name == "Mistral":
            llm = CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_type="mistral", config=config)
        elif self.model_name == "chat-gpt-3.5":
            llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.5)
        else:
            raise ValueError(f"Unrecognized model name: {self.model_name}")
        
        if self.model_name in ["Llama-2-7B", "Mistral"]:
            llm, config = self.accelerator.prepare(llm, config)
        return llm

class VectorStoreManager:
    def __init__(self, db_faiss_path, device, documents):
        self.db_faiss_path = db_faiss_path
        self.device = device
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})
        self.db = self.load_db()
        cohere_api_key = os.getenv('COHERE_API_KEY')

        self.bm25_retriever = BM25Retriever.from_documents(documents, k=2)

        self.db_retriever = self.db.as_retriever(search_kwargs={'k': 2})

        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.db_retriever], weights=[0.7, 0.3])

        self.compressor = CohereRerank(cohere_api_key=cohere_api_key)

        self.compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor, base_retriever=self.ensemble_retriever)

    def load_db(self):
        return FAISS.load_local(self.db_faiss_path, self.embeddings, allow_dangerous_deserialization=True)
    
    def get_retriever(self):
        return self.compression_retriever
    
class RetrievalChain:
    def __init__(self, retriever, llm, prompt_template):
        self.retriever = retriever
        self.prompt_template = prompt_template  # Keep the PromptTemplate object
        self.qa_chain = self.create_retrieval_qa_chain(llm)
    
    def create_retrieval_qa_chain(self, llm):
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.prompt_template}  # Pass the PromptTemplate object directly
        )
    
    def get_answer(self, query):
        return self.qa_chain({'query': query})

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ChatbotApp:
    def __init__(self):
        try:
            self.config = ChatbotConfig()
            self.prompt_manager = PromptManager()
            
            self.documents = self.load_documents()

            self.vector_store_manager = VectorStoreManager(self.config.DB_FAISS_PATH, self.config.device, self.documents)
            self.retriever = self.vector_store_manager.get_retriever()

            self.app = self.config.app
            self.setup_routes()
        except Exception as e:
            logger.error("Error during initialization: %s", e, exc_info=True)
            raise

    def load_documents(self):
        try:
            loader = DirectoryLoader("data/")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
            documents = loader.load_and_split(text_splitter)

            return documents
        except Exception as e:
            logger.error("Error loading documents: %s", e, exc_info=True)
            raise

    def setup_routes(self):
        @self.app.route('/', methods=["GET", "POST"])
        def index():
            try:
                if request.method == "POST":
                    selected_llm = request.form["llm_choice"]
                    session["llm"] = selected_llm
                    print("Selected LLM:", session["llm"])
                    return redirect(url_for("chatbot"))
                return render_template('index.html')
            except Exception as e:
                logger.error("Error in index route: %s", e, exc_info=True)
                return "An error occurred", 500

        @self.app.route('/chatbot')
        def chatbot():
            try:
                return render_template('chatbot.html')
            except Exception as e:
                logger.error("Error in chatbot route: %s", e, exc_info=True)
                return "An error occurred", 500
        
        @self.app.route('/chat', methods=["GET", "POST"])
        def chat():
            try:
                user_input = request.form["user_input"]
                session["query"] = user_input
                selected_llm = session.get("llm", 'Llama-2-7B')
                llm_select = LLMSelect(selected_llm, self.config.accelerator)
                retrieval_qa = RetrievalChain(self.retriever, llm_select.llm, self.prompt_manager.get_prompt_template())
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = retrieval_qa.get_answer(user_input)
                return result['result']
            except Exception as e:
                logger.error("Error in chat route: %s", e, exc_info=True)
                return "An error occurred while processing your request", 500
        
    def run(self):
        try:
            self.app.run(debug=True)
        except Exception as e:
            logger.error("Error running the application: %s", e, exc_info=True)
            raise

app_instance = ChatbotApp().app

if __name__ == '__main__':
    app = ChatbotApp()
    app.run(debug=True)
