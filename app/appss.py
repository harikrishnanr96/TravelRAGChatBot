from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers,LlamaCpp
from langchain.chains import RetrievalQA
import chainlit as cl
import os
from langchain_openai import ChatOpenAI
import openai
import asyncio
import torch
from flask import Flask,request, jsonify,redirect,url_for,render_template,session,render_template_string

app = Flask(__name__)
app.secret_key = os.urandom(24)

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """ Use the following
 pieces of information to answer the user's question.
 If you don't know the answer, don't try to make up an answer.
 
 Context:{context}
 Question:{question}
 
 Only returns the helpful answer below and nothing else.
 Helpful answer: 
 """



def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template = custom_prompt_template,
                            input_variables=['context','question'])
    
    return prompt


def load_llm(model):
    config = {'max_new_tokens': 1024, 'repetition_penalty': 1.1, 'context_length': 8000, 'temperature':0.5}


    if model== "Llama-2-7B":

        llm = CTransformers( model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type = "llama",
        config = config)

 
    
    elif model == "Mistral":
        
        llm = CTransformers( model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_type = "mistral",
        config = config)

        

    elif model == "chat-gpt-3.5":

        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",temperature=0.5)
    
    else:
        raise ValueError(f"Unrecognized model name: {model}")
        
     
    return llm


def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = db.as_retriever(search_kwargs = {'k':2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt':prompt}
    )
    return qa_chain  



def qa_bot(selected_llm):
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device':device})
    db = FAISS.load_local(DB_FAISS_PATH,embeddings,allow_dangerous_deserialization=True)
    llm = load_llm(selected_llm)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt,db)

    return qa



def final_result(query):
    qa_result = qa_bot()
    response =qa_result({'query':query})
    return response



@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        selected_llm = request.form["llm_choice"]
        session["llm"] = selected_llm 
        print("Selected LLM:",session["llm"])
        return redirect(url_for("chatbot")) 
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    
    return render_template('chatbot.html')

@app.route("/chat",methods = ["GET","POST"])
def chat():
    user_input = request.form["user_input"]
    session["query"] = user_input
    selected_llm = session.get("llm", 'Llama-2-7B')
    print("ss",selected_llm)
    chain = qa_bot(selected_llm)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = chain({"query": user_input, "chat_history": []})
    sources = result["source_documents"]
    # if sources:
    #     answer += f"\n Sources" + str(sources)
    # else:
    #     answer += f"\nNo Sources Found"
    return result['result']




if  __name__ == '__main__':
    app.run(debug= True)








