import os
import json
import google.generativeai as genai

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# working directory 
working_dir = os.path.dirname(os.path.abspath(__file__))

config_file_path = f"{working_dir}/config.json"
config_data = json.load(open(config_file_path))
# print(config_data)

# load google api key 
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]
# print(GOOGLE_API_KEY)

# configuring google.generative with API key
key = genai.configure(api_key=GOOGLE_API_KEY) 

# loading gemini pro model
def load_gemini_pro_model() :
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model

# function for image capationing
def gemini_pro_vision_response(prompt,image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = gemini_pro_vision_model.generate_content([prompt,image])
    result = response.text
    return result

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result

# output = gemini_pro_response("what is machine learning?")
# print(output)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# chat with PDF 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def about():
        st.title('👩‍💻 About ')
        st.write("""
    <div style="text-align: justify;">
    <h3> Generative AI web app, seamlessly integrating: </h3>
       <ul> <li> <b> Chatbot Assistant: </b> Engage in natural conversations and get personalized answers.</li>
                <li> <b> Image Captioning: </b>Generate comprehensive captions for your images.</li>
                <li> <b> Ask Me Anything: </b> Access a vast knowledge base with instant answers.</li>
                <li> <b> Chat with PDF Pages: </b> Chat with text extracted from PDF pages. </li>
        </ul>
        <ul>  Built with Streamlit and Python, our web app offers a user-friendly and accessible platform for AI/ML exploration.
        </ul>
    </div>
    <ul style="list-style-type: none;">
    <h3> Guided By:</h3>
    <p>
    Abdul Aziz md
    <br>
    Master trainer
    <br>
    Edunet Foundation
    </p>
    <h3> Presented By:</h3>
    <p>Komal Suthar</p>
    </ul>
    """, unsafe_allow_html=True)