import os
import json
from dotenv import load_dotenv

from data_loader import load_pdfs, load_json, load_db_data
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

pdf_documents = load_pdfs()
json_data = load_json()
db_text = load_db_data()

pdf_text = "\n".join([doc.page_content for doc in pdf_documents])
json_text = json.dumps(json_data, indent=2)
combined_text = f"{pdf_text}\n{json_text}\n{db_text}"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
documents = text_splitter.create_documents([combined_text])

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

system_prompt = (
    "You are an AI assistant providing clear, concise, and structured responses with a professional yet friendly tone. "
    "Ensure responses are well-organized and easy to read. "
    "If answering about eligibility, fees, duration, or placements, keep the response informative and to the point. "
    "Avoid unnecessary phrases like 'I hope this helps' or 'Let me know if you have any questions.' "
    "Keep responses within 3-4 lines while maintaining clarity and warmth.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def get_rag_response(query):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    return response['answer']
