from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import sqlite3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import faiss
from langchain_community.vectorstores import FAISS

load_dotenv()

app = Flask(__name__)
CORS(app)

DATA_FOLDER = "Data"
DB_FILE = os.path.join(DATA_FOLDER, "college.db")


def load_pdfs():
    pdf_file = "NGPASC.pdf"
    pdf_texts = []
    file_path = os.path.join(DATA_FOLDER, pdf_file)
    if os.path.exists(file_path):
        try:
            loader = PyPDFLoader(file_path)
            pdf_texts.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")
    else:
        print(f"{pdf_file} not found in {DATA_FOLDER}.")
    return pdf_texts


def load_json_files():
    json_file = "HT.json"
    json_data = {}
    file_path = os.path.join(DATA_FOLDER, json_file)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data[json_file] = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    else:
        print(f"{json_file} not found in {DATA_FOLDER}.")
    return json_data

def load_db_data():
    if not os.path.exists(DB_FILE):
        print("Database file not found.")
        return ""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]

        db_text = []
        for table in tables:
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            for row in rows:
                db_text.append(" | ".join([f"{columns[i]}: {row[i]}" for i in range(len(columns))]))

        conn.close()
        return "\n".join(db_text)
    except Exception as e:
        print(f"Error loading DB {DB_FILE}: {e}")
        return ""


pdf_documents = load_pdfs()
json_data = load_json_files()
db_text = load_db_data()

pdf_text = "\n".join([doc.page_content for doc in pdf_documents])
json_text = json.dumps(json_data, indent=2)
all_text = f"{pdf_text}\n{json_text}\n{db_text}"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.create_documents([all_text])

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

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


@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})
        return jsonify({"answer": response['answer']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
