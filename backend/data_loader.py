import os
import json
import sqlite3
from langchain_community.document_loaders import PyPDFLoader

DATA_FOLDER = "Data"
DB_FILE = os.path.join(DATA_FOLDER, "college.db")

def load_pdfs():
    pdf_texts = []
    path = os.path.join(DATA_FOLDER, "NGPASC.pdf")
    if os.path.exists(path):
        loader = PyPDFLoader(path)
        pdf_texts.extend(loader.load())
    return pdf_texts

def load_json():
    path = os.path.join(DATA_FOLDER, "HT.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_db_data():
    if not os.path.exists(DB_FILE):
        return ""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    db_text = []
    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        for row in rows:
            db_text.append(" | ".join(f"{cols[i]}: {row[i]}" for i in range(len(cols))))
    conn.close()
    return "\n".join(db_text)
