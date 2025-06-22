from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chain import get_rag_response
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        response = get_rag_response(query)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
