import os
import uvicorn
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

import google.generativeai as genai


# -----------------------------
# Load environment variables
# -----------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI()


# -----------------------------
# Request schema
# -----------------------------

class QuestionRequest(BaseModel):
    question: str


# -----------------------------
# MongoDB connection
# -----------------------------

client = MongoClient(MONGO_URI)

db = client["edulearn"]

collection = db["curriculum"]


# -----------------------------
# Embedding model
# -----------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Retrieve context
# -----------------------------

def retrieve_context(question):

    question_embedding = embedding_model.encode([question])

    documents = list(collection.find())

    texts = [doc["content"] for doc in documents]

    embeddings = embedding_model.encode(texts)

    nn = NearestNeighbors(n_neighbors=3)

    nn.fit(embeddings)

    distances, indices = nn.kneighbors(question_embedding)

    context = ""

    for i in indices[0]:
        context += texts[i] + "\n"

    return context


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(question):

    context = retrieve_context(question)

    prompt = f"""
Answer the question using the curriculum context.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# API endpoint
# -----------------------------

@app.post("/ask")
def ask_bot(request: QuestionRequest):

    question = request.question

    answer = generate_answer(question)

    return {"response": answer}


# -----------------------------
# Run server
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT"))

    uvicorn.run(app, host="0.0.0.0", port=port)
