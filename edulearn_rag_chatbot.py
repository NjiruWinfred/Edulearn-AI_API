"""
EduLearn RAG Chatbot API
=========================

Production-ready FastAPI application with:
- RAG (Retrieval Augmented Generation) using MongoDB vector storage
- Google Gemini AI integration
- Student lesson recommendation system
- Hybrid online/offline mode
- Chat history tracking

Author: Winfred Mutitu
Date: February 2026
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import logging
import os

# AI & ML Imports
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Database
from pymongo import MongoClient

# ========================================
# CONFIGURATION
# ========================================

# Load from environment variables (use .env file)
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "YOUR_API_KEY_HERE")
MONGODB_CONNECTION_STRING = os.getenv(
    "MONGODB_CONNECTION_STRING",
    "mongodb+srv://winniemutitu_db_user:3YjLUw0l5ekVlUDq@edulearn.4mcdat3.mongodb.net/?appName=Edulearn"
)
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "edulearn")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edulearn_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================================
# INITIALIZE SERVICES
# ========================================

# FastAPI app
app = FastAPI(
    title="EduLearn RAG Chatbot API",
    description="Intelligent chatbot with RAG, recommendations, and offline support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google AI
try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
    gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    logger.info("✅ Google Gemini AI initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini: {e}")
    gemini_model = None

# Initialize MongoDB
try:
    mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
    db = mongo_client[MONGODB_DATABASE_NAME]
    
    # Collections
    curriculum_collection = db["curriculum vectors"]
    chat_history_collection = db["messages"]
    students_collection = db["students_info"]
    
    logger.info("✅ MongoDB connected")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    mongo_client = None

# Initialize embedding model for RAG
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ Embedding model loaded")
except Exception as e:
    logger.error(f"❌ Embedding model failed: {e}")
    embedding_model = None

# ========================================
# DATA MODELS
# ========================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., min_length=1, description="Student's question")
    student_id: str = Field(..., description="Student ID")
    class_level: Optional[str] = Field("Junior 1", description="Student class level")
    subject: Optional[str] = Field(None, description="Subject context")
    force_offline: bool = Field(False, description="Force offline mode")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    answer: str
    mode: str  # "online" or "offline"
    context_used: bool
    recommended_lessons: Optional[List[str]] = None
    timestamp: datetime

class RecommendationRequest(BaseModel):
    """Request for lesson recommendations"""
    student_id: str
    quiz_score: float = Field(..., ge=0, le=100)
    attempt_number: int = Field(..., ge=1)
    time_spent_minutes: float = Field(..., ge=0)
    lesson_id: str

class RecommendationResponse(BaseModel):
    """Response with lesson recommendations"""
    success: bool
    recommended_lessons: List[Dict[str, str]]
    difficulty_level: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ai_status: str
    database_status: str
    embedding_status: str
    timestamp: datetime

# ========================================
# RAG FUNCTIONS (Retrieval Augmented Generation)
# ========================================

def get_context_from_vectors(question: str, top_k: int = 3) -> str:
    """
    Retrieve relevant context from MongoDB vector database
    Uses semantic search with embeddings
    """
    if not embedding_model or not curriculum_collection:
        logger.warning("⚠️ RAG components not available")
        return ""
    
    try:
        # Generate question embedding
        question_vector = embedding_model.encode(question).tolist()
        
        # Vector search in MongoDB (cosine similarity)
        pipeline = [
            {
                "$search": {
                    "index": "vector_index",
                    "knnBeta": {
                        "vector": question_vector,
                        "path": "vector",
                        "k": top_k
                    }
                }
            },
            {"$limit": top_k},
            {"$project": {"text": 1, "chapter": 1, "_id": 0}}
        ]
        
        results = list(curriculum_collection.aggregate(pipeline))
        
        if not results:
            logger.info("⚠️ No context found in vector database")
            return ""
        
        # Combine retrieved texts
        context = "\n\n".join([doc.get("text", "") for doc in results])
        logger.info(f"✅ Retrieved {len(results)} context chunks")
        
        return context
    
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        # Fallback to simple text search
        return get_context_fallback(question)

def get_context_fallback(question: str) -> str:
    """
    Fallback context retrieval using simple text search
    Used when vector search fails
    """
    try:
        results = curriculum_collection.find(
            {"$text": {"$search": question}},
            {"text": 1, "_id": 0}
        ).limit(3)
        
        context = "\n\n".join([doc.get("text", "") for doc in results])
        return context
    
    except Exception as e:
        logger.error(f"❌ Fallback search failed: {e}")
        return ""

# ========================================
# RECOMMENDATION SYSTEM
# ========================================

def get_student_data(student_id: str) -> Optional[Dict]:
    """Retrieve student data from database"""
    try:
        if not students_collection:
            return None
        
        student = students_collection.find_one({"student_id": student_id})
        return student
    
    except Exception as e:
        logger.error(f"❌ Failed to get student data: {e}")
        return None

def recommend_lessons(
    quiz_score: float,
    attempt_number: int,
    time_spent_minutes: float,
    lesson_id: str,
    top_k: int = 3
) -> Dict:
    """
    Recommend next lessons based on student performance
    Uses K-Nearest Neighbors with student activity features
    """
    try:
        # Determine difficulty level
        if quiz_score < 50:
            difficulty = "Recommend easier lesson"
            difficulty_level = "easy"
        elif quiz_score <= 75:
            difficulty = "Recommend similar level lesson"
            difficulty_level = "medium"
        else:
            difficulty = "Recommend advanced lesson"
            difficulty_level = "advanced"
        
        # In production, this would use the trained KNN model
        # For now, return rule-based recommendations
        recommended_lessons = [
            {
                "lesson_id": f"lesson_{difficulty_level}_1",
                "title": f"{difficulty_level.title()} Level Lesson 1",
                "difficulty": difficulty_level,
                "reason": difficulty
            },
            {
                "lesson_id": f"lesson_{difficulty_level}_2",
                "title": f"{difficulty_level.title()} Level Lesson 2",
                "difficulty": difficulty_level,
                "reason": "Based on your progress pattern"
            },
            {
                "lesson_id": f"lesson_{difficulty_level}_3",
                "title": f"{difficulty_level.title()} Level Lesson 3",
                "difficulty": difficulty_level,
                "reason": "Recommended for skill building"
            }
        ]
        
        return {
            "success": True,
            "recommended_lessons": recommended_lessons[:top_k],
            "difficulty_level": difficulty
        }
    
    except Exception as e:
        logger.error(f"❌ Recommendation failed: {e}")
        return {
            "success": False,
            "recommended_lessons": [],
            "difficulty_level": "unknown"
        }

# ========================================
# CHATBOT LOGIC
# ========================================

def generate_online_response(question: str, context: str, class_level: str) -> Optional[str]:
    """Generate response using Gemini AI with context"""
    if not gemini_model:
        return None
    
    try:
        if context:
            prompt = f"""You are a helpful tutor for secondary school students.

Answer ONLY using this lesson content:

{context}

Student Question: {question}
Class Level: {class_level}

Provide a clear, simple answer suitable for the student's level. If the question cannot be answered from the lesson content, say so politely."""
        else:
            prompt = f"""You are a helpful tutor for secondary school students.

Student Question: {question}
Class Level: {class_level}

This topic is not in the current curriculum. Provide a brief, general explanation suitable for the student's level."""
        
        response = gemini_model.generate_content(prompt)
        logger.info("✅ Online response generated")
        return response.text
    
    except Exception as e:
        logger.error(f"❌ Gemini generation failed: {e}")
        return None

def generate_offline_response(question: str, context: str) -> str:
    """Generate offline response using only context"""
    if context:
        return f"Based on the lesson content:\n\n{context[:500]}...\n\nNote: You're currently offline. For a more detailed explanation, please connect to the internet."
    else:
        return "I don't have information about that in the downloaded lessons. Please connect to the internet for more help, or check your downloaded materials."

def save_chat_history(student_id: str, question: str, response: str, mode: str):
    """Save chat interaction to database"""
    try:
        if not chat_history_collection:
            return
        
        chat_history_collection.insert_one({
            "student_id": student_id,
            "user_message": question,
            "bot_response": response,
            "mode": mode,
            "timestamp": datetime.utcnow()
        })
        
        logger.info(f"✅ Chat history saved for student {student_id}")
    
    except Exception as e:
        logger.error(f"❌ Failed to save chat history: {e}")

def hybrid_chat(
    question: str,
    student_id: str,
    class_level: str = "Junior 1",
    subject: Optional[str] = None,
    force_offline: bool = False
) -> Dict:
    """
    Main hybrid chat function
    Combines RAG + AI + Offline fallback
    """
    # Step 1: Retrieve context using RAG
    context = get_context_from_vectors(question)
    context_found = bool(context)
    
    # Step 2: Try online mode (unless forced offline)
    mode = "offline"
    answer = None
    
    if not force_offline and gemini_model:
        answer = generate_online_response(question, context, class_level)
        if answer:
            mode = "online"
    
    # Step 3: Fallback to offline mode
    if not answer:
        answer = generate_offline_response(question, context)
        mode = "offline"
    
    # Step 4: Get recommendations (if question is about recommendations)
    recommended_lessons = None
    if "recommend" in question.lower():
        student_data = get_student_data(student_id)
        if student_data:
            rec_result = recommend_lessons(
                quiz_score=student_data.get("quiz_score", 75),
                attempt_number=student_data.get("attempt_number", 1),
                time_spent_minutes=student_data.get("time_spent_minutes", 30),
                lesson_id=student_data.get("lesson_id", "default")
            )
            if rec_result["success"]:
                recommended_lessons = [l["lesson_id"] for l in rec_result["recommended_lessons"]]
    
    # Step 5: Save to history
    save_chat_history(student_id, question, answer, mode)
    
    return {
        "success": True,
        "answer": answer,
        "mode": mode,
        "context_used": context_found,
        "recommended_lessons": recommended_lessons,
        "timestamp": datetime.utcnow()
    }

# ========================================
# API ENDPOINTS
# ========================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EduLearn RAG Chatbot API",
        "version": "1.0.0",
        "features": [
            "RAG with vector search",
            "Google Gemini AI",
            "Lesson recommendations",
            "Hybrid online/offline mode",
            "Chat history tracking"
        ],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ai_status = "available" if gemini_model else "unavailable"
    db_status = "connected" if mongo_client else "disconnected"
    embedding_status = "loaded" if embedding_model else "unavailable"
    
    overall_status = "healthy" if (gemini_model and mongo_client) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        ai_status=ai_status,
        database_status=db_status,
        embedding_status=embedding_status,
        timestamp=datetime.utcnow()
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint
    
    Features:
    - RAG-based context retrieval
    - Hybrid online/offline mode
    - Automatic lesson recommendations
    - Chat history tracking
    """
    logger.info(f"📝 Chat request from student {request.student_id}: {request.question}")
    
    try:
        result = hybrid_chat(
            question=request.question,
            student_id=request.student_id,
            class_level=request.class_level,
            subject=request.subject,
            force_offline=request.force_offline
        )
        
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_endpoint(request: RecommendationRequest):
    """
    Lesson recommendation endpoint
    
    Returns personalized lesson recommendations based on student performance
    """
    logger.info(f"📊 Recommendation request for student {request.student_id}")
    
    try:
        result = recommend_lessons(
            quiz_score=request.quiz_score,
            attempt_number=request.attempt_number,
            time_spent_minutes=request.time_spent_minutes,
            lesson_id=request.lesson_id
        )
        
        return RecommendationResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{student_id}")
async def get_chat_history(student_id: str, limit: int = 50):
    """Get chat history for a student"""
    try:
        if not chat_history_collection:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        history = list(chat_history_collection.find(
            {"student_id": student_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        return {
            "success": True,
            "student_id": student_id,
            "count": len(history),
            "history": history
        }
    
    except Exception as e:
        logger.error(f"❌ History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/context/search")
async def search_context(query: str, limit: int = 5):
    """
    Search the curriculum vector database
    Useful for testing RAG functionality
    """
    try:
        context = get_context_from_vectors(query, top_k=limit)
        
        return {
            "success": True,
            "query": query,
            "context_found": bool(context),
            "context": context[:1000] if context else None
        }
    
    except Exception as e:
        logger.error(f"❌ Context search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# STARTUP & SHUTDOWN
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting EduLearn RAG Chatbot API...")
    logger.info(f"✅ AI Status: {'Available' if gemini_model else 'Unavailable'}")
    logger.info(f"✅ Database: {'Connected' if mongo_client else 'Disconnected'}")
    logger.info(f"✅ Embeddings: {'Loaded' if embedding_model else 'Unavailable'}")
    logger.info("✅ API Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down...")
    if mongo_client:
        mongo_client.close()
    logger.info("✅ Cleanup complete")

# ========================================
# RUN APPLICATION
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 EduLearn RAG Chatbot API")
    print("=" * 60)
    print(f"📝 Documentation: http://localhost:8000/docs")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print(f"💬 Chat Endpoint: POST http://localhost:8000/api/chat")
    print(f"📊 Recommendations: POST http://localhost:8000/api/recommend")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
