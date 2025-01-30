import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Dict, List
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

# Load API Key
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("ERROR: OPENAI_API_KEY is missing. Please set it in .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
faiss_index_path = "faiss_index"

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Load FAISS index
try:
    logger.info("Loading FAISS index...")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    raise RuntimeError("Failed to load FAISS index. Check if index file exists.")

# Set up LLM
llm = ChatOpenAI(api_key=openai_api_key)

# Create FastAPI app
app = FastAPI()

# User memory (to store previous chats & user profile)
user_profiles: Dict[str, Dict] = {}

# Define prompt template for AI
prompt_template = ChatPromptTemplate.from_template(
    """
    Context:
    {context}

    User Profile:
    Name: {name}
    Salary: ₹{salary}
    Deductions: {deductions}
    Investments: {investments}

    Task:
    The user is asking: "{question}"
    - If the question is tax-related, analyze their profile, deductions, and salary.
    - If math is needed, calculate their tax according to the **Indian Old and New Tax Regime**.
    - If the question is general, provide an informative response.
    - Always return clear and helpful information.
    """
)

# Define User Profile Schema
class UserProfile(BaseModel):
    user_id: str
    name: str
    salary: float
    deductions: List[str] = []
    investments: List[str] = []

# Define Chat Message Schema
class ChatMessage(BaseModel):
    user_id: str
    message: str

# **API Endpoint: Save User Profile**
@app.post("/save_user")
async def save_user_profile(profile: UserProfile):
    user_profiles[profile.user_id] = profile.dict()
    return {"message": "User profile saved successfully", "profile": profile}

# **API Endpoint: AI-Powered Chat (Tax & General)**
@app.post("/chat")
async def chat_with_bot(chat: ChatMessage):
    user_id = chat.user_id
    user_message = chat.message

    if user_id not in user_profiles:
        return {"response": "Please set up your tax profile first!"}

    # Retrieve user profile data
    user_info = user_profiles[user_id]
    salary = user_info["salary"]
    deductions = ", ".join(user_info["deductions"])
    investments = ", ".join(user_info["investments"])

    # Retrieve relevant docs
    retrieved_docs = retriever.invoke(user_message)
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No direct reference found."

    # Generate AI response dynamically
    prompt = prompt_template.format(
        context=context,
        name=user_info["name"],
        salary=salary,
        deductions=deductions,
        investments=investments,
        question=user_message
    )
    response = llm.invoke(prompt)

    return {"response": response.content}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Tax Advisor API!"}
