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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from sentence_transformers import CrossEncoder

# Load API Key
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("ERROR: OPENAI_API_KEY is missing. Please set it in .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
pdf_path = "storage/income_tax_act.pdf"
faiss_index_path = "faiss_index"

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Check if FAISS index exists
if os.path.exists(faiss_index_path):
    try:
        logger.info("Loading FAISS index...")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise RuntimeError("Failed to load FAISS index. Check if index file exists.")
else:
    logger.info("FAISS index not found. Processing PDF and creating embeddings...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"ERROR: PDF file not found at {pdf_path}.")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    logger.info("FAISS index created and saved successfully.")

# Initialize re-ranking model
re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Set up retriever
retriever = vectorstore.as_retriever()

# Initialize OpenAI LLM
llm = ChatOpenAI(api_key=openai_api_key)

# Initialize Memory for Conversations
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create FastAPI app
app = FastAPI()

# User memory (to store previous chats & user profile)
user_profiles: Dict[str, Dict] = {}

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Context:
    {context}

    User Profile:
    Name: {name}
    Salary: ₹{salary}
    tax_comparison: {tax_comparison}
    Deductions: {deductions}
    Investments: {investments}

    Previous Chat History:
    {chat_history}

    Task:
    The user is asking: "{question}"
    - If tax-related, calculate tax liability step by step.
    - **For the Old Regime**: Apply deductions and compute tax using slabs.
    - **For the New Regime**: Compute tax without deductions.
    - Provide a **detailed breakdown** of the tax calculation.
    - Compare both regimes and suggest which is better.
    - If needed, explain how deductions are applied.
    """
)
class CalculateTax:
    @staticmethod
    def old_regime(salary):
        """Calculate tax under the Old Regime (without deductions)."""
        taxable_income = max(0, salary - 250000)  # Income above exemption

        tax = 0
        slabs = [
            (250000, 0.05),  # 5% tax from 2.5L to 5L
            (500000, 0.20),  # 20% tax from 5L to 10L
            (float("inf"), 0.30)  # 30% tax above 10L
        ]

        tax_breakdown = []
        for slab, rate in slabs:
            if taxable_income <= 0:
                break
            taxable_at_slab = min(taxable_income, slab)
            tax_amount = taxable_at_slab * rate
            tax += tax_amount
            taxable_income -= taxable_at_slab
            tax_breakdown.append(f"₹{taxable_at_slab:,.2f} taxed at {rate*100:.0f}% = ₹{tax_amount:,.2f}")

        return tax, tax_breakdown

    @staticmethod
    def new_regime(salary):
        """Calculate tax under the New Regime (no deductions)."""
        taxable_income = max(0, salary - 300000)  # Income above exemption

        tax = 0
        slabs = [
            (300000, 0.05),  # 5% from 3L to 6L
            (300000, 0.10),  # 10% from 6L to 9L
            (300000, 0.15),  # 15% from 9L to 12L
            (300000, 0.20),  # 20% from 12L to 15L
            (float("inf"), 0.30)  # 30% above 15L
        ]

        tax_breakdown = []
        for slab, rate in slabs:
            if taxable_income <= 0:
                break
            taxable_at_slab = min(taxable_income, slab)
            tax_amount = taxable_at_slab * rate
            tax += tax_amount
            taxable_income -= taxable_at_slab
            tax_breakdown.append(f"₹{taxable_at_slab:,.2f} taxed at {rate*100:.0f}% = ₹{tax_amount:,.2f}")

        return tax, tax_breakdown




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
    #user_name: str
    #salary: float
    message: str

# **API Endpoint: Save User Profile**
@app.post("/save_user")
async def save_user_profile(profile: UserProfile):
    user_profiles[profile.user_id] = profile.dict()
    return {"message": "User profile saved successfully", "profile": profile}

@app.post("/chat")
async def chat_with_bot(chat: ChatMessage):
    user_id = chat.user_id
    #user_name = chat.user_name
    #salary = chat.salary
    user_message = chat.message
    #print("user_id:", user_id, "user_name:", user_name, "salary:", salary, "user_message:", user_message)
    if user_id not in user_profiles:
        return {"response": "Please set up your tax profile first!"}

    # Retrieve user profile
    user_info = user_profiles[user_id]
    salary = user_info["salary"]

    # Calculate taxes (Old Regime - No deductions, New Regime - No deductions)
    old_tax, old_breakdown = CalculateTax.old_regime(salary)
    new_tax, new_breakdown = CalculateTax.new_regime(salary)

    tax_comparison = f"""
    **📌 Tax Calculation Breakdown**
    
    **🔷 Old Regime (No Deductions)**
    - Gross Salary: ₹{salary:,.2f}
    - **Taxable Income:** ₹{salary:,.2f}
    - Tax Computation:
      {"\n      ".join(old_breakdown)}
    - **Total Tax (Old Regime): ₹{old_tax:,.2f}**

    **🔷 New Regime (No Deductions)**
    - Gross Salary: ₹{salary:,.2f}
    - **Taxable Income:** ₹{salary:,.2f}
    - Tax Computation:
      {"\n      ".join(new_breakdown)}
    - **Total Tax (New Regime): ₹{new_tax:,.2f}**

    **💡 Suggested Action:** Tax calculations vary based on individual exemptions, rebates, and surcharges. Please consult a **tax auditor** for personalized guidance.  

    **✅ Best Option: {"Old Regime ✅" if old_tax < new_tax else "New Regime ✅"} (Lower Tax Liability)**
    """

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(user_message)

    if retrieved_docs:
        reranked_scores = re_ranker.predict([(user_message, doc.page_content) for doc in retrieved_docs])
        sorted_docs = sorted(zip(retrieved_docs, reranked_scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc[0].page_content for doc in sorted_docs[:3]]
        context = "\n".join(top_docs)
    else:
        context = "No direct reference found."

    # Retrieve conversation memory
    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    # Generate AI response
    prompt = prompt_template.format(
        context=context,
        name=user_info["name"],
        salary=salary,
        tax_comparison=tax_comparison,
        deductions=", ".join(user_info["deductions"]),
        investments=", ".join(user_info["investments"]),
        chat_history=chat_history,
        question=user_message
    )
    
    response = llm.invoke(prompt)

    # Store conversation in memory
    memory.save_context({"input": user_message}, {"output": response.content})

    return {"response": f"{tax_comparison}\n\n{response.content}"}


# **API Endpoint: Clear Memory for a User**
@app.post("/clear_memory")
async def clear_user_memory(user_id: str = Body(..., embed=True)):
    memory.clear()
    return {"message": f"Memory cleared for user {user_id}."}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Tax Advisor API!"}
