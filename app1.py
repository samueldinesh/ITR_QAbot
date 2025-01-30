import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Validate API key
if not openai_api_key:
    raise ValueError("ERROR: OPENAI_API_KEY is missing in .env file. Please set it before running the script.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
pdf_path = "storage/income_tax_act.pdf"
faiss_index_path = "faiss_index"

# Ensure PDF exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"ERROR: PDF file not found at {pdf_path}. Ensure it's placed correctly.")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Load or create FAISS index
if os.path.exists(faiss_index_path):
    logger.info("FAISS index found. Loading existing embeddings...")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    logger.info("FAISS index not found. Processing PDF and creating embeddings...")

    # Load and process the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    # Store embeddings in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    logger.info("FAISS index created and saved successfully.")

# Set up LLM and retriever
llm = ChatOpenAI(api_key=openai_api_key)
retriever = vectorstore.as_retriever()

# Create FastAPI app
app = FastAPI()

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
)

# API Endpoint for querying the chatbot
@app.get("/query")
async def query(text: str = Query(..., description="Your question about the Income Tax Act")):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        logger.info(f"Processing query: {text}")

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(text)

        if retrieved_docs:
            logger.info("Retrieved relevant sections from FAISS.")
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            # Generate response
            prompt = prompt_template.format(context=context, question=text)
            response = llm.invoke(prompt)
            return {"response": response.content}
        else:
            logger.warning("No relevant sections found.")
            return {"response": "No relevant information found in the Income Tax Act."}

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your query.")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Income Tax Act Chatbot API!"}

