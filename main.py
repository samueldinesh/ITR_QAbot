import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set
if not openai_api_key:
    raise ValueError("ERROR: OPENAI_API_KEY is missing in .env file. Please set it before running the script.")

# Path to the Income Tax Act PDF and FAISS index
pdf_path = "storage/income_tax_act.pdf"
faiss_index_path = "faiss_index"

# Ensure PDF exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"ERROR: PDF file not found at {pdf_path}. Ensure it's placed correctly.")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Check if FAISS index already exists
if os.path.exists(faiss_index_path):
    print("FAISS index found. Loading existing embeddings...")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("FAISS index not found. Processing PDF and creating embeddings...")

    # Load and process the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    # Store embeddings in FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    print("FAISS index created and saved successfully.")

# Set up LLM and retriever
llm = ChatOpenAI(api_key=openai_api_key)
retriever = vectorstore.as_retriever()

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
)

# Test query
query = "What is Section 80C of the Income Tax Act?"
print(f"\nQuery: {query}")

# Retrieve relevant documents
retrieved_docs = retriever.get_relevant_documents(query)

if retrieved_docs:
    print("\nRetrieved relevant sections from the Income Tax Act.\n")
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    print("\nAI Response:\n", response.content)
else:
    print("\nNo relevant sections found. Try adjusting chunk sizes or document processing.\n")
