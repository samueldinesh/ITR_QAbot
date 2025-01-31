# 📌 ITR Advisor Bot

## 📢 About the Project
The **ITR Advisor Bot** is a Retrieval-Augmented Generation (RAG) based chatbot that helps salaried employees understand their income tax liabilities under the **Old and New Tax Regimes**. It utilizes **FastAPI** for backend services, **LangChain** for AI processing, **FAISS** for document retrieval, and **Streamlit** for a user-friendly interface.

## 🚀 Features
- 📊 **Tax Calculation:** Computes tax liability under **both tax regimes**.
- 🏦 **Personalized Tax Advisory:** Uses user profile data (salary, deductions, investments) for tailored responses.
- 📑 **RAG-based Retrieval:** Fetches relevant tax-related laws and sections using FAISS.
- 🏗 **Re-Ranking:** Ensures the most relevant tax laws appear first.
- 🧠 **Chat Memory:** Remembers user conversations for context-aware responses.
- 🖥 **Interactive UI:** Streamlit-based frontend with user profile management.

## 📂 Project Structure
```bash
ITR_BOT/
├── storage/                 # Stores PDF documents for retrieval
│   ├── income_tax_act.pdf   # Official tax act document
├── faiss_index/             # FAISS index for document retrieval
├── app.py                   # FastAPI backend for chatbot & tax calculations
├── ui.py                    # Streamlit frontend for user interaction
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API Keys)
└── README.md                # Documentation
```

## 🔧 Installation
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/ITR_BOT.git
cd ITR_BOT
```

### 2️⃣ **Set Up a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate      # For Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Set Up Environment Variables**
Create a `.env` file in the root directory and add your OpenAI API key:
```ini
OPENAI_API_KEY=your-api-key-here
```

### 5️⃣ **Run the Backend (FastAPI Server)**
```bash
uvicorn app:app --reload
```

### 6️⃣ **Run the Frontend (Streamlit UI)**
```bash
streamlit run ui.py
```

## 📝 API Endpoints
| Method | Endpoint | Description |
|--------|------------|--------------------------------|
| `POST` | `/save_user` | Saves user profile (salary, deductions, etc.) |
| `POST` | `/chat` | Retrieves relevant tax information and calculates tax liability |
| `POST` | `/clear_memory` | Clears chat memory for a user |
| `GET`  | `/` | Returns welcome message |

## 🎯 Usage Instructions
1. Open the Streamlit UI (`http://localhost:8501`)
2. **Create a user profile** by entering salary and deductions.
3. Ask tax-related queries (e.g., *"What is my tax liability for ₹18,00,000 salary?"*)
4. Get a **detailed tax breakdown** under both regimes.
5. Check **retrieved legal sections** for additional tax laws.

## 📌 Future Enhancements
- ✅ Add **support for multiple tax years**
- ✅ Implement **PDF export of tax reports**
- ✅ Add **GraphQL API for structured tax queries**

## 💡 Contributing
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature-name`)
3. **Commit changes** (`git commit -m "Added new feature"`)
4. **Push to GitHub** (`git push origin feature-name`)
5. **Create a Pull Request**

## 🛠 Tech Stack
- **Backend:** FastAPI, LangChain
- **AI Models:** OpenAI GPT-4, FAISS, SentenceTransformer
- **Frontend:** Streamlit
- **Database:** FAISS for vector retrieval
- **Deployment:** Docker, Uvicorn

## 📜 License
This project is licensed under the **MIT License**.

## 📝 Author
Developed by **[Your Name]**
For contributions, contact: [your-email@example.com]

---

🚀 **Transform your tax filing experience with ITR Advisor Bot!** 🚀
