import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/query"

# Streamlit UI
#st.set_page_config(page_title="Income Tax AI Chatbot", layout="wide")
st.set_page_config(
    page_title="ITR QA Bot",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("ITR QA Bot")
st.markdown("Ask questions about the Indian Income Tax Act")

# User input
user_query = st.text_input("Enter your question:")

# Call API and display response
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Fetching response..."):
            response = requests.get(API_URL, params={"text": user_query})

            if response.status_code == 200:
                answer = response.json()["response"]
                st.success("✅ Answer:")
                st.write(answer)
            else:
                st.error("❌ Failed to get response. Please check the API.")

# Add a footer
st.markdown("---")
st.markdown("🤖 Powered by **LangChain + FastAPI + OpenAI** | Created by Dinesh")
