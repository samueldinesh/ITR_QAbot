import streamlit as st
import requests
import json
import os

API_BASE_URL = "http://127.0.0.1:8000"
DATA_FILE = "data/users.json"  # File to store session data

st.set_page_config(page_title="ITR Advisor Bot", layout="wide")
st.title("ITR Advisor Bot")

# **Function to Save Data to JSON**
def save_data():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)  # Ensure directory exists
    with open(DATA_FILE, "w") as file:
        json.dump(st.session_state["users"], file)

# **Function to Load Data from JSON**
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    return {}

# **Initialize Session Storage**
if "users" not in st.session_state:
    st.session_state["users"] = load_data()
if "selected_user" not in st.session_state:
    st.session_state["selected_user"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "edit_mode" not in st.session_state:
    st.session_state["edit_mode"] = False  # Tracks if user is editing profile

# **Sidebar: Manage Users & Profile Editing**
st.sidebar.header("Manage Users")

# **Load or Switch Users**
user_list = list(st.session_state["users"].keys())
selected_user = st.sidebar.selectbox("Select User", ["New User"] + user_list)

# **Reset Chat and Load User Data when Switching Users**
if selected_user != st.session_state["selected_user"]:
    st.session_state["chat_history"] = []  # Clear chat history
    st.session_state["selected_user"] = selected_user
    st.session_state["edit_mode"] = False  # Reset edit mode when switching users

# **User Profile Section (Sidebar)**
st.sidebar.subheader("📋 User Profile")

if selected_user == "New User" or st.session_state["edit_mode"]:
    # **Edit Profile Mode**
    user_id = st.sidebar.text_input("Enter New User ID:" if selected_user == "New User" else "User ID", value=selected_user if selected_user != "New User" else "")
    user_name = st.sidebar.text_input("Enter Your Name:", value=st.session_state["users"].get(user_id, {}).get("name", "") if selected_user != "New User" else "")
    salary = st.sidebar.number_input("Enter Your Annual Salary (₹):", min_value=0.0, step=10000.0, format="%.2f", 
                                     value=st.session_state["users"].get(user_id, {}).get("salary", 0.0) if selected_user != "New User" else 0.0)
    
    # **Save or Update Profile**
    if st.sidebar.button("Save Profile"):
        if user_id and user_name:
            profile_data = {
                "user_id": user_id,
                "name": user_name,
                "salary": salary,
            }
            
            # Store locally in session
            st.session_state["users"][user_id] = profile_data
            st.session_state["selected_user"] = user_id
            st.session_state["edit_mode"] = False  # **Ensure Edit Mode is Turned Off**
            
            # Send to FastAPI
            response = requests.post(f"{API_BASE_URL}/save_user", json=profile_data)
            if response.status_code == 200:
                st.sidebar.success(f"✅ Profile saved for {user_name}!")
                save_data()  # Save to JSON
                st.rerun()  # **Force UI refresh to show consolidated profile view**
            else:
                st.sidebar.error("❌ Error saving profile.")
        else:
            st.sidebar.error("❌ Please enter a valid User ID and Name.")

else:
    # **View Profile Mode (Not Editing)**
    user_data = st.session_state["users"][selected_user]
    st.sidebar.markdown(f"""
    **👤 Name:** {user_data['name']}  
    **💰 Salary:** ₹{user_data['salary']:,}  
    """)

    if st.sidebar.button("Edit Profile"):
        st.session_state["edit_mode"] = True  # **Enable edit mode when clicked**
        st.rerun()

# **Chat Section**
st.subheader("💬 Chat with AI Tax Advisor")

if st.session_state["selected_user"] and st.session_state["selected_user"] != "New User":
    user_message = st.text_input("Ask a tax-related question:")
    col1, col2,col3 = st.columns([1,1,6])
    with col1:
        if st.button("Send"):
            with st.spinner("Thinking..."):
                chat_data = {
                    "user_id": st.session_state["selected_user"],
                    "user_name": st.session_state["users"][st.session_state["selected_user"]]["name"],
                    "user_salary": st.session_state["users"][st.session_state["selected_user"]]["salary"],
                    "message": user_message
                }
                print(chat_data)
                response = requests.post(f"{API_BASE_URL}/chat", json=chat_data)

                if response.status_code == 200:
                    reply = response.json()["response"]
                    st.session_state["chat_history"].insert(0, ("You", user_message))  # Insert new message at top
                    st.session_state["chat_history"].insert(0, ("AI", reply))  # Insert AI response at top  
    with col2:
        if st.button("Clear Chat"):
            response = requests.post(f"{API_BASE_URL}/clear_chat")
            if response.status_code == 200:
                reply = response.json()["response"]
                print(reply)
            

# **Display Chat History (Latest Messages on Top)**
st.subheader("🗨 Chat History")
for sender, msg in st.session_state["chat_history"]:
    with st.chat_message(sender):
        st.write(msg)
