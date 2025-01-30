import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ITR Advisor Bot", layout="wide")
st.title("ITR Advisor Bot")

# **🔹 Initialize Session Storage**
if "users" not in st.session_state:
    st.session_state["users"] = {}
if "selected_user" not in st.session_state:
    st.session_state["selected_user"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "edit_mode" not in st.session_state:
    st.session_state["edit_mode"] = False  # Tracks if user is editing profile

# **🔹 Sidebar: Manage Users & Profile Editing**
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
    deduction_choices = st.sidebar.multiselect("Select Your Tax Deductions", 
                                               ["PPF", "EPF", "NPS", "LIC Premium", "ELSS", "Home Loan Principal", "Tuition Fees"],
                                               default=st.session_state["users"].get(user_id, {}).get("deductions", []) if selected_user != "New User" else [])
    manual_deductions = st.sidebar.number_input("Enter Additional Deductions (₹):", min_value=0.0, step=1000.0, format="%.2f", 
                                                value=st.session_state["users"].get(user_id, {}).get("manual_deductions", 0.0) if selected_user != "New User" else 0.0)
    investment_choices = st.sidebar.multiselect("Select Your Investments", 
                                                ["Fixed Deposits", "Stocks", "Mutual Funds", "Real Estate", "Gold", "Cryptocurrency"],
                                                default=st.session_state["users"].get(user_id, {}).get("investments", []) if selected_user != "New User" else [])
    manual_investments = st.sidebar.number_input("Enter Additional Investments (₹):", min_value=0.0, step=1000.0, format="%.2f", 
                                                 value=st.session_state["users"].get(user_id, {}).get("manual_investments", 0.0) if selected_user != "New User" else 0.0)

    # **Save or Update Profile**
    if st.sidebar.button("Save Profile"):
        if user_id and user_name:
            profile_data = {
                "user_id": user_id,
                "name": user_name,
                "salary": salary,
                "deductions": deduction_choices,
                "manual_deductions": manual_deductions,
                "investments": investment_choices,
                "manual_investments": manual_investments
            }
            
            # Store locally in session
            st.session_state["users"][user_id] = profile_data
            st.session_state["selected_user"] = user_id
            st.session_state["edit_mode"] = False  # **Ensure Edit Mode is Turned Off**

            # Send to FastAPI
            response = requests.post(f"{API_BASE_URL}/save_user", json=profile_data)
            if response.status_code == 200:
                st.sidebar.success(f"✅ Profile saved for {user_name}!")
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
    **📉 Deductions:** {", ".join(user_data['deductions'])}  
    **➕ Additional Deductions:** ₹{user_data['manual_deductions']:,}  
    **📈 Investments:** {", ".join(user_data['investments'])}  
    **➕ Additional Investments:** ₹{user_data['manual_investments']:,}  
    """)

    if st.sidebar.button("Edit Profile"):
        st.session_state["edit_mode"] = True  # **Enable edit mode when clicked**

# **🔹 Chat Section**
st.subheader("💬 Chat with AI Tax Advisor")

if st.session_state["selected_user"]:
    user_message = st.text_input("Ask a tax-related question:")

    if st.button("Send"):
        with st.spinner("Thinking..."):
            chat_data = {
                "user_id": st.session_state["selected_user"],
                "message": user_message
            }
            response = requests.post(f"{API_BASE_URL}/chat", json=chat_data)

            if response.status_code == 200:
                reply = response.json()["response"]
                st.session_state["chat_history"].insert(0, ("You", user_message))  # Insert new message at top
                st.session_state["chat_history"].insert(0, ("AI", reply))  # Insert AI response at top

# **🔹 Display Chat History (Latest Messages on Top)**
st.subheader("🗨 Chat History")
for sender, msg in st.session_state["chat_history"]:
    with st.chat_message(sender):
        st.write(msg)
