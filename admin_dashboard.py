import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- FIREBASE SETUP FOR STREAMLIT CLOUD ---
# Do not import from app.firebase_config because it expects local files.
# Instead, we initialize directly from Streamlit Secrets.

if not firebase_admin._apps:
    # Check if we are on Streamlit Cloud (secrets exist)
    if "FIREBASE_JSON" in st.secrets:
        try:
            # Parse the JSON string from secrets
            key_dict = json.loads(st.secrets["FIREBASE_JSON"])
            cred = credentials.Certificate(key_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Failed to initialize Firebase from secrets: {e}")
            st.stop()
    else:
        # Fallback for local testing (if you have local creds setup elsewhere)
        # or show a warning if no secrets found.
        st.warning("No FIREBASE_JSON found in Streamlit secrets. Trying default initialization.")
        try:
            firebase_admin.initialize_app()
        except Exception:
            pass

db = firestore.client()

# 1. Page Config
st.set_page_config(page_title="FauxLens Finance", layout="wide")
st.title("💸 FauxLens Profit & Loss")

# Authentication
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not ADMIN_PASSWORD:
    st.warning("⚠️ ADMIN_PASSWORD not set in environment variables.")
    st.stop()

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == ADMIN_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Enter Admin Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again.
        st.text_input(
            "Enter Admin Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# 2. Load Data Function
@st.cache_data(ttl=60) # Refresh every 60 seconds
def load_data():
    try:
        # Fetch from Firestore
        docs = db.collection("financial_events").stream()
        data = [d.to_dict() for d in docs]
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        # Firestore timestamps might be objects or strings depending on how they come back
        # If they are datetime objects, pd.to_datetime handles them fine
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# 3. Empty State Handling
if df.empty:
    st.info("Waiting for first transaction... No financial events found.")
    st.stop()

# 4. Sidebar Filters
st.sidebar.header("Filters")
# Date filtering could go here

# 5. Top Level KPI Cards
# Filter for Revenue (> 0) and Expense (< 0)
# Make sure 'amount' is numeric
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

total_revenue = df[df['amount'] > 0]['amount'].sum()
total_expense = df[df['amount'] < 0]['amount'].sum()
net_profit = total_revenue + total_expense # Expense is negative

col1, col2, col3 = st.columns(3)
col1.metric("💰 Revenue", f"${total_revenue:,.2f}")
col2.metric("🔥 Server Burn", f"${abs(total_expense):,.2f}")
col3.metric("📈 Net Profit", f"${net_profit:,.2f}", 
            delta=f"{net_profit:,.2f}", delta_color="normal")

# 6. Charts
st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Income Source")
    income_df = df[df['amount'] > 0]
    if not income_df.empty:
        fig_income = px.pie(income_df, values='amount', names='category', hole=0.4, title="Revenue by Category")
        st.plotly_chart(fig_income, use_container_width=True)
    else:
        st.info("No income data yet.")

with c2:
    st.subheader("Expense Breakdown")
    expense_df = df[df['amount'] < 0].copy()
    if not expense_df.empty:
        expense_df['positive_cost'] = expense_df['amount'].abs()
        fig_expense = px.bar(expense_df, x='category', y='positive_cost', color='category', title="Expenses by Category")
        st.plotly_chart(fig_expense, use_container_width=True)
    else:
        st.info("No expense data yet.")

# 7. Recent Transaction Feed
st.subheader("Live Ledger")
if 'timestamp' in df.columns:
    st.dataframe(
        df.sort_values(by="timestamp", ascending=False)[['timestamp', 'type', 'category', 'amount', 'meta']], 
        use_container_width=True
    )
else:
    st.dataframe(df, use_container_width=True)
