import subprocess
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF
from datetime import datetime

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002500-\U00002BEF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" u"\U0001f926-\U0001f937" u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" u"\u2600-\u2B55" u"\u200d" u"\u23cf" u"\u23e9" u"\u231a"
        u"\ufe0f" u"\u3030"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

working_dir = os.path.dirname(os.path.realpath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# SILENCE CHROMADB WARNINGS
os.environ["ANONYMIZED_TELEMETRY"] = "False"

app = FastAPI()

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chatbot(request: MessageRequest):
    message = request.message
    vectorstore = setup_vectorstore()
    conversational_chain = chat_chain(vectorstore)

    if contains_sensitive_topics(message):
        response = "It seems you may be asking questions outside my context, please ask questions related to K.R. Mangalam University only."
    else:
        response = conversational_chain({"question": message})["answer"]

    return {"response": response}

DEFAULT_SYSTEM_PROMPT = """You are a **specialized AI assistant** dedicated exclusively to **K.R. Mangalam University** and its services. Your responses must be **accurate, concise, and strictly based on K.R. Mangalam University's verified data**.

Your goals:
1. Quickly understand the user’s needs with **minimal follow-up questions**.
2. Provide **clear, concise, helpful answers** using K.R. Mangalam University data.
3. Maintain a **warm, professional, and empathetic tone**.

---
### **CONTEXT & RESPONSE RULES**
1. If provided context contains relevant K.R. Mangalam University info → build on it.
2. If context is empty or irrelevant → politely inform the user you can only discuss K.R. Mangalam University topics.
3. Always answer using **verified K.R. Mangalam University data** only.
"""

DEFAULT_NEGATIVE_PROMPT = """
- Do **NOT** provide any information that is **not supported by verified K.R. Mangalam University data**.
- Do **NOT** fabricate or invent university **services, features, pricing, policies, internal processes, or proprietary details**.
- Do **NOT** offer **legal, financial, medical, or other unrelated professional advice**.
- Do **NOT** use, cite, or reference **external sources, external knowledge, or outside databases** beyond the authorized context.
"""

def contains_sensitive_topics(question):
    sensitive_keywords = []
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)

# CACHE THE DATABASE SO IT DOESN'T CRASH ON REFRESH
@st.cache_resource
def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    prompt_template = f"""{system_prompt}\n{negative_prompt}\nContext (from university database):\n{{context}}\nChat History:\n{{chat_history}}\nQuestion: {{question}}\nAnswer:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(llm=llm, output_key="answer", memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, chain_type="stuff", memory=memory,
        verbose=False, return_source_documents=True, combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

st.set_page_config(page_title="K.R. Mangalam University AI", page_icon="🎓", layout="wide")

# --- GLASSMORPHISM UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #030712 !important;
        color: #e2e8f0 !important;
    }

    h1, h2, h3, .st-emotion-cache-10trblm h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
    }

    div.css-textbarboxtype {
        background-color: rgba(17, 24, 39, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 20px !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        color: #e2e8f0 !important;
        margin-bottom: 15px !important;
        transition: all 0.3s ease;
    }
    
    div.css-textbarboxtype:hover {
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15) !important;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
    }

    [data-testid="stChatMessage"] {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 1rem !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(8px) !important;
    }

    [data-testid="stChatInput"] {
        background-color: rgba(17, 24, 39, 0.9) !important;
        border-radius: 1rem !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
    }
    .stChatInput textarea { color: #ffffff !important; }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.3) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("About Bot")
    
    st.markdown("## Description")
    st.markdown("""<div class="css-textbarboxtype">An AI-powered chatbot designed to provide answers related to K.R. Mangalam University.</div>""", unsafe_allow_html=True)
    
    st.markdown("## Goals")
    st.markdown("""<div class="css-textbarboxtype">- Student Support<br>- Admissions Guidance<br>- Academic Information<br>- Campus Services<br>- Program Details</div>""", unsafe_allow_html=True)
    
    # Quick Links
    st.markdown("---")
    st.markdown("## 🔗 University Portals")
    st.markdown("""
        <div class="css-textbarboxtype" style="padding: 15px !important; text-align: center;">
            <a href="https://gu.krmangalam.edu.in/" target="_blank" style="color: #06b6d4; text-decoration: none; font-weight: bold; display: block; margin-bottom: 8px;">📚 Student ERP Login</a>
            <a href="https://www.krmangalam.edu.in/examination/" target="_blank" style="color: #06b6d4; text-decoration: none; font-weight: bold; display: block; margin-bottom: 8px;">📝 PYQ Vault (Past Papers)</a>
            <a href="https://www.krmangalam.edu.in/" target="_blank" style="color: #06b6d4; text-decoration: none; font-weight: bold; display: block;">🏛️ Official KRMU Website</a>
        </div>
    """, unsafe_allow_html=True)

    # Architecture Expander
    st.markdown("---")
    with st.expander("⚙️ System Architecture"):
        st.markdown("""
        <div style="font-size: 13px; color: #cbd5e1;">
        <b>RAG Pipeline:</b><br>
        1. <b>Ingestion:</b> PDFs are parsed and chunked.<br>
        2. <b>Vectorization:</b> <i>all-MiniLM-L6-v2</i> generates embeddings.<br>
        3. <b>Storage:</b> Vectors saved in ChromaDB.<br>
        4. <b>Retrieval:</b> Similarity search fetches top contexts.<br>
        5. <b>Generation:</b> Groq (Llama-3.3-70b) synthesizes response.
        </div>
        """, unsafe_allow_html=True)

    # Developer Signature
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #818cf8; font-size: 14px; font-weight: 600;">
            <p style="margin-bottom: 2px;">Developed by Narayan Kumar Chatry  (BACK END AND RAG DEVELOPMENT)</p>
            <p style="font-size: 12px; color: #94a3b8; margin-top: 0;">B.Tech CSE AI & ML</p>
                <p style="margin-bottom: 2px;">Developed by Vanshika Yadav       (FRONT END DEVELOPMENT)</p>
            <p style="font-size: 12px; color: #94a3b8; margin-top: 0;">B.Tech CSE AI & ML</p>
                <p style="margin-bottom: 2px;">Developed by Akash Usiyal           (Front end and research </p>
            <p style="font-size: 12px; color: #94a3b8; margin-top: 0;">B.Tech CSE AI & ML</p>
                <p style="margin-bottom: 2px;">Developed by Saham Mansoori          (BACK END AND API INTEGRATION)</p>
            <p style="font-size: 12px; color: #94a3b8; margin-top: 0;">B.Tech CSE AI & ML</p>
        </div>
    """, unsafe_allow_html=True)

    # Admin Panel
    st.markdown("---")
    st.markdown("## 🔒 Admin Panel")
    admin_password = st.text_input("Enter Admin Password", type="password")
    
    if admin_password == "krmu2024":
        st.success("Admin Access Granted")
        uploaded_file = st.file_uploader("Upload new Academic Notice (PDF)", type="pdf")
        if uploaded_file is not None:
            save_path = os.path.join("data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("🧠 The AI is reading and memorizing the new document..."):
                try:
                    subprocess.run([sys.executable, "vectorize_documents.py"], check=True)
                    st.success(f"Success! The bot has learned '{uploaded_file.name}'.")
                    st.balloons()
                    if st.button("Restart Bot to Apply Changes"):
                        st.session_state.clear()
                        st.rerun()
                except Exception as e:
                    st.error("Error updating brain. Check terminal.")
                    
    # Chat History
    st.markdown("---")
    st.markdown("## Chat History")
    
    # Auto-Welcome & Toast
    if "chat_history" not in st.session_state:
        st.toast('System Online. AI Brain Connected.', icon='🟢')
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! 👋 I am the official K.R. Mangalam University AI Assistant. How can I help you today? 🎓"}
        ]
    
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if st.button(f"Chat {idx//2 + 1}: {message['content'][:30]}...", key=f"history_{idx}"):
                st.session_state.selected_chat = idx//2
    
    # PDF Export
    st.markdown("---")
    if st.button("Export Chat to PDF"):
        if len(st.session_state.chat_history) > 1:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, "K.R. Mangalam University Chatbot - History", ln=True, align='C')
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)
                
                pdf.set_font('Arial', '', 10)
                for message in st.session_state.chat_history:
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 10, message["role"].capitalize(), ln=True)
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 10, remove_emojis(message["content"]))
                    pdf.ln(5)
                
                filename = f"krmu_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(filename)
                
                with open(filename, "rb") as f:
                    st.download_button(label="Download PDF", data=f, file_name=filename, mime="application/pdf")
                os.remove(filename)
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("No chat history to export!")
            
    # Clear Chat
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! 👋 I am the official K.R. Mangalam University AI Assistant. How can I help you today? 🎓"}
        ]
        if "conversational_chain" in st.session_state:
            st.session_state.conversational_chain.memory.clear()
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🎓 K.R. Mangalam University AI Assistant")
st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/en/5/52/K.R._Mangalam_University_logo.png" width="300">
    </div>
""", unsafe_allow_html=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
button_q = None
if col1.button("📅 How many teaching days are there?"): button_q = "How many teaching days are there?"
if col2.button("🏫 Show me the academic calendar dates"): button_q = "What are the important dates in the academic calendar?"

user_input = st.chat_input("Ask a question about K.R. Mangalam University") or button_q

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # NEW: The "Thinking" Spinner!
        with st.spinner("🧠 Searching university records..."):
            response = st.session_state.conversational_chain({"question": user_input})
            assistant_response = response["answer"]
            
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})