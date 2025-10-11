import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import re
import traceback
import sys
from io import BytesIO

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import Core Libraries ---
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- LlamaIndex Imports ---
import llama_index
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google import GooglePairedEmbeddings

# --- Custom Module Imports ---
from tuning_loader import read_map_by_description
from xdf_parser import list_available_maps

# --- Constants ---
XDF_SUBFOLDER = "XDFs"
PREDEFINED_FIRMWARES = ['S50', 'A05', 'V30', 'O30', 'LB6']
ALL_FIRMWARES = PREDEFINED_FIRMWARES + ['Other']

# --- Page Configuration ---
st.set_page_config(page_title="Diagnostic Assistant", layout="wide")
st.title("💡 Diagnostic Assistant")
st.markdown("Ask a technical question about your tune, logs, or general ECU concepts. The assistant can use your uploaded `.bin` tune file for context.")

# --- RAG and Tool Functionality ---

# Constants for LlamaIndex RAG
INDEX_PERSIST_DIR = "./storage"
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATION_MODEL = 'gemini-1.5-pro-latest'

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_base_index():
    """
    Loads the LlamaIndex VectorStoreIndex from disk. This function is cached
    and does NOT depend on the API key, ensuring the UI always loads.
    """
    if not os.path.exists(INDEX_PERSIST_DIR):
        st.warning(f"Knowledge base not found at `{INDEX_PERSIST_DIR}`. Please run `build_rag_index.py` first.")
        return None
    try:
        vector_store = FaissVectorStore.from_persist_dir(INDEX_PERSIST_DIR)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=INDEX_PERSIST_DIR
        )
        index = VectorStoreIndex.from_storage(storage_context)
        return index
    except Exception as e:
        st.error(f"Error loading LlamaIndex knowledge base: {e}")
        return None

# --- Tool Functions (Unchanged) ---
def get_tune_data(map_description: str) -> str:
    if 'bin_content' not in st.session_state: return "Error: User has not uploaded a .bin file."
    xdf_content = _get_xdf_content()
    if not xdf_content: return "Error: Could not find the required XDF file."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
        tmp_xdf.write(xdf_content)
        tmp_xdf_path = tmp_xdf.name
    try:
        map_data = read_map_by_description(tmp_xdf_path, map_description, st.session_state.bin_content)
        if not map_data: return f"Error: Map with description '{map_description}' not found."
        return "\n".join([f"Data for map '{name}':\n{pd.DataFrame(data).to_string()}" for name, data in map_data.items()])
    finally:
        os.remove(tmp_xdf_path)

def list_available_maps_tool() -> dict:
    xdf_content = _get_xdf_content()
    if not xdf_content: return "Error: Could not find the required XDF file."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
        tmp_xdf.write(xdf_content)
        tmp_xdf_path = tmp_xdf.name
    try:
        return list_available_maps(tmp_xdf_path)
    finally:
        os.remove(tmp_xdf_path)

def _get_xdf_content():
    firmware_choice = st.session_state.get('firmware')
    if firmware_choice in PREDEFINED_FIRMWARES:
        path = os.path.join(XDF_SUBFOLDER, f"{firmware_choice}.xdf")
        if os.path.exists(path):
            with open(path, "rb") as f: return f.read()
    elif firmware_choice == 'Other' and st.session_state.get('xdf_content'):
        return st.session_state.xdf_content
    return None

def render_thinking_process(history):
    if not history:
        st.info("The assistant's thinking process will be shown here.")
        return
    # This function remains the same

# --- UI Layout ---
with st.sidebar:
    st.divider()
    st.subheader("💡 Assistant API Key")
    api_key_input = st.text_input(
        "Enter Google API Key", type="password", key="api_key_input",
        value=st.session_state.get('google_api_key', ''),
        help="Your key powers the Diagnostic Assistant and is not stored."
    )
    if api_key_input: st.session_state.google_api_key = api_key_input
    st.divider()
    st.page_link("yaktuner_streamlit.py", label="Main YAKtuner Analysis", icon="📈")
    st.divider()
    st.header("⚙️ Assistant Settings")
    firmware = st.radio(
        "Firmware Version", options=ALL_FIRMWARES, horizontal=True,
        help="Select your ECU's firmware to load correct map definitions.", key="firmware"
    )

if 'diag_chat_history' not in st.session_state: st.session_state.diag_chat_history = []
if 'diag_chat' not in st.session_state: st.session_state.diag_chat = None

st.subheader("1. Upload Tune & Log Files")
uploaded_bin_file = st.file_uploader("Upload .bin tune file", type=['bin', 'all'], key="uploaded_bin_file_assistant")
if uploaded_bin_file: st.session_state.bin_content = uploaded_bin_file.getvalue()

if firmware == 'Other':
    st.info("Please provide an XDF file for 'Other' firmware.")
    uploaded_xdf_file = st.file_uploader("Upload .xdf definition file", type=['xdf'], key="uploaded_xdf_file_assistant")
    if uploaded_xdf_file: st.session_state.xdf_content = uploaded_xdf_file.getvalue()

st.subheader("2. Ask Your Question")
user_query = st.text_input("Enter your diagnostic question:", placeholder="e.g., What does 'combmodes_MAF' control?", key="rag_query")
uploaded_diag_log = st.file_uploader("Upload a CSV data log (Optional)", type="csv", key="diag_log")

# --- Load Base Index ---
# This is safe to run on every script run because it's cached and has no dependencies on session_state.
base_index = load_base_index()

# --- Main Logic Block ---
if st.button("Get Diagnostic Answer", key="get_diag_answer", use_container_width=True):
    # --- Input Validation ---
    api_key = st.session_state.get('google_api_key')
    if not api_key: st.error("Please enter your Google API Key in the sidebar.")
    elif not uploaded_bin_file: st.error("Please upload your .bin tune file.")
    elif firmware == 'Other' and not st.session_state.get('xdf_content'): st.error("Please upload the XDF file for 'Other' firmware.")
    elif not user_query: st.warning("Please enter a question.")
    elif not base_index: st.error("Knowledge base could not be loaded. Please ensure the index has been built.")
    else:
        with st.status("Analyzing...", expanded=True) as status:
            try:
                # --- Step 1: Configure API-dependent components ---
                status.update(label="Initializing models...")
                genai.configure(api_key=api_key)

                embed_model = GooglePairedEmbeddings(
                    model_name=EMBEDDING_MODEL, api_key=api_key,
                    query_task_type="retrieval_query", doc_task_type="retrieval_document"
                )
                llama_index.core.Settings.embed_model = embed_model

                model = genai.GenerativeModel(GENERATION_MODEL, tools=[get_tune_data, list_available_maps_tool])
                query_engine = base_index.as_query_engine(similarity_top_k=5)

                # --- Step 2: Process Logs and Retrieve Context ---
                log_data_str = ""
                if uploaded_diag_log:
                    status.update(label="Processing log file...")
                    log_df = pd.read_csv(uploaded_diag_log, encoding='latin1')
                    log_data_str = f'--- **USER-UPLOADED LOG FILE DATA:**\n{log_df.to_string()}\n---'

                status.update(label="Retrieving context from knowledge base...")
                response_from_rag = query_engine.query(user_query)
                context_str = "\n\n".join([f"Source: {node.metadata.get('source_filename', 'N/A')} | Chapter: {node.metadata.get('chapter', 'N/A')}\nContent: {node.get_content()}" for node in response_from_rag.source_nodes])

                # --- Step 3: Run Chat ---
                if st.session_state.diag_chat is None:
                    st.session_state.diag_chat = model.start_chat(enable_automatic_function_calling=True)
                    st.session_state.diag_chat_history = []

                chat = st.session_state.diag_chat
                initial_prompt = f"CONTEXT FROM DOCUMENTATION:\n{context_str}\n{log_data_str}\n\nUSER'S QUESTION:\n{user_query}"

                status.update(label="Sending request to the generative model...")
                response = chat.send_message(initial_prompt)
                st.session_state.diag_chat_history = chat.history

                st.markdown("#### Assistant's Answer")
                st.info(response.text)
                status.update(label="Response received.", state="complete", expanded=False)

                with st.expander("Show Retrieved Context from Documentation"):
                    for node in response_from_rag.source_nodes:
                        st.markdown(f"**Source:** {node.metadata.get('source_filename', 'N/A')} | **Chapter:** {node.metadata.get('chapter', 'N/A')} | **Relevance:** {node.score:.2f}")
                        st.text_area("Content", node.get_content(), height=150, disabled=True, key=f"context_{node.node_id}")

            except Exception as e:
                if st.session_state.diag_chat:
                    st.session_state.diag_chat_history = st.session_state.diag_chat.history
                st.error(f"An error occurred with the generative model: {e}")
                st.session_state.diag_chat = None

st.subheader("3. Assistant's Thinking Process")
with st.expander("Show/Hide the detailed reasoning process", expanded=True):
    render_thinking_process(st.session_state.diag_chat_history)
