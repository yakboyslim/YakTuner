import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import re
import traceback
import sys
import difflib
from st_copy_button import st_copy_button
from io import BytesIO
from scipy import interpolate

# --- Add project root to sys.path ---
# This is a robust way to ensure that local modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --- Import the custom tuning modules ---
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import faiss
import pickle
import fitz  # PyMuPDF
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Content, Part
from tuning_loader import TuningData, read_map_by_description
from xdf_parser import list_available_maps

# --- Constants ---
XDF_SUBFOLDER = "XDFs"
PREDEFINED_FIRMWARES = ['S50', 'A05', 'V30', 'O30', 'LB6']
ALL_FIRMWARES = PREDEFINED_FIRMWARES + ['Other']

# --- Page Configuration ---
st.set_page_config(page_title="Diagnostic Assistant", layout="wide")
st.title("💡 Diagnostic Assistant")
st.markdown("Ask a technical question about your tune, logs, or general ECU concepts. The assistant can use your uploaded `.bin` tune file for context.")

# --- 4. Interactive Diagnostic Assistant (NEW SECTION) ---
# --- RAG and Tool Functionality ---

# Constants for RAG
INDEX_FILE = "faiss_index.index"
CHUNKS_FILE = "chunks.pkl"
EMBEDDING_MODEL = 'models/text-embedding-004'
# Use a model that supports tools and has a high context window
GENERATION_MODEL = 'gemini-2.5-pro'

# Load RAG data (cached)
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_rag_data():
    """Loads the FAISS index and text chunks from disk."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        st.warning(f"Knowledge base files not found (`{INDEX_FILE}`, `{CHUNKS_FILE}`). Please run `build_rag_index.py` first to enable the Diagnostic Assistant.")
        return None, None
    try:
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        return faiss_index, all_chunks
    except Exception as e:
        st.error(f"Error loading RAG data: {e}")
        return None, None

faiss_index, all_chunks = load_rag_data()

# Tool: Function to get data from the tune file
def get_tune_data(map_description: str) -> str:
    """
    Reads a specific map or table from the user's uploaded .bin file by looking
    up its definition in the corresponding .xdf file using the map's description.
    This tool reads the data on-demand.

    Args:
        map_description (str): The exact description of the map to retrieve, as found in the XDF.
                               For example: "Base table for multiplicative fuel adaptation"

    Returns:
        str: The map data as a string-formatted table, or an error message if not found.
    """
    if 'bin_content' not in st.session_state:
        return "Error: The user has not uploaded a .bin file yet. Please ask them to do so."

    xdf_content = _get_xdf_content()
    if not xdf_content:
        return "Error: Could not find the required XDF file. Please select a firmware or upload an XDF file if you are using the 'Other' option."

    bin_content = st.session_state.bin_content

    # Create temporary files to pass to the parser
    tmp_xdf_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
            tmp_xdf.write(xdf_content)
            tmp_xdf_path = tmp_xdf.name

        # Use the on-demand reader
        map_data_dict = read_map_by_description(tmp_xdf_path, map_description, bin_content)

        if not map_data_dict:
            return f"Error: Could not find or parse map with description '{map_description}'. Please try a different description."

        # Format the output for the model
        output = ""
        for name, data in map_data_dict.items():
            df = pd.DataFrame(data)
            output += f"Data for map '{name}':\n{df.to_string()}\n\n"

        return output.strip()

    except Exception as e:
        return f"An unexpected error occurred while reading the tune data: {e}"
    finally:
        if os.path.exists(tmp_xdf_path):
            os.remove(tmp_xdf_path)


def list_available_maps_tool() -> dict:
    """
    Parses the user's uploaded XDF file to find all available maps and returns
    a dictionary mapping the machine-readable map title to its full description.
    This is useful for finding the correct `map_description` to use with the
    `get_tune_data` tool.
    """
    xdf_content = _get_xdf_content()
    if not xdf_content:
        return "Error: Could not find the required XDF file. Please select a firmware or upload an XDF file if you are using the 'Other' option."

    tmp_xdf_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
            tmp_xdf.write(xdf_content)
            tmp_xdf_path = tmp_xdf.name

        return list_available_maps(tmp_xdf_path)
    except Exception as e:
        return f"An error occurred while listing available maps: {e}"
    finally:
        if os.path.exists(tmp_xdf_path):
            os.remove(tmp_xdf_path)


def _get_xdf_content():
    """
    Helper function to get the correct XDF content based on the selected firmware.
    It will use the default XDF for predefined firmwares or the user-uploaded
    file for the 'Other' option.
    Returns the content as bytes or None if not found.
    """
    firmware_choice = st.session_state.get('firmware', None)

    if firmware_choice in PREDEFINED_FIRMWARES:
        local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{firmware_choice}.xdf")
        if os.path.exists(local_xdf_path):
            with open(local_xdf_path, "rb") as f:
                return f.read()
        else:
            return None  # Or raise an error
    elif firmware_choice == 'Other':
        # For 'Other', we rely on the uploaded file being in session state
        if 'xdf_content' in st.session_state and st.session_state.xdf_content:
            return st.session_state.xdf_content
        else:
            # Check for uploaded file object if content not yet in state
            if 'uploaded_xdf_file_assistant' in st.session_state and st.session_state.uploaded_xdf_file_assistant is not None:
                xdf_content = st.session_state.uploaded_xdf_file_assistant.getvalue()
                st.session_state.xdf_content = xdf_content
                return xdf_content
    return None


def render_thinking_process(history):
    """Renders the chat history in a user-friendly format."""
    if not history:
        st.info("The assistant's thinking process will be shown here once a question is asked.")
        return

    for i, message in enumerate(history):
        if message.role == "user":
            with st.expander("Show Initial Prompt", expanded=(i == 0)):
                # The user's prompt text is in the first part.
                st.code(message.parts[0].text, language=None)

        elif message.role == "model":
            st.markdown("---")
            st.markdown("##### 🧠 Assistant's Turn")

            for part in message.parts:
                if hasattr(part, 'text') and part.text:
                    st.markdown(part.text)

                if hasattr(part, 'function_call'):
                    fc = part.function_call
                    st.markdown("##### 📞 Tool Call")
                    st.code(f"{fc.name}({dict(fc.args)})", language="python")

        elif message.role == "tool":
            st.markdown("---")
            for part in message.parts:
                if hasattr(part, 'function_response'):
                    fr = part.function_response
                    st.markdown(f"##### 🔧 Tool Output for `{fr.name}`")
                    # Use a text area for potentially long outputs
                    st.text_area("", value=str(fr.response), height=200, disabled=True, key=f"tool_{fr.name}_{i}")

# --- UI FOR THE ASSISTANT ---

# --- Sidebar ---
with st.sidebar:
    st.divider()
    st.subheader("💡 Assistant API Key")
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = ''

    api_key_input = st.text_input(
        "Enter Google API Key",
        type="password",
        key="api_key_input",
        value=st.session_state.google_api_key,
        help="Your key is used to power the Diagnostic Assistant. It is not stored or shared."
    )
    if api_key_input:
        st.session_state.google_api_key = api_key_input

    st.divider()
    st.page_link("yaktuner_streamlit.py", label="Main YAKtuner Analysis", icon="📈")

    st.divider()
    st.header("⚙️ Assistant Settings")

    # --- Firmware Selection ---
    firmware = st.radio(
        "Firmware Version",
        options=ALL_FIRMWARES,
        horizontal=True,
        help="Select your ECU's firmware. This loads the correct map definitions for the assistant.",
        key="firmware"
    )

# --- Initialize Session State for Chat ---
if 'diag_chat_history' not in st.session_state:
    st.session_state.diag_chat_history = []
if 'diag_chat' not in st.session_state:
    st.session_state.diag_chat = None

# --- Main Area for File Uploads & Chat ---
st.subheader("1. Upload Tune & Log Files")
uploaded_bin_file = st.file_uploader(
    "Upload .bin tune file",
    type=['bin', 'all'],
    key="uploaded_bin_file_assistant",
    help="Upload your tune file. The assistant needs this to answer questions about your maps."
)
if uploaded_bin_file:
    st.session_state.bin_content = uploaded_bin_file.getvalue()

uploaded_xdf_file = None
if firmware == 'Other':
    st.info("Since you selected 'Other' firmware, you must provide an XDF file.")
    uploaded_xdf_file = st.file_uploader(
        "Upload .xdf definition file",
        type=['xdf'],
        key="uploaded_xdf_file_assistant",
        help="Upload the XDF definition file corresponding to your tune."
    )
    if uploaded_xdf_file:
        st.session_state.xdf_content = uploaded_xdf_file.getvalue()

st.subheader("2. Ask Your Question")
user_query = st.text_input("Enter your diagnostic question:", placeholder="e.g., What does the 'combmodes_MAF' map control?", key="rag_query")
uploaded_diag_log = st.file_uploader("Upload a CSV data log for diagnostics (Optional)", type="csv", key="diag_log")

# --- Main Logic Block ---
if st.button("Get Diagnostic Answer", key="get_diag_answer", use_container_width=True):
    # --- Input Validation ---
    if not st.session_state.google_api_key:
        st.error("Please enter your Google API Key in the sidebar to use the assistant.")
    elif not uploaded_bin_file:
        st.error("Please upload your .bin tune file first.")
    elif firmware == 'Other' and not uploaded_xdf_file:
        st.error("Please upload the corresponding XDF file for your 'Other' firmware tune.")
    elif not user_query:
        st.warning("Please enter a question.")
    elif not faiss_index or not all_chunks:
        st.error("The knowledge base is not loaded. Cannot proceed. Please ensure the index files are present and valid.")
    else:
        # --- All checks passed, proceed with the generative AI logic ---
        with st.spinner("Calling the assistant... This may take a moment."):
            try:
                genai.configure(api_key=st.session_state.google_api_key)
                model = genai.GenerativeModel(GENERATION_MODEL, tools=[get_tune_data, list_available_maps_tool])

                log_data_str = ""
                if uploaded_diag_log is not None:
                    try:
                        log_dataframe = pd.read_csv(uploaded_diag_log, encoding='latin1')
                        log_data_str = f'--- **USER-UPLOADED LOG FILE DATA:**\n{log_dataframe.to_string()}\n---'
                    except Exception as e:
                        st.error(f"Could not read the log file: {e}")
                        log_data_str = "Error: Could not read the log file."

                query_embedding_result = genai.embed_content(model=EMBEDDING_MODEL, content=user_query, task_type="retrieval_query")
                query_embedding = query_embedding_result['embedding']

                D, I = faiss_index.search(np.array([query_embedding], dtype='float32'), k=5)
                retrieved_context = [all_chunks[i] for i in I[0]]
                context_str = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['content']}" for chunk in retrieved_context])

                if st.session_state.diag_chat is None:
                    st.info("Starting a new diagnostic session...")
                    st.session_state.diag_chat = model.start_chat(enable_automatic_function_calling=True)
                    st.session_state.diag_chat_history = []

                chat = st.session_state.diag_chat
                initial_prompt = f'''
                You are an expert automotive systems engineer and a master diagnostician for ECUs.
                Your primary goal is to provide a comprehensive and accurate answer to the user's question by acting as a detective.

                **Your Process:**
                1.  **Analyze the user's question and the provided documentation and log file data (CONTEXT) to form an initial hypothesis.**
                2.  **If you need to look up a map from the tune file, you MUST use a two-step process:**
                    a. **First, call the `list_available_maps_tool()`** to get a dictionary of all available maps.
                    b. **Second, use this dictionary to find the exact `map_description` string** for the map you need to investigate.
                    c. **Finally, call the `get_tune_data()` tool** with the precise `map_description`.
                3.  **Synthesize all the evidence.** Your final answer MUST be a synthesis of information from the documentation, the log data, and the tune data.
                4.  **Formulate your final answer ONLY when you are confident you have a complete picture.**

                **Available Tools:**
                - `list_available_maps_tool()`: Returns a dictionary mapping map titles to their full descriptions.
                - `get_tune_data(map_description: str)`: Use this to look up a specific map.

                ---
                **CONTEXT FROM DOCUMENTATION:**
                {context_str}
                {log_data_str}
                ---
                **USER'S QUESTION:**
                {user_query}
                '''
                response = chat.send_message(initial_prompt)
                st.session_state.diag_chat_history = chat.history
                answer = response.text

                st.markdown("#### Assistant's Answer")
                st.info(answer)

                with st.expander("Show Retrieved Context from Documentation"):
                    for i, chunk in enumerate(retrieved_context):
                        st.markdown(f"**Source:** {chunk['source']}")
                        st.text_area("Content", chunk['content'], height=150, disabled=True, key=f"context_{i}")

            except Exception as e:
                if "TRIGGER_TOKEN_ERROR" not in user_query:
                    if st.session_state.diag_chat:
                        st.session_state.diag_chat_history = st.session_state.diag_chat.history
                st.error(f"An error occurred with the generative model: {e}")
                st.error("The conversation history up to the point of error has been saved. You can view it in the 'Thinking Process' expander below.")
                st.session_state.diag_chat = None

# --- Display Warnings or Thinking Process ---
if not faiss_index or not all_chunks:
    st.warning("Could not load the knowledge base. The Diagnostic Assistant is unavailable.")
    st.info("Please run `build_rag_index.py` from the command line to create the necessary index files.")

st.subheader("3. Assistant's Thinking Process")
with st.expander("Show/Hide the detailed reasoning process", expanded=True):
    render_thinking_process(st.session_state.diag_chat_history)