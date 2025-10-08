import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import faiss
import pickle
import fitz  # PyMuPDF
import re
from dotenv import load_dotenv

# --- Helper Function to Clean Text ---
def clean_text(text):
    """Cleans text by removing excessive newlines and whitespace."""
    text = re.sub(r'\s*\n\s*', '\n', text)  # Replace multiple newlines/spaces around a newline with a single newline
    text = re.sub(r'[ \t]+', ' ', text)      # Replace multiple spaces/tabs with a single space
    return text.strip()

# --- PDF and TXT Processing Functions ---
def load_and_chunk_pdfs(folder_path, chunk_size=1000, chunk_overlap=100):
    """Loads PDFs from a folder, cleans text, and splits them into chunks."""
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            doc = fitz.open(filepath)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()

            cleaned_text = clean_text(full_text)

            # Split text into chunks
            for i in range(0, len(cleaned_text), chunk_size - chunk_overlap):
                chunk = cleaned_text[i:i + chunk_size]
                all_chunks.append({
                    "source": filename,
                    "content": chunk
                })
    print(f"Loaded and chunked {len(all_chunks)} sections from PDFs in '{folder_path}'.")
    return all_chunks

def load_and_chunk_txts(folder_path, chunk_size=1000, chunk_overlap=100):
    """Loads TXT files from a folder, cleans text, and splits them into chunks."""
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                full_text = f.read()

            cleaned_text = clean_text(full_text)

            # Split text into chunks
            for i in range(0, len(cleaned_text), chunk_size - chunk_overlap):
                chunk = cleaned_text[i:i + chunk_size]
                all_chunks.append({
                    "source": filename,
                    "content": chunk
                })
    print(f"Loaded and chunked {len(all_chunks)} sections from TXT files in '{folder_path}'.")
    return all_chunks

# --- Main Indexing Function ---
def build_and_save_index():
    """
    Main function to build and save the FAISS index and text chunks.
    It requires the GOOGLE_API_KEY to be set as an environment variable.
    """
    # --- Configuration ---
    load_dotenv() # Load environment variables from .env file
    PDF_PATH = "Split_Chapters"
    TXT_PATH = "Combined_Descriptions"
    INDEX_FILE = "faiss_index.index"
    CHUNKS_FILE = "chunks.pkl"
    EMBEDDING_MODEL = 'models/text-embedding-004'

    # --- API Key Check ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        print("Please create a .env file in the same directory as this script and add your key, e.g.:")
        print("GOOGLE_API_KEY='your_api_key_here'")
        return

    try:
        genai.configure(api_key=api_key)
        print("Google Generative AI configured successfully.")
    except Exception as e:
        print(f"Error configuring Google AI: {e}")
        return

    # --- Create Dummy Directories if they don't exist ---
    if not os.path.exists(PDF_PATH):
        os.makedirs(PDF_PATH)
        print(f"Created directory: {PDF_PATH}. Please add your PDF documents here.")
    if not os.path.exists(TXT_PATH):
        os.makedirs(TXT_PATH)
        print(f"Created directory: {TXT_PATH}. Please add your text documents here.")

    # --- Load and Process Documents ---
    print("\n--- Starting Document Processing ---")
    pdf_chunks = load_and_chunk_pdfs(PDF_PATH)
    txt_chunks = load_and_chunk_txts(TXT_PATH)
    all_chunks = pdf_chunks + txt_chunks

    if not all_chunks:
        print("\nNo documents found in 'Split_Chapters' or 'Combined_Descriptions' folders.")
        print("Please add your documentation files and run the script again.")
        return

    # --- Generate Embeddings ---
    print(f"\n--- Generating Embeddings for {len(all_chunks)} Chunks ---")
    print(f"Using embedding model: {EMBEDDING_MODEL}")

    try:
        # The genai.embed_content function can handle a list of strings directly.
        content_list = [chunk['content'] for chunk in all_chunks]

        # The API has a limit on requests per minute. We'll batch the requests.
        batch_size = 100 # As per API documentation
        all_embeddings = []

        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i+batch_size]
            result = genai.embed_content(model=EMBEDDING_MODEL, content=batch, task_type="retrieval_document")
            all_embeddings.extend(result['embedding'])
            print(f"  Embedded batch {i//batch_size + 1}/{(len(content_list) + batch_size - 1)//batch_size}...")

        embeddings_np = np.array(all_embeddings, dtype='float32')
        print(f"Successfully generated {len(embeddings_np)} embeddings.")

    except Exception as e:
        print(f"\nAn error occurred during embedding generation: {e}")
        print("This could be due to an invalid API key, network issues, or API rate limits.")
        return

    # --- Build FAISS Index ---
    print("\n--- Building FAISS Index ---")
    dimension = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_np)
    print(f"FAISS index built successfully. Index contains {faiss_index.ntotal} vectors.")

    # --- Save Index and Chunks ---
    print(f"\n--- Saving Files ---")
    faiss.write_index(faiss_index, INDEX_FILE)
    print(f"Index saved to: {INDEX_FILE}")

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Chunks saved to: {CHUNKS_FILE}")

    print("\n✅ RAG Indexing Complete.")
    print("You can now run the main Yaktuner Streamlit application.")

if __name__ == "__main__":
    build_and_save_index()