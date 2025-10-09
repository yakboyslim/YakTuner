import os
import pickle
import requests
import faiss
import numpy as np
import google.generativeai as genai
from trafilatura import fetch_url, extract

# --- Configuration ---
# Add the websites you want to include in your knowledge base here.
URLS_TO_ADD = [
    "https://cobbtuning.atlassian.net/wiki/spaces/PRS/pages/143753246/Volkswagen+MQB+Tuning+Guide",
    "https://cobbtuning.atlassian.net/wiki/spaces/PRS/pages/725221419/VW+Reference+Torque+Set+Point+Calculations",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/21430325/VW+AG+EA888+Engine+Tuning",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/378765322/EA888+Multi-Port+Injection",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/607420417/EA888+Low+Pressure+Fuel+Pump+LPFP+Control",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/1796866149/VW+EA888+Combustion+Modes+Configuring+MPI",


    # Add more URLs here...
]

# Files for your RAG index
INDEX_FILE = "faiss_index.index"
CHUNKS_FILE = "chunks.pkl"

# Model for creating embeddings
EMBEDDING_MODEL = 'models/text-embedding-004'

# Your Google API Key (ensure it's set as an environment variable or paste it here)
# For better security, it's recommended to load from environment variables.
# from dotenv import load_dotenv
# load_dotenv()
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyB8a78H-nOmTzrFPABp4N8_BTLLN20LhFA"  # Replace with your key

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the script.")

genai.configure(api_key=GOOGLE_API_KEY)


# --- Main Script ---

def create_text_chunks(text, source_url, chunk_size=1000, chunk_overlap=100):
    """Splits a long text into smaller, overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "source": source_url,
            "content": chunk_text
        })
        start += chunk_size - chunk_overlap
    return chunks


def update_rag_index():
    """
    Scrapes websites, generates embeddings, and adds them to the existing
    FAISS index and chunks database.
    """
    print("Loading existing RAG data...")
    try:
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        print(f"Loaded {len(all_chunks)} existing chunks and an index with {faiss_index.ntotal} vectors.")
    except FileNotFoundError:
        print("Existing RAG files not found. Please run the initial build script first.")
        return

    new_chunks = []
    for url in URLS_TO_ADD:
        print(f"\nProcessing URL: {url}")
        try:
            # 1. Fetch and extract main content from the URL
            downloaded = fetch_url(url)
            if not downloaded:
                print(f"  -> Failed to download content from {url}")
                continue

            main_content = extract(downloaded, include_comments=False, include_tables=True)
            if not main_content:
                print(f"  -> Could not extract main content from {url}")
                continue

            print(f"  -> Successfully extracted content ({len(main_content)} characters).")

            # 2. Split the extracted text into chunks
            url_chunks = create_text_chunks(main_content, source_url=url)
            new_chunks.extend(url_chunks)
            print(f"  -> Created {len(url_chunks)} new text chunks.")

        except Exception as e:
            print(f"  -> An error occurred while processing {url}: {e}")

    if not new_chunks:
        print("\nNo new content was added. Exiting.")
        return

    print(f"\nGenerating embeddings for {len(new_chunks)} new chunks... (This may take a moment)")

    # 3. Generate embeddings for the new chunks in batches
    new_contents = [chunk['content'] for chunk in new_chunks]
    embedding_results = genai.embed_content(model=EMBEDDING_MODEL, content=new_contents, task_type="retrieval_document")
    new_embeddings = np.array(embedding_results['embedding'], dtype='float32')

    # 4. Add the new embeddings to the FAISS index
    print("Adding new vectors to the FAISS index...")
    faiss_index.add(new_embeddings)

    # 5. Append the new chunk metadata to the existing list
    all_chunks.extend(new_chunks)

    # 6. Save the updated index and chunks
    print("Saving updated index and chunk files...")
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print("\n--- Update Complete! ---")
    print(f"Total chunks in knowledge base: {len(all_chunks)}")
    print(f"Total vectors in FAISS index: {faiss_index.ntotal}")
    print("Your Diagnostic Assistant will now use this updated knowledge.")


if __name__ == "__main__":
    update_rag_index()