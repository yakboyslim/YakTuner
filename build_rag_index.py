import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import llama_index
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.google import GooglePairedEmbeddings
from llama_index.vector_stores.faiss import FaissVectorStore
from trafilatura import fetch_url, extract

# --- Configuration ---
URLS_TO_ADD = [
    "https://cobbtuning.atlassian.net/wiki/spaces/PRS/pages/143753246/Volkswagen+MQB+Tuning+Guide",
    "https://cobbtuning.atlassian.net/wiki/spaces/PRS/pages/725221419/VW+Reference+Torque+Set+Point+Calculations",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/21430325/VW+AG+EA888+Engine+Tuning",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/378765322/EA888+Multi-Port+Injection",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/607420417/EA888+Low+Pressure+Fuel+Pump+LPFP+Control",
    "https://ecutek.atlassian.net/wiki/spaces/SUPPORT/pages/1796866149/VW+EA888+Combustion+Modes+Configuring+MPI",
]

def load_web_documents(urls: list) -> list[Document]:
    """Scrapes content from a list of URLs and returns them as LlamaIndex Documents."""
    web_docs = []
    for url in urls:
        print(f"  -> Processing URL: {url}")
        try:
            downloaded = fetch_url(url)
            if not downloaded:
                print(f"     - Failed to download content.")
                continue

            main_content = extract(downloaded, include_comments=False, include_tables=True)
            if not main_content:
                print(f"     - Could not extract main content.")
                continue

            # Create a LlamaIndex Document object
            doc = Document(
                text=main_content,
                metadata={
                    "source_url": url,
                    "document_type": "Web Page"
                }
            )
            web_docs.append(doc)
            print(f"     - Successfully created document.")

        except Exception as e:
            print(f"     - An error occurred: {e}")

    return web_docs

# --- Main Indexing Function ---
def build_and_save_index():
    """
    Main function to build and save the LlamaIndex RAG index from local files and web pages.
    """
    # --- Configuration ---
    load_dotenv()
    PDF_PATH = "Split_Chapters"
    TXT_PATH = "Combined_Descriptions"
    INDEX_PERSIST_DIR = "./storage"
    EMBEDDING_MODEL = 'models/text-embedding-004'

    # --- API Key Check ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not found.")
        return

    try:
        genai.configure(api_key=api_key)
        print("Google Generative AI configured successfully.")
    except Exception as e:
        print(f"Error configuring Google AI: {e}")
        return

    # --- Create Dummy Directories ---
    for path in [PDF_PATH, TXT_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}. Please add your documents here.")

    # --- Load Local Documents ---
    print("\n--- Starting Document Processing with LlamaIndex ---")

    def get_file_metadata(file_path: str) -> dict:
        file_name = os.path.basename(file_path)
        chapter = file_name.split('_')[1] if '_' in file_name else "Unknown"
        doc_type = "Diagram Description" if "Combined_Descriptions" in file_path else "Technical Guide"
        return {"chapter": chapter, "document_type": doc_type, "source_filename": file_name}

    pdf_reader = SimpleDirectoryReader(PDF_PATH, file_metadata=get_file_metadata)
    txt_reader = SimpleDirectoryReader(TXT_PATH, file_metadata=get_file_metadata)

    local_documents = pdf_reader.load_data() + txt_reader.load_data()
    print(f"Successfully loaded {len(local_documents)} local documents.")

    # --- Load Web Documents ---
    print("\n--- Starting Web Content Processing ---")
    web_documents = load_web_documents(URLS_TO_ADD)
    print(f"Successfully loaded {len(web_documents)} web documents.")

    all_documents = local_documents + web_documents
    if not all_documents:
        print("\nNo documents found. Please add local files or URLs and run again.")
        return

    # --- Set up LlamaIndex Service Context ---
    print(f"\n--- Setting up Embedding Model: {EMBEDDING_MODEL} ---")
    embed_model = GooglePairedEmbeddings(
        model_name=EMBEDDING_MODEL,
        api_key=api_key,
        query_task_type="retrieval_query",
        doc_task_type="retrieval_document"
    )
    llama_index.core.Settings.embed_model = embed_model
    llama_index.core.Settings.chunk_size = 1024
    llama_index.core.Settings.chunk_overlap = 100

    # --- Build and Persist Vector Store Index ---
    print("\n--- Building and Persisting Vector Store Index ---")
    print(f"This will be saved to: {INDEX_PERSIST_DIR}")

    vector_store = FaissVectorStore.from_defaults()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        show_progress=True
    )

    index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)

    print("\n✅ LlamaIndex RAG Indexing Complete.")
    print("The knowledge base now contains content from local files and web pages.")

if __name__ == "__main__":
    build_and_save_index()
