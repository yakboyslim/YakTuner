import sys
print("Python version:", sys.version)
print("--- Starting dependency import test ---")

try:
    print("Importing streamlit...")
    import streamlit
    print("  -> streamlit imported successfully.")
except ImportError as e:
    print(f"  -> FAILED to import streamlit: {e}")

try:
    print("Importing faiss...")
    import faiss
    print("  -> faiss imported successfully.")
except ImportError as e:
    print(f"  -> FAILED to import faiss: {e}")
except Exception as e:
    print(f"  -> An unexpected error occurred while importing faiss: {e}")

try:
    print("Importing fitz (PyMuPDF)...")
    import fitz
    print("  -> fitz (PyMuPDF) imported successfully.")
except ImportError as e:
    print(f"  -> FAILED to import fitz (PyMuPDF): {e}")

try:
    print("Importing google.generativeai...")
    import google.generativeai
    print("  -> google.generativeai imported successfully.")
except ImportError as e:
    print(f"  -> FAILED to import google.generativeai: {e}")

print("--- Dependency import test complete ---")