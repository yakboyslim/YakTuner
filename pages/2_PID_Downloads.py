import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(page_title="PID List Downloads", layout="wide")
st.title("📄 PID List Downloads")
st.markdown("Here you can download predefined PID lists for use with your logging software.")

# --- Constants for Directories ---
PID_LISTS_DIR = "PID Lists"
FULL_LISTS_SUBDIR = "Full PID Lists"
ADDITIONS_SUBDIR = "Additions"

def create_download_section(title, directory_path):
    """
    A helper function to find CSV files in a directory and display
    download buttons for them in a structured layout.
    """
    st.subheader(title)

    if not os.path.isdir(directory_path):
        st.error(f"Directory not found: '{directory_path}'. Please check your folder structure.")
        return

    csv_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.csv')])

    if not csv_files:
        st.info("No PID lists found in this category.")
        return

    # Create a clean layout for the download links
    for file_name in csv_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            with open(file_path, "rb") as f:
                # Reading as bytes is the most robust way for st.download_button
                file_data = f.read()

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{file_name}**")
            with col2:
                st.download_button(
                    label="📥 Download",
                    data=file_data,
                    file_name=file_name,
                    mime="text/csv",
                    use_container_width=True
                )
            st.divider()
        except FileNotFoundError:
            st.warning(f"Could not read file: {file_name}")
        except Exception as e:
            st.error(f"An error occurred while processing {file_name}: {e}")

# --- Main Page Logic ---
# Create the download sections for each category
full_list_path = os.path.join(PID_LISTS_DIR, FULL_LISTS_SUBDIR)
additions_path = os.path.join(PID_LISTS_DIR, ADDITIONS_SUBDIR)

create_download_section("Full PID Lists", full_list_path)
create_download_section("PID List Additions", additions_path)
