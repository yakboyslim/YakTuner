# error_reporter.py

import streamlit as st
import gspread
from datetime import datetime
import json

def send_to_google_sheets(traceback_str, user_description, user_contact):
    """
    Connects to Google Sheets using Streamlit Secrets and appends the error report.
    """
    try:
        # --- FIX: Load and parse credentials from Streamlit Secrets ---
        creds_json_str = st.secrets["google_sheets_creds"]["creds_json"]
        creds_dict = json.loads(creds_json_str)

        # Authenticate with Google Sheets
        gc = gspread.service_account_from_dict(creds_dict)
        # --- END FIX ---

        spreadsheet = gc.open("YAKtuner Error Reports")
        worksheet = spreadsheet.sheet1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_to_add = [timestamp, user_description, user_contact, traceback_str]
        worksheet.append_row(row_to_add)
        return True

    except Exception as e:
        print(f"--- FAILED TO SEND ERROR REPORT ---")
        print(e)
        print("--- END OF REPORTING FAILURE ---")
        return False