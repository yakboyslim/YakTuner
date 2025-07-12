# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/error_reporter.py

import streamlit as st
import gspread
from datetime import datetime
import json
import traceback

def send_to_google_sheets(traceback_str, user_description, user_contact):
    """
    Connects to Google Sheets and appends the error report.
    Returns a tuple: (success_boolean, message_string)
    """
    try:
        print("--- Attempting to send to Google Sheets... ---")

        # 1. Load secrets from Streamlit
        print("Step 1: Loading secrets from st.secrets...")
        if "google_sheets_creds" not in st.secrets or "creds_json" not in st.secrets["google_sheets_creds"]:
            raise KeyError("Secrets not configured correctly. 'google_sheets_creds' or 'creds_json' key is missing.")
        creds_json_str = st.secrets["google_sheets_creds"]["creds_json"]
        print("Step 1: Secrets loaded successfully.")

        # 2. Parse the JSON string
        print("Step 2: Parsing JSON credentials...")
        creds_dict = json.loads(creds_json_str)
        print("Step 2: JSON parsed successfully.")

        # 3. Authenticate with Google
        print("Step 3: Authenticating with gspread...")
        gc = gspread.service_account_from_dict(creds_dict)
        print("Step 3: Authentication successful.")

        # 4. Open the spreadsheet
        print("Step 4: Opening spreadsheet 'YAKtuner Error Reports'...")
        spreadsheet = gc.open("YAKtuner Error Reports")
        worksheet = spreadsheet.sheet1
        print("Step 4: Spreadsheet and worksheet opened successfully.")

        # 5. Append the data
        print("Step 5: Preparing and appending row...")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Truncate for logging clarity
        row_to_add = [timestamp, user_description[:50], user_contact, traceback_str[:150]]
        worksheet.append_row([timestamp, user_description, user_contact, traceback_str])
        print("Step 5: Row appended successfully.")
        print("--- Google Sheets send successful. ---")

        return True, "Report sent successfully!"

    except Exception as e:
        # This block will now catch any failure from the steps above
        error_message = f"Failed to send error report: {e}"
        print("--- ERROR IN send_to_google_sheets ---")
        print(error_message)
        # Print the full traceback to the server logs for detailed debugging
        print(traceback.format_exc())
        print("------------------------------------")
        return False, error_message