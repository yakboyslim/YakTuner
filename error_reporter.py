# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/error_reporter.py
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
        creds_json_str = st.secrets["google_sheets_creds"]["creds_json"]
        creds_dict = json.loads(creds_json_str)

        gc = gspread.service_account_from_dict(creds_dict)

        spreadsheet = gc.open("YAKtuner Error Reports")
        worksheet = spreadsheet.sheet1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_to_add = [timestamp, user_description, user_contact, traceback_str]
        worksheet.append_row(row_to_add)

        return True, "Report sent successfully!"

    except Exception as e:
        # Create a specific error message to return and log it for debugging
        error_message = f"Failed to send error report: {e}"
        print(f"--- ERROR IN GOOGLE SHEETS REPORTER: {error_message} ---")
        print(traceback.format_exc())
        return False, error_message