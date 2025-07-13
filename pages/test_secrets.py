# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/test_secrets.py
import streamlit as st
import json

st.set_page_config(layout="wide")
st.title("Secrets and JSON Parsing Test")

try:
    st.header("1. Accessing Secrets")
    st.write("Attempting to access `st.secrets['google_sheets_creds']['creds_json']`...")

    # This is the line that is likely causing the silent crash.
    # If the app crashes here, the problem is 100% with the secrets file or the platform.
    creds_json_str = st.secrets["google_sheets_creds"]["creds_json"]

    st.success("SUCCESS: Successfully loaded the JSON string from secrets!")
    st.code(creds_json_str, language='json')

    st.header("2. Parsing JSON")
    st.write("Attempting to parse the loaded string with `json.loads()`...")

    creds_dict = json.loads(creds_json_str)

    st.success("SUCCESS: Successfully parsed the JSON string into a dictionary!")
    st.json(creds_dict)

except Exception as e:
    st.error("AN EXCEPTION OCCURRED!")
    st.exception(e)
