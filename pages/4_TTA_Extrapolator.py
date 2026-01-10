import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import sys
import re
from st_copy_button import st_copy_button

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports ---
from tuning_loader import TuningData
from TTA_Extrapolator import run_tta_extrapolation

# --- Constants (Copied from main app) ---
MAP_DEFINITIONS_CSV_PATH = "map_definitions.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'
XDF_SUBFOLDER = "XDFs"
PREDEFINED_FIRMWARES = ['S50', 'A05', 'V30', 'O30', 'LB6']
ALL_FIRMWARES = PREDEFINED_FIRMWARES + ['Other']

# --- Page Configuration ---
st.set_page_config(
    page_title="TTA Extrapolator",
    layout="wide"
)

st.title("📈 TTA Extrapolator")
st.markdown("Extrapolate Torque-to-Airflow (TTA) tables to a new maximum torque and regenerate the Airflow-to-Torque (ATT) tables.")

# --- Helper Functions (Duplicated from main app to avoid refactoring risk) ---
def display_table_with_copy_button(title: str, styled_df, raw_df: pd.DataFrame):
    """
    Displays a title, a styled DataFrame with its index, and a button to copy
    the raw data (without index/header) to the clipboard.
    """
    st.write(title)
    clipboard_text = raw_df.to_csv(sep='\t', index=False, header=False)
    st.dataframe(styled_df)

    # Unique key for button
    clean_title = re.sub(r'[^a-zA-Z0-9]', '', title)
    button_key = f"copy_btn_{clean_title}_{hash(title)}"

    st_copy_button(clipboard_text, f"📋 Copy Data", key=button_key)

def style_changed_cells(new_df: pd.DataFrame, old_df: pd.DataFrame):
    """Compares two DataFrames and returns a Styler object with changed cells highlighted."""
    try:
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)
        old_df_aligned, new_df_aligned = old_df_c.align(new_df_c, join='outer', axis=None)

        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)
        increase_style = 'background-color: #2B442B'
        decrease_style = 'background-color: #442B2B'

        # Use numpy isclose for float comparison to avoid noise
        is_diff = ~np.isclose(new_df_aligned, old_df_aligned, rtol=1e-5)

        style_df[is_diff & (new_df_aligned > old_df_aligned)] = increase_style
        style_df[is_diff & (new_df_aligned < old_df_aligned)] = decrease_style

        return new_df.style.apply(lambda x: style_df, axis=None).format("{:.2f}")
    except (ValueError, TypeError):
        return new_df.style.format("{:.2f}")

@st.cache_resource(show_spinner=False)
def load_maps(bin_content, xdf_content, xdf_name, firmware_setting):
    """Loads maps from bin/xdf."""
    # Logic copied/adapted from load_all_maps_streamlit
    try:
        loader = TuningData(bin_content)
    except Exception as e:
        st.error(f"Failed to load binary: {e}")
        return None

    if xdf_content:
        tmp_xdf_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
                tmp_xdf.write(xdf_content)
                tmp_xdf_path = tmp_xdf.name

            loader.load_from_xdf(tmp_xdf_path, XDF_MAP_LIST_CSV)

        except Exception as e:
            st.error(f"Failed to parse XDF.")
            return None
        finally:
            if os.path.exists(tmp_xdf_path):
                os.remove(tmp_xdf_path)

    if firmware_setting != 'Other':
        try:
            firmware_col = f"address_{firmware_setting}"
            loader.load_from_manual_config(MAP_DEFINITIONS_CSV_PATH, firmware_col)
        except Exception as e:
            st.warning(f"Could not load manual definitions: {e}")

    return loader.maps


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    firmware = st.radio(
        "Firmware Version",
        options=ALL_FIRMWARES,
        horizontal=True,
        key="firmware"
    )

# --- Main UI ---
st.subheader("1. Upload & Configure")
col1, col2 = st.columns(2)

with col1:
    uploaded_bin_file = st.file_uploader("Upload .bin file", type=['bin', 'all'])

    uploaded_xdf_file = None
    if firmware == 'Other':
        uploaded_xdf_file = st.file_uploader("Upload .xdf file", type=['xdf'])

with col2:
    desired_max_tq = st.number_input(
        "Desired Max Torque",
        min_value=0.0,
        max_value=2000.0,
        value=500.0,
        step=10.0,
        help="The new maximum torque value for the last row of the TTA tables."
    )

if st.button("🚀 Run Extrapolation", type="primary", use_container_width=True):
    if not uploaded_bin_file:
        st.error("Please upload a BIN file.")
    elif firmware == 'Other' and not uploaded_xdf_file:
        st.error("Please upload an XDF file for 'Other' firmware.")
    else:
        with st.spinner("Loading maps..."):
            bin_content = uploaded_bin_file.getvalue()
            xdf_content = uploaded_xdf_file.getvalue() if uploaded_xdf_file else None
            xdf_name = uploaded_xdf_file.name if uploaded_xdf_file else None

            xdf_content_bytes = None
            if firmware in PREDEFINED_FIRMWARES:
                local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{firmware}.xdf")
                if os.path.exists(local_xdf_path):
                    with open(local_xdf_path, "rb") as f:
                        xdf_content_bytes = f.read()
            elif xdf_content:
                xdf_content_bytes = xdf_content

            all_maps = load_maps(bin_content, xdf_content_bytes, xdf_name, firmware)

        if all_maps:
            with st.spinner("Extrapolating..."):
                results = run_tta_extrapolation(all_maps, desired_max_tq)

            if results['status'] == 'Success':
                st.balloons()
                st.success("Extrapolation Complete!")

                # Iterate through results
                res_data = results['results']
                sorted_names = sorted(res_data.keys())

                if not sorted_names:
                    st.warning("No TTA/ATT pairs found in this file.")
                else:
                    # Display the New Torque Axis first (since it's shared)
                    # We pick the first result to show the axis
                    first_res = res_data[sorted_names[0]]
                    new_axis = first_res['new_torque_axis']
                    st.info(f"**New Torque Axis (Last Value: {new_axis[-1]})**")
                    st.code('\t'.join(map(str, new_axis)), language=None)
                    st_copy_button('\t'.join(map(str, new_axis)), "📋 Copy New Torque Axis", key="copy_axis")

                    st.divider()

                    # Create tabs for each map pair
                    tabs = st.tabs(sorted_names)
                    for i, tab in enumerate(tabs):
                        with tab:
                            map_name = sorted_names[i]
                            pair_res = res_data[map_name]

                            if 'error' in pair_res:
                                st.error(f"Error processing {map_name}: {pair_res['error']}")
                                continue

                            col_tta, col_att = st.columns(2)

                            with col_tta:
                                st.subheader("TTA (Torque -> Airflow)")
                                styled_tta = style_changed_cells(pair_res['new_tta'], pair_res['original_tta'])
                                display_table_with_copy_button(f"New TTA: {map_name}", styled_tta, pair_res['new_tta'])

                            with col_att:
                                st.subheader("ATT (Airflow -> Torque)")
                                styled_att = style_changed_cells(pair_res['new_att'], pair_res['original_att'])
                                display_table_with_copy_button(f"New ATT: {map_name}", styled_att, pair_res['new_att'])
            else:
                st.error(results.get('message', "Analysis Failed"))
